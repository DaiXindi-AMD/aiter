import torch
import triton

from aiter.ops.triton._triton_kernels.quant.mxfp4_layout import (
    MXFP4_SCALE_KCHUNK,
    MXFP4_SCALE_STRIPE,
    _swizzle_expanded_2d_scale_kernel,
    _swizzle_mxfp4_scale_gfx950_kernel,
)


def _resolve_arch(arch):
    """Resolve the current GPU architecture only when the caller needs it."""
    if arch is not None:
        return arch

    # Importing arch_info probes the active Triton/JAX GPU backend. Keep that
    # probe out of the module import path so explicit-arch CPU references work
    # on build and test hosts without an accelerator.
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    return get_arch()

# =============================================================================
# WEIGHTS
# =============================================================================


def _shuffle_weight_gfx1250(w: torch.Tensor) -> torch.Tensor:
    """gfx1250 WMMA weight preshuffle
    Callers wanting the flattened (N//16, K*16) / transposed (E, K*16, N//16) TDM
    view reshape it.
    """
    x_type = w.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        w = w.view(torch.uint8)
    if w.ndim == 2:
        N, K = w.shape
        assert N % 16 == 0, f"N={N} must be divisible by 16"
        assert K % 32 == 0, f"K={K} must be divisible by 32"
        w = w.view(N // 16, 16, K // 32, 2, 16)
        w = w.permute(0, 2, 3, 1, 4).contiguous()
        w = w.view(N, K)
    elif w.ndim == 3:
        E, K, N = w.shape
        assert K % 32 == 0, f"K={K} must be divisible by 32"
        assert N % 16 == 0, f"N={N} must be divisible by 16"
        w = w.transpose(-1, -2)  # (E, N, K)
        w = w.view(E, N // 16, 16, K // 32, 2, 16)
        w = w.permute(0, 1, 3, 4, 2, 5).contiguous()
        w = w.view(E, K, N)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {w.ndim}D")
    w = w.view(x_type)
    w.is_shuffled = True
    return w


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4=False,
    is_guinterleave=False,
    gate_up: bool = False,
    pad_k_to: int = 0,
    arch=None,
) -> torch.Tensor:
    """Arch-aware weight preshuffle.

    On gfx1250 the WMMA TDM layout (``_shuffle_weight_gfx1250``) is used; on every
    other arch this delegates to the base ``aiter.ops.shuffle.shuffle_weight``.
    """
    if _resolve_arch(arch) == "gfx1250":
        if use_int4 or is_guinterleave or gate_up or pad_k_to:
            raise NotImplementedError(
                "shuffle_weight on gfx1250 does not support use_int4 / is_guinterleave / gate_up / pad_k_to "
            )
        return _shuffle_weight_gfx1250(x)

    from aiter.ops.shuffle import shuffle_weight as shuffle_weight_base

    return shuffle_weight_base(
        x,
        layout=layout,
        use_int4=use_int4,
        is_guinterleave=is_guinterleave,
        gate_up=gate_up,
        pad_k_to=pad_k_to,
    )


def moe_weight_decode_view(w: torch.Tensor) -> torch.Tensor:
    """zero-copy fn
    Input: ``(E, N, K)`` (K byte-packed for mxfp4). Output shares storage.
    """
    w_u8 = w if w.dtype == torch.uint8 else w.view(torch.uint8)
    E, N, K = w_u8.shape
    assert N % 16 == 0, f"N={N} must be divisible by 16"
    return w_u8.view(E, N // 16, K * 16).transpose(-1, -2)


# =============================================================================
# SCALES
# =============================================================================


# --- shared gfx1250 scale tile (GEMM + MoE) ---
def _shuffle_scale_tile_gfx1250(scales, preshuffle_factor, scale_kwidth):
    """Shared gfx1250 scale tile-permute over the last two dims.

    row = the output M/N axis, packed into stripes of ``preshuffle_factor`` lanes
    col = the scale-K axis (K_groups / K_SCALE), packed into ``scale_kwidth`` groups

    Shared by the GEMM ((M, K_groups)) and MoE ((E, N, K_SCALE), transposed) gfx1250 scale shuffles.
    """
    # rows and cols grab the last two dims, and *batch collects everything before them into a list (possibly empty)
    *batch, rows, cols = scales.shape
    num_stripes = rows // preshuffle_factor
    num_kchunks = cols // scale_kwidth
    x = scales.reshape(-1, rows, cols)  # fold batch/expert dims into one axis
    x = x.view(-1, num_stripes, preshuffle_factor, num_kchunks, scale_kwidth)
    x = x.permute(0, 1, 3, 2, 4).contiguous()  # swap lanes <-> k-chunks
    out = x.view(-1, num_stripes, cols * preshuffle_factor)
    return out.reshape(*batch, num_stripes, cols * preshuffle_factor)


# --- shared gfx950 scale tile (GEMM + MoE) ---
def _shuffle_scale_tile_gfx950(scales, preshuffle_factor, scale_kwidth):
    """Shared gfx950 (CDNA4) scale tile-permute over the last two dims.

    row = the output M/N axis, packed into stripes of ``preshuffle_factor`` lanes (split 2 x preshuffle_factor//2)
    col = the scale-K axis (K_groups / K_SCALE), packed into ``scale_kwidth`` groups (split 2 x scale_kwidth//2)

    Shared by the GEMM ((M, K_groups)) and MoE ((E, N, K_SCALE), transposed) gfx950 scale shuffles.
    """
    # rows and cols grab the last two dims, and *batch collects everything before them into a list (possibly empty)
    *batch, rows, cols = scales.shape
    num_stripes = rows // preshuffle_factor
    num_kchunks = cols // scale_kwidth
    x = scales.reshape(-1, rows, cols)  # fold batch/expert dims into one axis
    x = x.view(
        -1, num_stripes, 2, preshuffle_factor // 2, num_kchunks, 2, scale_kwidth // 2, 1
    )
    x = x.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    out = x.view(-1, num_stripes, cols * preshuffle_factor)
    return out.reshape(*batch, num_stripes, cols * preshuffle_factor)


# --- GEMM scales (afp4wfp4) ---


def shuffle_scale_gemm(
    scales: torch.Tensor,
    arch=None,
    preshuffle_factor: int = 16,
    scale_kwidth: int = 4,
) -> torch.Tensor:
    """Arch-aware GEMM scale shuffle.

    Inverse: ``unshuffle_scale_gemm`` (gfx950 only).
    gfx950: preshuffle_factor = 32, scale_kwidth = 8
    gfx1250: preshuffle_factor = 16, scale_kwidth = 4
    """
    arch = _resolve_arch(arch)
    if arch == "gfx1250":
        return _shuffle_scale_tile_gfx1250(scales, preshuffle_factor, scale_kwidth)

    if arch == "gfx950":
        if (
            scales.device.type == "cuda"
            and scales.dim() == 2
            and scales.is_contiguous()
            and preshuffle_factor == MXFP4_SCALE_STRIPE
            and scale_kwidth == MXFP4_SCALE_KCHUNK
        ):
            rows, cols = scales.shape
            if rows % preshuffle_factor or cols % scale_kwidth:
                raise ValueError(
                    f"scale shape ({rows}, {cols}) must be divisible by "
                    f"({preshuffle_factor}, {scale_kwidth})"
                )
            out = torch.empty(
                (rows // preshuffle_factor, cols * preshuffle_factor),
                dtype=scales.dtype,
                device=scales.device,
            )
            num_kchunks = cols // scale_kwidth
            block_k = min(8, triton.next_power_of_2(num_kchunks))
            grid = (rows // preshuffle_factor, triton.cdiv(num_kchunks, block_k))
            _swizzle_mxfp4_scale_gfx950_kernel[grid](
                scales,
                out,
                cols,
                scales.stride(0),
                STRIPE=preshuffle_factor,
                KCHUNK=scale_kwidth,
                BLOCK_K=block_k,
            )
            return out
        return _shuffle_scale_tile_gfx950(scales, preshuffle_factor, scale_kwidth)
    raise ValueError(f"Unsupported arch: {arch}")


def shuffle_scale_gemm_expanded(
    scales: torch.Tensor,
    block_size: int = 32,
    transpose: bool = False,
    arch=None,
    preshuffle_factor: int = 16,
    scale_kwidth: int = 4,
) -> torch.Tensor:
    """Expand a 2-D scale grid by block rows and emit GEMM scale layout.

    The gfx950 ``32 x 8`` case uses a fused Triton expand-and-shuffle kernel
    for contiguous GPU inputs. Other inputs use the canonical tensor-layout
    reference, which also makes explicit-architecture CPU tests possible.
    """
    if scales.dim() != 2:
        raise ValueError(f"expected 2-D scales, got {scales.dim()}-D")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    arch = _resolve_arch(arch)
    tile_rows, tile_cols = scales.shape
    if transpose:
        tile_rows, tile_cols = tile_cols, tile_rows
    rows, cols = tile_rows * block_size, tile_cols
    if rows % preshuffle_factor or cols % scale_kwidth:
        raise ValueError(
            f"expanded scale shape ({rows}, {cols}) must be divisible by "
            f"({preshuffle_factor}, {scale_kwidth})"
        )

    if (
        arch == "gfx950"
        and scales.device.type == "cuda"
        and scales.is_contiguous()
        and preshuffle_factor == MXFP4_SCALE_STRIPE
        and scale_kwidth == MXFP4_SCALE_KCHUNK
    ):
        out = torch.empty(
            (rows // preshuffle_factor, cols * preshuffle_factor),
            dtype=scales.dtype,
            device=scales.device,
        )
        num_kchunks = cols // scale_kwidth
        block_k = min(8, triton.next_power_of_2(num_kchunks))
        grid = (rows // preshuffle_factor, triton.cdiv(num_kchunks, block_k))
        stride_tile_row, stride_tile_col = scales.stride()
        if transpose:
            stride_tile_row, stride_tile_col = stride_tile_col, stride_tile_row
        _swizzle_expanded_2d_scale_kernel[grid](
            scales,
            out,
            cols,
            stride_tile_row,
            stride_tile_col,
            QUANT_BLOCK_SIZE=block_size,
            STRIPE=preshuffle_factor,
            KCHUNK=scale_kwidth,
            BLOCK_K=block_k,
        )
        return out

    grid = scales.transpose(0, 1) if transpose else scales
    expanded = grid.repeat_interleave(block_size, dim=0).contiguous()
    return shuffle_scale_gemm(
        expanded,
        arch=arch,
        preshuffle_factor=preshuffle_factor,
        scale_kwidth=scale_kwidth,
    )


def unshuffle_scale_gemm(scales_shuffled: torch.Tensor, arch=None) -> torch.Tensor:
    """Inverse of ``shuffle_scale_gemm`` (gfx950 layout). gfx1250 has no consumer."""
    if _resolve_arch(arch) == "gfx1250":
        raise NotImplementedError("unshuffle_scale_gemm is not implemented for gfx1250")
    scales = scales_shuffled.clone()
    sm, sn = scales.shape
    scales = scales.view(sm * 32, sn // 32)
    sm, sn = scales.shape
    scales = scales.view(sm // 32, sn // 8, 4, 16, 2, 2, 1)
    scales = scales.permute(0, 5, 3, 1, 4, 2, 6).contiguous()
    scales = scales.view(sm, sn)
    return scales


# --- MoE MX scales (a8w4 / a8w8 / a16w4 / a4w4) ---
def shuffle_scale_moe(
    data: torch.Tensor,
    arch=None,
    preshuffle_factor: int = 32,
    scale_kwidth: int = 8,
    return_layout: bool = False,
):
    """Arch-aware MoE scale shuffle (a8w4 / a8w8 / a16w4 / a4w4 family).

    gfx950 / gfx1250: preshuffle_factor = 32, scale_kwidth = 8.

    With ``return_layout=True`` also returns the matching ``SWIZZLE_MX_SCALE``
    label ("GFX1250_SCALE" for gfx1250, "CDNA4_SCALE" for gfx950) as
    ``(scale, label)``, so callers stay arch-agnostic; otherwise returns just
    the shuffled scale tensor.

    On archs with no native preshuffled MX scale layout (i.e. not gfx950 /
    gfx1250, e.g. gfx942), this is a no-op: returns ``data`` unchanged and
    ``layout=None``, so callers can invoke this unconditionally without an
    arch check of their own.
    """
    arch = _resolve_arch(arch)
    if arch == "gfx1250":
        tiled = _shuffle_scale_tile_gfx1250(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
        layout = "GFX1250_SCALE"
    elif arch == "gfx950":
        tiled = _shuffle_scale_tile_gfx950(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
        layout = "CDNA4_SCALE"
    else:
        return (data, None) if return_layout else data
    scale = tiled.transpose(-1, -2)
    return (scale, layout) if return_layout else scale


# --- batched scales (FP4 blockscale16, attention) ---
def shuffle_scale_batched(data: torch.Tensor, scale_k_width=None) -> torch.Tensor:
    """Batched shuffle scales for the FP4 blockscale16 format.

    Single-layout permute, no arch branch: the blockscale16 layout is
    arch-independent and is consumed by the FP4 MLA KV-cache path on both gfx950
    and gfx1250
    https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py#L1014
    """
    data_shape = data.shape
    N = data_shape[-2]
    SCALE_K = data_shape[-1]
    PRESHUFFLE_FACTOR = 128
    if scale_k_width is None:
        SCALE_KWIDTH = (
            min(16, 1 << (SCALE_K - 1).bit_length()) if SCALE_K >= 4 else SCALE_K
        )
    else:
        assert scale_k_width in [4, 8, 16]
        SCALE_KWIDTH = scale_k_width if SCALE_K >= 4 else SCALE_K
    data = data.view(
        -1,
        N // PRESHUFFLE_FACTOR,
        4,
        PRESHUFFLE_FACTOR // 4,
        SCALE_K // SCALE_KWIDTH,
        SCALE_KWIDTH,
    )
    data = data.permute(0, 1, 4, 3, 2, 5).contiguous()
    data = data.view(
        *data_shape[:-2], N // PRESHUFFLE_FACTOR, SCALE_K * PRESHUFFLE_FACTOR
    )
    return data
