# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 conversion, layout, and training helpers.

Ordinary 1x32 round-to-nearest quantization is intentionally delegated to
AITER's existing ``quant_mxfp4_hip`` or ``dynamic_mxfp4_quant`` implementation.
The migrated Triton kernels are used only for features those paths do not
provide: stochastic payload rounding, 2-D scales, fused layout stores, and
fused Hadamard training transforms.
"""

import functools
import random
from typing import Optional, Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.mxfp4_layout import (
    MXFP4_SCALE_KCHUNK,
    MXFP4_SCALE_STRIPE,
    MXFP4_SHUFFLE_GROUP_BYTES,
    MXFP4_SHUFFLE_TILE_ROWS,
    MXFP4_SHUFFLE_UNIT_BYTES,
    _transpose_packed_fp4_kernel,
)
from aiter.ops.triton._triton_kernels.quant.mxfp4_training import (
    _dequant_hadamard_quant_mxfp4_kernel,
    _dequant_transpose_mxfp4_kernel,
    _dual_layout_quant_mxfp4_kernel,
    _fused_hadamard_quant_mxfp4_kernel,
)
from aiter.ops.triton._triton_kernels.quant.quant_mxfp4 import (
    _convert_to_mxfp4_kernel,
)
from aiter.ops.triton.utils.shuffle import (
    shuffle_scale_gemm,
    shuffle_scale_gemm_expanded,
)
from aiter.utility.mx_types import MxScaleRoundModeInt

__all__ = [
    "convert_from_mxfp4",
    "convert_from_mxfp4_2d",
    "convert_to_mxfp4",
    "convert_to_mxfp4_2d",
    "dequant_hadamard_quant_mxfp4",
    "dequant_transpose_mxfp4",
    "dual_layout_quant_mxfp4",
    "hadamard_quant_mxfp4",
    "hadamard_transform",
    "mxfp4_data_shuffle_supported",
    "mxfp4_scale_swizzle_supported",
    "swizzle_expanded_mxfp4_scale",
    "swizzle_mxfp4_scale",
    "transpose_packed_fp4",
]


_MXFP4_BLOCK_SIZE = 32


def _validate_block_size(block_size: int) -> None:
    if block_size != _MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 block_size must be {_MXFP4_BLOCK_SIZE}, got {block_size}"
        )


def _prepare_sign_vector(
    sign_vector: torch.Tensor,
    reference: torch.Tensor,
    g: int,
) -> torch.Tensor:
    if sign_vector.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "sign_vector must have dtype float32 or bfloat16, "
            f"got {sign_vector.dtype}"
        )
    if sign_vector.device != reference.device:
        raise ValueError(
            "sign_vector must be on the same device as the input, "
            f"got {sign_vector.device} and {reference.device}"
        )
    if sign_vector.numel() != g:
        raise ValueError(f"sign_vector must have {g} elements")
    if sign_vector.dim() == 1 and sign_vector.is_contiguous():
        return sign_vector
    return sign_vector.reshape(-1).contiguous()


@functools.lru_cache(maxsize=None)
def _triton_target(device: int):
    with torch.cuda.device(device):
        return triton.runtime.driver.active.get_current_target()


def _is_gfx950(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    target = _triton_target(index)
    return (
        target is not None
        and target.backend == "hip"
        and target.arch == "gfx950"
    )


def _require_gfx950(x: torch.Tensor, feature: str) -> None:
    if not _is_gfx950(x.device):
        raise RuntimeError(
            f"{feature} requires gfx950's native E2M1 stochastic/scale "
            f"conversion instructions; got device {x.device}. Ordinary "
            "1x32 RTN quantization remains available through AITER's existing "
            "MXFP4 quantizers."
        )


def _dividing_block(dim: int, cap: int, floor: int = 1) -> int:
    """Largest power-of-two block at most ``cap`` that divides ``dim``."""
    if dim <= 0:
        raise ValueError(f"dimension must be positive, got {dim}")
    block = 1 << (min(cap, dim).bit_length() - 1)
    while block > floor and dim % block:
        block >>= 1
    return max(block, floor)


def _philox_args(
    use_sr: bool,
    philox_seed: Optional[int],
    philox_offset: Optional[int],
) -> tuple[int, int]:
    if not use_sr:
        return philox_seed or 0, philox_offset or 0
    if philox_seed is None:
        philox_seed = random.randint(0, 2**31 - 2)
    if philox_offset is None:
        philox_offset = random.randint(0, 2**31 - 2)
    return philox_seed, philox_offset


def _aiter_rtn_quant(
    data_2d: torch.Tensor,
    *,
    swizzle_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AITER-first 1x32 EVEN-scale, RNE-payload quantization.

    Scale swizzling is existing AITER functionality too.  Prefer the HIP
    quantizer's fused store when its padded physical layout is already the
    exact requested shape; otherwise compose the ordinary quantizer with
    AITER's canonical scale shuffle.
    """
    data_bf16 = (
        data_2d.to(torch.bfloat16)
        if data_2d.dtype != torch.bfloat16
        else data_2d
    )
    rows, cols = data_bf16.shape
    scale_cols = cols // 32

    # The HIP implementation is the production path in a full AITER install.
    # Triton-only deployments already have the equivalent dynamic quantizer.
    from aiter import AITER_TRITON_ONLY

    if not AITER_TRITON_ONLY:
        from aiter.ops.quant import quant_mxfp4_hip

        # quant_mxfp4_hip pads a shuffled scale image to 256x8.  Use that
        # fused path only when the padding is a no-op; the public wrapper
        # deliberately returns an exact, unpadded logical layout.
        fused_swizzle = swizzle_scale and rows % 256 == 0 and scale_cols % 8 == 0
        packed, scales = quant_mxfp4_hip(
            data_bf16,
            group_size=32,
            round_mode=MxScaleRoundModeInt.Even,
            e8m0_shuffle=fused_swizzle,
        )
    else:
        from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

        packed, scales = dynamic_mxfp4_quant(data_bf16, scaling_mode="even")

        fused_swizzle = False

    packed = packed.view(torch.uint8)
    scales = scales.view(torch.uint8)
    if swizzle_scale:
        if fused_swizzle:
            scales = scales.reshape(
                rows // MXFP4_SCALE_STRIPE,
                scale_cols * MXFP4_SCALE_STRIPE,
            )
        else:
            scales = scales[:rows, :scale_cols].contiguous()
            scales = shuffle_scale_gemm(
                scales,
                arch="gfx950",
                preshuffle_factor=MXFP4_SCALE_STRIPE,
                scale_kwidth=MXFP4_SCALE_KCHUNK,
            )
    else:
        scales = scales[:rows, :scale_cols].contiguous()
    return packed, scales


def convert_to_mxfp4(
    data_hp: torch.Tensor,
    block_size: int = 32,
    axis: int = -1,
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    swizzle_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert BF16/FP32 to packed E2M1 plus E8M0 block scales.

    The ordinary BF16, 1x32, deterministic case is an adapter over AITER's
    existing MXFP4 quantizers, including their scale-layout support.  The
    migrated Triton kernel is selected only for payload SR, FP32 input, or a
    non-standard block size.

    Scale selection is always AITER/torchao ``EVEN``. ``use_sr`` changes only
    E2M1 payload rounding and is implemented with gfx950's unbiased hardware
    stochastic conversion.
    """
    _validate_block_size(block_size)
    if data_hp.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"expected float32 or bfloat16, got {data_hp.dtype}")
    if data_hp.dim() < 2:
        raise ValueError(f"expected at least 2 dimensions, got {data_hp.dim()}")
    if axis not in (-2, -1, 0, 1):
        raise ValueError(f"axis must address one of the final two dims, got {axis}")

    transpose_axis = axis in (0, -2)
    if transpose_axis:
        data_hp = data_hp.transpose(-2, -1).contiguous()

    orig_shape = data_hp.shape
    data_2d = data_hp.reshape(-1, orig_shape[-1]).contiguous()
    M, N = data_2d.shape
    if N % block_size:
        raise ValueError(f"N={N} is not divisible by block_size={block_size}")
    if N % 2:
        raise ValueError(f"N={N} must be even for FP4 packing")

    if swizzle_scale:
        if transpose_axis:
            raise ValueError("swizzled scales cannot be transposed afterwards")
        if not mxfp4_scale_swizzle_supported(M, N // block_size):
            raise ValueError(f"scale shape ({M}, {N // block_size}) does not tile")
        _require_gfx950(data_2d, "MXFP4 scale swizzle")

    # Exact overlap with existing AITER functionality: do not launch a migrated
    # kernel merely to reproduce the ordinary 1x32 RTN format.
    if (
        data_2d.dtype == torch.bfloat16
        and block_size == 32
        and not use_sr
    ):
        fp4_packed, scales_e8m0 = _aiter_rtn_quant(
            data_2d,
            swizzle_scale=swizzle_scale,
        )
    else:
        _require_gfx950(data_2d, "MXFP4 SR/layout quantization")
        philox_seed, philox_offset = _philox_args(
            use_sr, philox_seed, philox_offset
        )
        n_scale_cols = N // block_size
        fp4_packed = torch.empty(
            (M, N // 2), dtype=torch.uint8, device=data_2d.device
        )
        scale_shape = (
            (M // MXFP4_SCALE_STRIPE, n_scale_cols * MXFP4_SCALE_STRIPE)
            if swizzle_scale
            else (M, n_scale_cols)
        )
        scales_e8m0 = torch.empty(
            scale_shape, dtype=torch.uint8, device=data_2d.device
        )
        block_m = _dividing_block(M, 64)
        block_n = _dividing_block(N, 64, floor=block_size)
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        _convert_to_mxfp4_kernel[grid](
            data_2d,
            fp4_packed,
            scales_e8m0,
            data_2d.stride(0),
            data_2d.stride(1),
            fp4_packed.stride(0),
            fp4_packed.stride(1),
            scales_e8m0.stride(0),
            scales_e8m0.stride(1),
            philox_seed,
            philox_offset,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            QUANT_BLOCK_SIZE=block_size,
            IS_2D_BLOCK=False,
            USE_SR=use_sr,
            USE_ASM=True,
            SWIZZLE_SCALE=swizzle_scale,
            NUM_SCALE_COLS=n_scale_cols,
        )

    out_shape = (*orig_shape[:-1], N // 2)
    if swizzle_scale:
        return fp4_packed.reshape(out_shape), scales_e8m0
    scale_shape = (*orig_shape[:-1], N // block_size)
    if transpose_axis:
        return (
            fp4_packed.reshape(out_shape).transpose(-2, -1).contiguous(),
            scales_e8m0.reshape(scale_shape).transpose(-2, -1).contiguous(),
        )
    return fp4_packed.reshape(out_shape), scales_e8m0.reshape(scale_shape)


def convert_to_mxfp4_2d(
    data_hp: torch.Tensor,
    block_size: int = 32,
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    shuffle_data: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize with one E8M0 scale per ``block_size x block_size`` tile."""
    _validate_block_size(block_size)
    if data_hp.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"expected float32 or bfloat16, got {data_hp.dtype}")
    if data_hp.dim() < 2:
        raise ValueError(f"expected at least 2 dimensions, got {data_hp.dim()}")
    orig_shape = data_hp.shape
    data_2d = data_hp.reshape(-1, orig_shape[-1]).contiguous()
    M, N = data_2d.shape
    if M % block_size or N % block_size:
        raise ValueError(
            f"({M}, {N}) is not a whole number of {block_size}x{block_size} tiles"
        )
    if shuffle_data and not mxfp4_data_shuffle_supported(M, N // 2):
        raise ValueError(f"packed shape ({M}, {N // 2}) cannot be shuffled")

    _require_gfx950(data_2d, "2-D MXFP4 quantization")
    philox_seed, philox_offset = _philox_args(
        use_sr, philox_seed, philox_offset
    )
    fp4_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=data_hp.device)
    scales_2d = torch.empty(
        (M // block_size, N // block_size),
        dtype=torch.uint8,
        device=data_hp.device,
    )
    block_m = _dividing_block(M, 64, floor=block_size)
    block_n = _dividing_block(N, 64, floor=block_size)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _convert_to_mxfp4_kernel[grid](
        data_2d,
        fp4_packed,
        scales_2d,
        data_2d.stride(0),
        data_2d.stride(1),
        fp4_packed.stride(0),
        fp4_packed.stride(1),
        scales_2d.stride(0),
        scales_2d.stride(1),
        philox_seed,
        philox_offset,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=True,
        USE_SR=use_sr,
        USE_ASM=True,
        SWIZZLE_SCALE=False,
        NUM_SCALE_COLS=N // block_size,
        SHUFFLE_DATA=shuffle_data,
        NUM_PACKED_COLS=N // 2,
    )
    return fp4_packed.reshape(*orig_shape[:-1], N // 2), scales_2d


def _decode_mxfp4(data_fp4: torch.Tensor) -> torch.Tensor:
    """Use AITER's canonical decoder, with an old-Torch import fallback."""
    from aiter import AITER_TRITON_ONLY

    if data_fp4.device.type != "cpu" and not AITER_TRITON_ONLY:
        try:
            from aiter.utility.fp4_utils import mxfp4_to_f32

            if hasattr(torch, "float4_e2m1fn_x2"):
                return mxfp4_to_f32(data_fp4)
        except (ImportError, ModuleNotFoundError):
            pass

    packed = data_fp4.view(torch.uint8)
    unpacked = packed.repeat_interleave(2, dim=-1)
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = (packed >> 4) & 0x0F
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    return table[unpacked.long()]


def _decode_e8m0(scales: torch.Tensor) -> torch.Tensor:
    """Use AITER's canonical decoder, retaining a standalone fallback."""
    from aiter import AITER_TRITON_ONLY

    if scales.device.type != "cpu" and not AITER_TRITON_ONLY:
        try:
            from aiter.utility.fp4_utils import e8m0_to_f32

            return e8m0_to_f32(scales)
        except (ImportError, ModuleNotFoundError):
            pass

    raw = scales.view(torch.uint8)
    decoded = torch.ldexp(
        torch.ones(raw.shape, dtype=torch.float32, device=raw.device),
        raw.to(torch.int32) - 127,
    )
    return torch.where(raw == 0xFF, torch.nan, decoded)


def convert_from_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
    axis: int = -1,
) -> torch.Tensor:
    """Dequantize packed E2M1 and 1-D E8M0 scales on CPU or GPU."""
    _validate_block_size(block_size)
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"unsupported output dtype {output_dtype}")
    if axis not in (-2, -1, 0, 1):
        raise ValueError(f"axis must address one of the final two dims, got {axis}")
    transpose_axis = axis in (0, -2)
    if transpose_axis:
        data_fp4 = data_fp4.transpose(-2, -1).contiguous()
        scales = scales.transpose(-2, -1).contiguous()

    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    scales_flat = scales.reshape(-1, scales.shape[-1])
    M, N_packed = data_flat.shape
    N = N_packed * 2
    if N % block_size:
        raise ValueError(f"logical N={N} is not divisible by block_size={block_size}")
    if scales_flat.shape != (M, N // block_size):
        raise ValueError(
            f"scale shape {tuple(scales_flat.shape)} does not match "
            f"({M}, {N // block_size})"
        )

    values = _decode_mxfp4(data_flat)
    scale_f32 = _decode_e8m0(scales_flat)
    scale_expanded = (
        scale_f32.unsqueeze(-1)
        .expand(M, N // block_size, block_size)
        .reshape(M, N)
    )
    result = (values * scale_expanded).to(output_dtype)
    out_shape = (*orig_packed_shape[:-1], N)
    if transpose_axis:
        return result.reshape(out_shape).transpose(-2, -1).contiguous()
    return result.reshape(out_shape)


def convert_from_mxfp4_2d(
    data_fp4: torch.Tensor,
    scales_2d: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    """Dequantize packed E2M1 with a 2-D E8M0 scale grid."""
    _validate_block_size(block_size)
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"unsupported output dtype {output_dtype}")
    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    M, N_packed = data_flat.shape
    N = N_packed * 2
    expected = (M // block_size, N // block_size)
    if M % block_size or N % block_size or tuple(scales_2d.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scales_2d.shape)} does not match data "
            f"({M}, {N}) with block_size={block_size}"
        )

    values = _decode_mxfp4(data_flat)
    sm, sn = expected
    scale_f32 = _decode_e8m0(scales_2d)
    scale_expanded = (
        scale_f32.view(sm, 1, sn, 1)
        .expand(sm, block_size, sn, block_size)
        .reshape(M, N)
    )
    return (values * scale_expanded).to(output_dtype).reshape(
        *orig_packed_shape[:-1], N
    )


def mxfp4_scale_swizzle_supported(rows: int, cols: int) -> bool:
    """Whether ``(rows, cols)`` exactly tiles gfx950's scale layout."""
    return rows % MXFP4_SCALE_STRIPE == 0 and cols % MXFP4_SCALE_KCHUNK == 0


def mxfp4_data_shuffle_supported(rows: int, packed_cols: int) -> bool:
    """Whether a packed tensor exactly tiles gfx950's B-operand layout."""
    return (
        rows % MXFP4_SHUFFLE_TILE_ROWS == 0
        and packed_cols % MXFP4_SHUFFLE_GROUP_BYTES == 0
    )


def _shuffle_mxfp4_data_reference(data: torch.Tensor) -> torch.Tensor:
    rows, cols = data.shape
    if not mxfp4_data_shuffle_supported(rows, cols):
        raise ValueError(f"packed shape ({rows}, {cols}) cannot be shuffled")
    original_dtype = data.dtype
    x = data.view(torch.uint8)
    x = x.view(
        rows // MXFP4_SHUFFLE_TILE_ROWS,
        MXFP4_SHUFFLE_TILE_ROWS,
        cols // MXFP4_SHUFFLE_GROUP_BYTES,
        2,
        2,
        MXFP4_SHUFFLE_UNIT_BYTES,
    )
    return x.permute(0, 2, 3, 1, 4, 5).contiguous().view(rows, cols).view(
        original_dtype
    )


def _unshuffle_mxfp4_data_reference(data: torch.Tensor) -> torch.Tensor:
    rows, cols = data.shape
    if not mxfp4_data_shuffle_supported(rows, cols):
        raise ValueError(f"packed shape ({rows}, {cols}) cannot be unshuffled")
    original_dtype = data.dtype
    x = data.view(torch.uint8)
    x = x.view(
        rows // MXFP4_SHUFFLE_TILE_ROWS,
        cols // MXFP4_SHUFFLE_GROUP_BYTES,
        2,
        MXFP4_SHUFFLE_TILE_ROWS,
        2,
        MXFP4_SHUFFLE_UNIT_BYTES,
    )
    return x.permute(0, 3, 1, 2, 4, 5).contiguous().view(rows, cols).view(
        original_dtype
    )


def transpose_packed_fp4(
    data_fp4: torch.Tensor,
    shuffle_data: bool = False,
    in_shuffled: bool = False,
) -> torch.Tensor:
    """Transpose packed FP4 ``(M, N/2) -> (N, M/2)`` without dequantizing."""
    if data_fp4.dim() != 2:
        raise ValueError(f"expected a 2-D packed tensor, got {data_fp4.dim()}-D")
    M, N_packed = data_fp4.shape
    N = N_packed * 2
    if M % 2:
        raise ValueError(f"M={M} must be even for packed transpose")
    if shuffle_data and not mxfp4_data_shuffle_supported(N, M // 2):
        raise ValueError(f"output shape ({N}, {M // 2}) cannot be shuffled")
    if in_shuffled and not mxfp4_data_shuffle_supported(M, N_packed):
        raise ValueError(f"input shape ({M}, {N_packed}) cannot be shuffled")

    if data_fp4.device.type == "cpu":
        packed = data_fp4.view(torch.uint8)
        if in_shuffled:
            packed = _unshuffle_mxfp4_data_reference(packed)
        unpacked = packed.repeat_interleave(2, dim=-1)
        unpacked[:, 0::2] = packed & 0x0F
        unpacked[:, 1::2] = (packed >> 4) & 0x0F
        transposed = unpacked.transpose(0, 1).contiguous()
        output = transposed[:, 0::2] | (transposed[:, 1::2] << 4)
        if shuffle_data:
            output = _shuffle_mxfp4_data_reference(output)
        return output

    output = torch.empty((N, M // 2), dtype=torch.uint8, device=data_fp4.device)
    # ``tl.arange`` requires power-of-two extents. The kernel masks tails, so
    # round small irregular dimensions up rather than specializing to an
    # invalid extent such as 6 or 12.
    block_m = min(32, triton.next_power_of_2(M))
    block_n_packed = min(16, triton.next_power_of_2(N_packed))
    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N_packed, block_n_packed),
    )
    _transpose_packed_fp4_kernel[grid](
        data_fp4,
        output,
        M,
        N_packed,
        data_fp4.stride(0),
        data_fp4.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N_PACKED=block_n_packed,
        SHUFFLE_DATA=shuffle_data,
        NUM_PACKED_COLS=M // 2,
        IN_SHUFFLED=in_shuffled,
    )
    return output


_HADAMARD_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _get_hadamard_matrix(g: int, device: torch.device) -> torch.Tensor:
    key = (g, device)
    if key not in _HADAMARD_CACHE:
        H = torch.tensor([[1.0]], device=device)
        while H.shape[0] < g:
            H = torch.cat(
                [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
            )
        _HADAMARD_CACHE[key] = H * (1.0 / (g**0.5))
    return _HADAMARD_CACHE[key]


def hadamard_transform(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    g: int = 64,
) -> torch.Tensor:
    """Apply a normalized blockwise random Hadamard transform."""
    if g <= 0 or g & (g - 1):
        raise ValueError(f"g={g} must be a positive power of two")
    if x.shape[-1] % g:
        raise ValueError(f"N={x.shape[-1]} is not divisible by g={g}")
    sign_vector = _prepare_sign_vector(sign_vector, x, g)

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape
    H = _get_hadamard_matrix(g, x.device)
    blocked = x_2d.float().reshape(M, N // g, g)
    blocked = blocked * sign_vector.float().reshape(1, 1, g)
    return (blocked @ H).to(x.dtype).reshape(orig_shape)


_RHT_MATRIX_ATTR = "_aiter_rht_matrix_bf16"
_RHT_MATRIX_VERSION_ATTR = "_aiter_rht_matrix_bf16_version"


def _rht_matrix_bf16(sign_vector: torch.Tensor, g: int) -> torch.Tensor:
    cached = getattr(sign_vector, _RHT_MATRIX_ATTR, None)
    cached_version = getattr(sign_vector, _RHT_MATRIX_VERSION_ATTR, None)
    if (
        cached is not None
        and cached.shape == (g, g)
        and cached.device == sign_vector.device
        and cached_version == sign_vector._version
    ):
        return cached
    matrix = (
        torch.diag(sign_vector.float())
        @ _get_hadamard_matrix(g, sign_vector.device)
    ).to(torch.bfloat16)
    setattr(sign_vector, _RHT_MATRIX_ATTR, matrix)
    setattr(sign_vector, _RHT_MATRIX_VERSION_ATTR, sign_vector._version)
    return matrix


def hadamard_quant_mxfp4(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    block_size: int = 32,
    g: int = 16,
    use_sr: bool = True,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused signed-Hadamard rotation and MXFP4 quantization."""
    _validate_block_size(block_size)
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"expected float32 or bfloat16, got {x.dtype}")
    if x.shape[-1] % g or x.shape[-1] % block_size:
        raise ValueError(
            f"N={x.shape[-1]} must be divisible by g={g} and block={block_size}"
        )
    sign_vector = _prepare_sign_vector(sign_vector, x, g)

    # The fused kernel is specialized for H16. Other group sizes retain the
    # public semantics through the reference transform plus AITER-first quant.
    if g != 16 or not _is_gfx950(x.device):
        if use_sr:
            _require_gfx950(x, "Hadamard MXFP4 stochastic quantization")
        rotated = hadamard_transform(x, sign_vector, g=g)
        return convert_to_mxfp4(
            rotated,
            block_size=block_size,
            use_sr=use_sr,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
        )

    philox_seed, philox_offset = _philox_args(
        use_sr, philox_seed, philox_offset
    )
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    fp4_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scales_e8m0 = torch.empty(
        (M, N // block_size), dtype=torch.uint8, device=x.device
    )
    block_m = _dividing_block(M, 64)
    block_n = _dividing_block(N, 64, floor=max(block_size, g))
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_hadamard_quant_mxfp4_kernel[grid](
        x_2d,
        fp4_packed,
        scales_e8m0,
        sign_vector,
        _rht_matrix_bf16(sign_vector, g),
        x_2d.stride(0),
        x_2d.stride(1),
        fp4_packed.stride(0),
        fp4_packed.stride(1),
        scales_e8m0.stride(0),
        scales_e8m0.stride(1),
        philox_seed,
        philox_offset,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=block_size,
        USE_SR=use_sr,
        USE_ASM=True,
    )
    return (
        fp4_packed.view(*orig_shape[:-1], N // 2),
        scales_e8m0.view(*orig_shape[:-1], N // block_size),
    )


def dual_layout_quant_mxfp4(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    block_size: int = 32,
    g: int = 16,
    use_sr_row: bool = True,
    use_sr_transposed: bool = True,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    swizzle_scale: bool = False,
    shuffle_col: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit both training layouts from one dense read of ``x`` on gfx950."""
    _validate_block_size(block_size)
    if x.dim() != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D tensor")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"expected float32 or bfloat16, got {x.dtype}")
    M, N = x.shape
    if M % block_size or N % block_size or M % g:
        raise ValueError(
            f"shape ({M}, {N}) must tile block={block_size} and Hadamard g={g}"
        )
    sign_vector = _prepare_sign_vector(sign_vector, x, g)

    # Preserve functionality on non-gfx950 for deterministic ordinary layouts
    # by composing the AITER-first wrappers. SR and fused GEMM layouts need the
    # native gfx950 instructions/layout consumer.
    if g != 16 or not _is_gfx950(x.device):
        if use_sr_row or use_sr_transposed or swizzle_scale or shuffle_col:
            _require_gfx950(x, "fused dual-layout MXFP4 quantization")
        row, row_s = convert_to_mxfp4(x, block_size=block_size, use_sr=False)
        col, col_s = hadamard_quant_mxfp4(
            x.t(), sign_vector, block_size=block_size, g=g, use_sr=False
        )
        return row, row_s, col, col_s

    philox_seed, philox_offset = _philox_args(
        use_sr_row or use_sr_transposed, philox_seed, philox_offset
    )
    n_scale_a, n_scale_b = N // block_size, M // block_size
    if swizzle_scale:
        if not mxfp4_scale_swizzle_supported(M, n_scale_a):
            raise ValueError(f"row scale shape ({M}, {n_scale_a}) does not tile")
        if not mxfp4_scale_swizzle_supported(N, n_scale_b):
            raise ValueError(f"column scale shape ({N}, {n_scale_b}) does not tile")
    if shuffle_col and not mxfp4_data_shuffle_supported(N, M // 2):
        raise ValueError(f"column packed shape ({N}, {M // 2}) does not tile")

    scale_a_shape = (
        (M // MXFP4_SCALE_STRIPE, n_scale_a * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (M, n_scale_a)
    )
    scale_b_shape = (
        (N // MXFP4_SCALE_STRIPE, n_scale_b * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (N, n_scale_b)
    )
    row_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    row_scales = torch.empty(scale_a_shape, dtype=torch.uint8, device=x.device)
    col_fp4 = torch.empty((N, M // 2), dtype=torch.uint8, device=x.device)
    col_scales = torch.empty(scale_b_shape, dtype=torch.uint8, device=x.device)

    block_m = _dividing_block(M, 256, floor=max(block_size, g))
    block_n = _dividing_block(N, 32, floor=block_size)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _dual_layout_quant_mxfp4_kernel[grid](
        x,
        row_fp4,
        row_scales,
        col_fp4,
        col_scales,
        sign_vector,
        _rht_matrix_bf16(sign_vector, g),
        x.stride(0),
        x.stride(1),
        row_fp4.stride(0),
        row_fp4.stride(1),
        row_scales.stride(0),
        row_scales.stride(1),
        col_fp4.stride(0),
        col_fp4.stride(1),
        col_scales.stride(0),
        col_scales.stride(1),
        philox_seed,
        philox_offset,
        philox_offset + 0x9E3779B9,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=block_size,
        USE_SR_A=use_sr_row,
        USE_SR_B=use_sr_transposed,
        USE_ASM=True,
        SWIZZLE_SCALE=swizzle_scale,
        NUM_SCALE_COLS_A=n_scale_a,
        NUM_SCALE_COLS_B=n_scale_b,
        SHUFFLE_B=shuffle_col,
        NUM_PACKED_COLS_B=M // 2,
    )
    return row_fp4, row_scales, col_fp4, col_scales


def dequant_hadamard_quant_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    sign_vector: torch.Tensor,
    block_size: int = 32,
    g: int = 16,
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    swizzle_scale: bool = False,
    shuffle_data: bool = False,
    in_scale_swizzled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused dequantize/transpose/Hadamard/requantize training transform."""
    _validate_block_size(block_size)
    if data_fp4.dim() != 2:
        raise ValueError(f"expected 2-D packed input, got {data_fp4.dim()}-D")
    M, K_packed = data_fp4.shape
    K = K_packed * 2
    if M % block_size or K % block_size or M % g:
        raise ValueError(
            f"shape ({M}, {K}) must tile block={block_size} and Hadamard g={g}"
        )
    sign_vector = _prepare_sign_vector(sign_vector, data_fp4, g)

    expected_scale_shape = (
        (M // MXFP4_SCALE_STRIPE, (K // block_size) * MXFP4_SCALE_STRIPE)
        if in_scale_swizzled
        else (M, K // block_size)
    )
    if tuple(scales.shape) != expected_scale_shape:
        layout = "swizzled" if in_scale_swizzled else "row-major"
        raise ValueError(
            f"{layout} scale shape {tuple(scales.shape)} does not match "
            f"expected {expected_scale_shape} for packed data shape "
            f"{tuple(data_fp4.shape)}"
        )
    if scales.dtype != torch.uint8:
        raise TypeError(f"scales must contain raw E8M0 bytes, got {scales.dtype}")
    if scales.device != data_fp4.device:
        raise ValueError(
            "scales must be on the same device as data_fp4, "
            f"got {scales.device} and {data_fp4.device}"
        )
    if not scales.is_contiguous():
        raise ValueError("scales must be physically contiguous")

    if g != 16 or not _is_gfx950(data_fp4.device):
        if use_sr or swizzle_scale or shuffle_data or in_scale_swizzled:
            _require_gfx950(data_fp4, "fused dequant-Hadamard MXFP4 quantization")
        dequantized_t = dequant_transpose_mxfp4(
            data_fp4, scales, block_size=block_size
        )
        return hadamard_quant_mxfp4(
            dequantized_t,
            sign_vector,
            block_size=block_size,
            g=g,
            use_sr=False,
        )

    philox_seed, philox_offset = _philox_args(
        use_sr, philox_seed, philox_offset
    )
    n_scale_cols = M // block_size
    if swizzle_scale and not mxfp4_scale_swizzle_supported(K, n_scale_cols):
        raise ValueError(f"output scale shape ({K}, {n_scale_cols}) does not tile")
    if in_scale_swizzled and not mxfp4_scale_swizzle_supported(
        M, K // block_size
    ):
        raise ValueError(f"input scale shape ({M}, {K // block_size}) does not tile")
    if shuffle_data and not mxfp4_data_shuffle_supported(K, M // 2):
        raise ValueError(f"output packed shape ({K}, {M // 2}) does not tile")

    out = torch.empty((K, M // 2), dtype=torch.uint8, device=data_fp4.device)
    out_scale_shape = (
        (K // MXFP4_SCALE_STRIPE, n_scale_cols * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (K, n_scale_cols)
    )
    out_scales = torch.empty(
        out_scale_shape, dtype=torch.uint8, device=data_fp4.device
    )
    block_m = _dividing_block(M, 128, floor=max(block_size, g))
    block_k = _dividing_block(K, 64, floor=block_size)
    grid = (triton.cdiv(M, block_m), triton.cdiv(K, block_k))
    _dequant_hadamard_quant_mxfp4_kernel[grid](
        data_fp4,
        scales,
        out,
        out_scales,
        _rht_matrix_bf16(sign_vector, g),
        data_fp4.stride(0),
        data_fp4.stride(1),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        out.stride(1),
        out_scales.stride(0),
        out_scales.stride(1),
        philox_seed,
        philox_offset,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        QUANT_BLOCK_SIZE=block_size,
        USE_SR=use_sr,
        USE_ASM=True,
        SWIZZLE_SCALE=swizzle_scale,
        NUM_SCALE_COLS=n_scale_cols,
        SHUFFLE_DATA=shuffle_data,
        NUM_PACKED_COLS=M // 2,
        IN_SCALE_SWIZZLED=in_scale_swizzled,
        NUM_IN_SCALE_COLS=K // block_size,
    )
    return out, out_scales


def dequant_transpose_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Fused packed-MXFP4 dequantization and transpose to BF16."""
    _validate_block_size(block_size)
    if data_fp4.device.type == "cpu":
        return convert_from_mxfp4(
            data_fp4, scales, output_dtype=torch.bfloat16, block_size=block_size
        ).transpose(-2, -1).contiguous()

    orig_packed_shape = data_fp4.shape
    data_flat = data_fp4.reshape(-1, orig_packed_shape[-1])
    scales_flat = scales.reshape(-1, scales.shape[-1])
    M, K_packed = data_flat.shape
    K = K_packed * 2
    if K % block_size or scales_flat.shape != (M, K // block_size):
        raise ValueError("scales do not match packed data and block_size")

    output = torch.empty((K, M), dtype=torch.bfloat16, device=data_fp4.device)
    # M need not be a power of two; the kernel masks its row tail.
    block_m = min(32, triton.next_power_of_2(M))
    block_k = max(min(64, K), block_size)
    grid = (triton.cdiv(M, block_m), triton.cdiv(K, block_k))
    _dequant_transpose_mxfp4_kernel[grid](
        data_flat,
        scales_flat,
        output,
        M,
        K,
        data_flat.stride(0),
        data_flat.stride(1),
        scales_flat.stride(0),
        scales_flat.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        QUANT_BLOCK_SIZE=block_size,
    )
    return output


def swizzle_mxfp4_scale(scales: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper for gfx950's canonical GEMM scale shuffle."""
    return shuffle_scale_gemm(
        scales,
        arch="gfx950",
        preshuffle_factor=MXFP4_SCALE_STRIPE,
        scale_kwidth=MXFP4_SCALE_KCHUNK,
    )


def swizzle_expanded_mxfp4_scale(
    scales_2d: torch.Tensor,
    block_size: int = 32,
    transpose: bool = False,
) -> torch.Tensor:
    """Compatibility wrapper for canonical expanded GEMM scale shuffle."""
    _validate_block_size(block_size)
    return shuffle_scale_gemm_expanded(
        scales_2d,
        block_size=block_size,
        transpose=transpose,
        arch="gfx950",
        preshuffle_factor=MXFP4_SCALE_STRIPE,
        scale_kwidth=MXFP4_SCALE_KCHUNK,
    )
