from typing import Literal

import torch
import triton

import aiter
from aiter.ops.triton._triton_kernels.activation import (
    _act_mul_and_dynamic_fp8_group_quant_kernel,
    _act_mul_and_dynamic_mxfp4_quant_kernel,
    _glu_bwd_kernel,
    _glu_fwd_kernel,
    fused_silu_mul_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def act_mul_and_mxfp4_quant(
    x: torch.Tensor,
    activation: Literal["silu", "gelu", "gelu_tanh"],
    scaling_mode: str = "even",
    shuffle: bool = False,
    scale_shuffle_padding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the activation function and quantize the result to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        activation: activation function to apply before quantization.
            - It splits the features into two parts and applies the activation to the first part.
            - Then, it adds the results together before quantization.
            - Supports the following activations:
                - "silu"
                - "gelu"
                - "gelu_tanh"

        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
        shuffle: Indicates whether to enable preshuffling of scales.
            - When enabled, scale dimensions (X, Y) are adjusted to be multiples of 8 and 256, respectively.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    _LOGGER.info(f"ACT_MUL_MXFP4_QUANT: x={tuple(x.shape)} activation={activation}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape
    # Activation (N/2) and storing results in uint8 (N/2) results in a feature dimension of N/4
    assert N % 4 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    MXFP4_QUANT_BLOCK_SIZE = 32
    N_half = N // 2
    x_fp4 = torch.empty((M, N_half // 2), dtype=torch.uint8, device=x.device)
    scaleN_valid = triton.cdiv(N_half, MXFP4_QUANT_BLOCK_SIZE)
    # Setting scale M to be multiple of 256 and scale N to be multiple of 8
    use_scale_shuffle_padding = shuffle or scale_shuffle_padding
    if use_scale_shuffle_padding:
        scaleM = triton.cdiv(M, 256) * 256
        scaleN = triton.cdiv(scaleN_valid, 8) * 8
    else:
        scaleM = M
        scaleN = scaleN_valid
    blockscale_e8m0 = torch.empty(
        (scaleM, scaleN),
        dtype=torch.uint8,
        device=x.device,
    )

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))
        BLOCK_SIZE_N = 128
        NUM_WARPS = 1 if BLOCK_SIZE_M < 4 else 4
        NUM_STAGES = 1
    else:
        NUM_ITER = 1
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 256
        NUM_WARPS = 4
        NUM_STAGES = 1

    # for small N values
    if N_half <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N_half))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(N_half))

    # shuffle requires block sizes to be multiple of 32
    if shuffle:
        BLOCK_SIZE_M = triton.cdiv(BLOCK_SIZE_M, 32) * 32
        BLOCK_SIZE_N = triton.cdiv(BLOCK_SIZE_N, 32) * 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N_half, BLOCK_SIZE_N * NUM_ITER),
    )
    _act_mul_and_dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N_half,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        ACTIVATION=activation,
        scaleN=scaleN_valid,
        scaleM_pad=(scaleM if use_scale_shuffle_padding else 1),
        scaleN_pad=scaleN,
        SHUFFLE=shuffle,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return x_fp4, blockscale_e8m0


def act_mul_and_fp8_group_quant(
    x: torch.Tensor,
    activation: Literal["silu", "gelu", "gelu_tanh"],
    group_size,
    dtype_quant=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the activation function and quantize the result to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        activation: activation function to apply before quantization.
            - It splits the features into two parts and applies the activation to the first part.
            - Then, it adds the results together before quantization.
            - Supports the following activations:
                - "silu"
                - "gelu"
                - "gelu_tanh"

        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
        shuffle: Indicates whether to enable preshuffling of scales.
            - When enabled, scale dimensions (X, Y) are adjusted to be multiples of 8 and 256, respectively.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    if dtype_quant is None:
        dtype_quant = aiter.dtypes.fp8

    _LOGGER.info(f"ACT_MUL_FP8_GROUP_QUANT: x={tuple(x.shape)} activation={activation}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape
    assert N % 2 == 0

    N_half = N // 2
    scaleN = triton.cdiv(N, group_size)
    x_fp8 = torch.empty((M, N_half), dtype=dtype_quant, device=x.device)
    out_bs = torch.empty(
        (M, triton.cdiv(N_half, group_size)), dtype=torch.float32, device=x.device
    )

    DTYPE_MAX = (
        torch.finfo(x_fp8.dtype).max
        if torch.is_floating_point(x_fp8)
        else torch.iinfo(x_fp8.dtype).max
    )
    BLOCK_SIZE_N = group_size

    grid = (
        M,
        triton.cdiv(N_half, BLOCK_SIZE_N),
    )
    _act_mul_and_dynamic_fp8_group_quant_kernel[grid](
        x,
        x_fp8,
        out_bs,
        *x.stride(),
        *x_fp8.stride(),
        *out_bs.stride(),
        N=N_half,
        ACTIVATION=activation,
        scaleN=scaleN,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=group_size,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        # num_warps=NUM_WARPS,
        # waves_per_eu=0,
        # num_stages=1,
    )

    return x_fp8, out_bs


def fused_silu_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused SiLU-and-mul along the last dimension (same pattern as MoE silu-fused GEMM).

    ``x`` must be contiguous with positive, even ``size(-1)``. For last size
    ``2 * d``, the first ``d`` lanes are passed through SiLU (``_silu_exp2``);
    the second ``d`` lanes are the multipliers. Output shape matches ``x``
    except ``out.size(-1) == d``.

    Returns:
        ``out`` if provided, else a newly allocated tensor.
    """

    def _pick_block_n(d: int, n_rows: int) -> int:
        """Tile size along the reduced last dim (cap 1024); at least 32 for vectorization.

        Tuned on ROCm for MoE TP4 locals (GLM-4.7 ``d=384``, Kimi-K2.5 ``d=512``) and wide
        MoE activations: ``n_rows`` selects decode vs prefill N-tiling (see sweep in repo
        history / ``bench_moe.py -bench_silu_mul``).
        """
        n = max(d, 1)
        # Kimi-K2.5 TP4 (d=512): prefill favors one 512-wide N tile; decode keeps 256×2.
        if n == 512:
            return 512 if n_rows > 4096 else 256
        # GLM-4.7 TP4 (d=384): wider decode rows use 256×2; larger batches favor 128×3 N tiles.
        if n == 384:
            return 256 if n_rows <= 128 else 128
        upper = min(n, 1024)
        p = 1
        while p * 2 <= upper:
            p *= 2
        return max(32, p)

    def _pick_block_m(n_rows: int, block_n: int, d: int) -> int:
        """Row tile size: latency shapes use wide M tiles; prefill uses tuned (d, n_rows) pairs."""
        if n_rows <= 64:
            return min(32, max(4, triton.next_power_of_2(n_rows)))
        if d == 384 and n_rows > 128:
            return 32 if n_rows > 8192 else 8
        if d == 512 and n_rows > 4096:
            return 8
        if d == 512 and 128 < n_rows <= 4096:
            return 8
        if block_n >= 1024:
            return 8
        if block_n >= 512:
            return 8
        return 16

    def _pick_num_warps(n_rows: int, block_m: int, block_n: int) -> int:
        """ROCm: 8 warps for tiny full-wavefront decode tiles; 2 warps for larger tiles."""
        if n_rows <= 128 and block_m >= 16 and block_n >= 128:
            return 8
        return 2

    assert x.is_cuda, "fused_silu_mul requires a CUDA tensor"
    assert x.is_contiguous(), "x must be contiguous"
    last = x.size(-1)
    assert last > 0, "last dimension must be non-zero"
    assert last % 2 == 0, "last dimension must be even (2 * d)"
    d = last // 2
    leading = x.shape[:-1]
    n_rows = x.numel() // (2 * d)
    if n_rows == 0:
        return (
            torch.empty(*leading, d, dtype=x.dtype, device=x.device)
            if out is None
            else out
        )

    _LOGGER.info(f"fused_silu_mul: x={tuple(x.shape)} last_half={d} rows={n_rows}")

    if out is None:
        out = torch.empty(*leading, d, dtype=x.dtype, device=x.device)
    else:
        assert out.is_contiguous(), "out must be contiguous"
        assert out.shape == (*leading, d), "out shape must match x with last dim halved"
        assert out.dtype == x.dtype and out.device == x.device

    row_stride_in = 2 * d
    col_stride_in = 1
    row_stride_out = d
    col_stride_out = 1

    block_n = _pick_block_n(d, n_rows)
    block_m = _pick_block_m(n_rows, block_n, d)
    grid_m = triton.cdiv(n_rows, block_m)
    grid_n = triton.cdiv(d, block_n)
    num_warps = _pick_num_warps(n_rows, block_m, block_n)

    grid = (grid_m, grid_n)
    fused_silu_mul_kernel[grid](
        x,
        out,
        n_rows,
        d,
        row_stride_in,
        col_stride_in,
        row_stride_out,
        col_stride_out,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=0,
    )
    return out


def _swiglu_launch_shape(rows: int, half_cols: int) -> tuple[tuple[int, int], int, int]:
    block_m = 32
    block_i = min(triton.next_power_of_2(half_cols), 1024)
    return (
        (triton.cdiv(rows, block_m), triton.cdiv(half_cols, block_i)),
        block_m,
        block_i,
    )


def swiglu_fwd(y: torch.Tensor) -> torch.Tensor:
    """Compute standard ``silu(gate) * up`` for a concatenated input.

    The last dimension must be positive and even.

    Unlike :func:`fused_silu_mul`, this API preserves the training semantics
    used by Lumen and SonicMoE: SiLU and the multiply are evaluated in FP32
    before the result is cast back to the input dtype. Explicit row and column
    strides support views; ``reshape`` materializes only when a 2D view is not
    possible.
    """
    assert y.is_cuda, "swiglu_fwd requires a CUDA tensor"
    last = y.size(-1)
    assert last > 0, "last dimension must be non-zero"
    assert last % 2 == 0, "last dimension must be even (2 * d)"
    half_cols = last // 2
    flat_y = y.reshape(-1, last)
    rows = flat_y.shape[0]
    out = torch.empty((rows, half_cols), dtype=y.dtype, device=y.device)
    if rows == 0:
        return out.reshape(*y.shape[:-1], half_cols)

    grid, block_m, block_i = _swiglu_launch_shape(rows, half_cols)
    _glu_fwd_kernel[grid](
        flat_y,
        out,
        rows,
        half_cols,
        flat_y.stride(0),
        flat_y.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_I=block_i,
        CONCAT_LAYOUT=True,
        ACT_TYPE=0,
    )
    return out.reshape(*y.shape[:-1], half_cols)


def swiglu_bwd(grad_output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the input gradient of :func:`swiglu_fwd`.

    The last dimension of ``y`` must be positive and even.

    Explicit row and column strides support non-contiguous incoming autograd
    gradients; ``reshape`` materializes only when a 2D view is not possible.
    """
    assert y.is_cuda and grad_output.is_cuda, "swiglu_bwd requires CUDA tensors"
    last = y.size(-1)
    assert last > 0, "last dimension must be non-zero"
    assert last % 2 == 0, "last dimension must be even (2 * d)"
    half_cols = last // 2
    expected_shape = (*y.shape[:-1], half_cols)
    assert grad_output.shape == expected_shape, (
        "grad_output shape must match y with the last dimension halved"
    )
    assert grad_output.dtype == y.dtype and grad_output.device == y.device, (
        "grad_output must have the same dtype and device as y"
    )

    flat_y = y.reshape(-1, last)
    flat_grad = grad_output.reshape(-1, half_cols)
    rows = flat_y.shape[0]
    grad_input = torch.empty_like(flat_y)
    if rows == 0:
        return grad_input.reshape_as(y)

    grid, block_m, block_i = _swiglu_launch_shape(rows, half_cols)
    _glu_bwd_kernel[grid](
        flat_y,
        grad_input,
        flat_grad,
        rows,
        half_cols,
        flat_y.stride(0),
        flat_y.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
        flat_grad.stride(0),
        flat_grad.stride(1),
        BLOCK_M=block_m,
        BLOCK_I=block_i,
        CONCAT_LAYOUT=True,
        ACT_TYPE=0,
    )
    return grad_input.reshape_as(y)
