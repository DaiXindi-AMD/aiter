import math
from typing import Literal, Optional
import triton
import torch
import aiter
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.activation import (
    _act_mul_and_dynamic_mxfp4_quant_kernel,
    _act_mul_and_dynamic_fp8_group_quant_kernel,
    fused_silu_mul_kernel,
    _swiglu_bwd_kernel,
)

fp8_dtype = aiter.dtypes.fp8

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
    dtype_quant=fp8_dtype,
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


def _pick_silu_block_n(d: int, n_rows: int) -> int:
    """Choose the existing ROCm-tuned feature tile for fused SiLU kernels."""
    n = max(d, 1)
    if n == 512:
        return 512 if n_rows > 4096 else 256
    if n == 384:
        return 256 if n_rows <= 128 else 128
    upper = min(n, 1024)
    p = 1
    while p * 2 <= upper:
        p *= 2
    return max(32, p)


def _pick_silu_block_m(n_rows: int, block_n: int, d: int) -> int:
    """Choose the existing ROCm-tuned row tile for fused SiLU kernels."""
    if n_rows <= 64:
        return min(32, max(4, triton.next_power_of_2(n_rows)))
    if d == 384 and n_rows > 128:
        return 32 if n_rows > 8192 else 8
    if d == 512 and n_rows > 4096:
        return 8
    if d == 512 and 128 < n_rows <= 4096:
        return 8
    if block_n >= 512:
        return 8
    return 16


def _pick_silu_num_warps(n_rows: int, block_m: int, block_n: int) -> int:
    """Use wide wave groups only for the established small-row regime."""
    if n_rows <= 128 and block_m >= 16 and block_n >= 128:
        return 8
    return 2


_SWIGLU_DTYPES = (torch.float16, torch.bfloat16)


def _validate_swiglu_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype not in _SWIGLU_DTYPES:
        raise TypeError(f"{name} must have dtype float16 or bfloat16")


def _view_feature_rows(name: str, tensor: torch.Tensor) -> torch.Tensor:
    d = tensor.size(-1)
    if tensor.numel() == 0:
        return tensor
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have a contiguous last dimension")
    try:
        return tensor.view(-1, d)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} leading dimensions must flatten without a copy"
        ) from exc


def _view_writable_feature_rows(name: str, tensor: torch.Tensor) -> torch.Tensor:
    flat = _view_feature_rows(name, tensor)
    if flat.numel() and flat.size(0) > 1 and flat.stride(0) < flat.size(1):
        raise ValueError(f"{name} must not have internal overlap")
    return flat


def _validate_matching_swiglu_tensors(
    reference_name: str,
    reference: torch.Tensor,
    *named_tensors: tuple[str, torch.Tensor],
) -> None:
    _validate_swiglu_tensor(reference_name, reference)
    for name, tensor in named_tensors:
        _validate_swiglu_tensor(name, tensor)
        if tensor.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
        if tensor.dtype != reference.dtype:
            raise TypeError(f"{name} must have dtype {reference.dtype}")
        if tensor.device != reference.device:
            raise ValueError(f"{name} must be on device {reference.device}")


def _ceil_div(value: int, divisor: int) -> int:
    return -((-value) // divisor)


def _continuous_region_overlaps_rows(
    region_start: int,
    region_end: int,
    row_start: int,
    row_step: int,
    row_count: int,
    row_width: int,
) -> bool:
    if row_step == 0:
        return row_start < region_end and region_start < row_start + row_width
    first_overlap = _ceil_div(region_start - row_width + 1 - row_start, row_step)
    last_overlap = (region_end - 1 - row_start) // row_step
    return max(first_overlap, 0) <= min(last_overlap, row_count - 1)


def _row_intervals_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_flat = _view_feature_rows("left", left)
    right_flat = _view_feature_rows("right", right)
    left_rows, left_cols = left_flat.shape
    right_rows, right_cols = right_flat.shape
    left_width = left_cols * left.element_size()
    right_width = right_cols * right.element_size()
    left_step = left_flat.stride(0) * left.element_size()
    right_step = right_flat.stride(0) * right.element_size()
    left_start = left_flat.data_ptr()
    right_start = right_flat.data_ptr()
    left_end = left_start + (left_rows - 1) * left_step + left_width
    right_end = right_start + (right_rows - 1) * right_step + right_width
    if left_end <= right_start or right_end <= left_start:
        return False

    if left_step == right_step and left_step > 0:
        delta = right_start - left_start
        min_row_delta = -(left_rows - 1)
        max_row_delta = right_rows - 1
        first_overlap = _ceil_div(-right_width + 1 - delta, left_step)
        last_overlap = (left_width - 1 - delta) // left_step
        return max(first_overlap, min_row_delta) <= min(last_overlap, max_row_delta)

    if left_rows == 1 or left_step == left_width:
        return _continuous_region_overlaps_rows(
            left_start,
            left_end,
            right_start,
            right_step,
            right_rows,
            right_width,
        )
    if right_rows == 1 or right_step == right_width:
        return _continuous_region_overlaps_rows(
            right_start,
            right_end,
            left_start,
            left_step,
            left_rows,
            left_width,
        )

    # Unequal padded row strides are uncommon; reject an ambiguous shared span.
    return True


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    if not left.numel() or not right.numel():
        return False
    return _row_intervals_overlap(left, right)


def _reject_output_overlap(
    outputs: tuple[tuple[str, torch.Tensor], ...],
    inputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    for output_index, (output_name, output) in enumerate(outputs):
        for input_name, input_tensor in inputs:
            if _tensors_overlap(output, input_tensor):
                raise ValueError(f"{output_name} must not overlap {input_name}")
        for other_name, other in outputs[:output_index]:
            if _tensors_overlap(output, other):
                raise ValueError(f"{output_name} must not overlap {other_name}")


def _launch_fused_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
    *,
    eager_rounding: bool,
) -> torch.Tensor:
    d = gate.size(-1)
    if gate.numel() == 0:
        return out

    flat_gate = _view_feature_rows("gate", gate)
    flat_up = _view_feature_rows("up", up)
    flat_out = _view_feature_rows("out", out)
    n_rows = flat_gate.size(0)
    block_n = _pick_silu_block_n(d, n_rows)
    block_m = _pick_silu_block_m(n_rows, block_n, d)
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(d, block_n))
    fused_silu_mul_kernel[grid](
        flat_gate,
        flat_up,
        flat_out,
        n_rows,
        d,
        flat_gate.stride(0),
        flat_gate.stride(1),
        flat_up.stride(0),
        flat_up.stride(1),
        flat_out.stride(0),
        flat_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        EAGER_ROUNDING=eager_rounding,
        num_warps=_pick_silu_num_warps(n_rows, block_m, block_n),
        waves_per_eu=0,
    )
    return out


def swiglu_fwd_split(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute eager-equivalent ``silu(gate) * up`` without concatenation."""
    _validate_matching_swiglu_tensors("gate", gate, ("up", up))
    _view_feature_rows("gate", gate)
    _view_feature_rows("up", up)

    if out is None:
        out = torch.empty_like(gate, memory_format=torch.preserve_format)
    else:
        _validate_matching_swiglu_tensors("gate", gate, ("out", out))
        _view_writable_feature_rows("out", out)
        _reject_output_overlap((("out", out),), (("gate", gate), ("up", up)))
    return _launch_fused_silu_mul(gate, up, out, eager_rounding=True)


def fused_silu_mul_two_input(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compatibility alias for :func:`swiglu_fwd_split`."""
    return swiglu_fwd_split(gate, up, out)


def fused_silu_mul(
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute packed ``silu(x[..., :d]) * x[..., d:]`` in one kernel."""

    _validate_swiglu_tensor("x", x)
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    last = x.size(-1)
    if last % 2 != 0:
        raise ValueError("last dimension must be even (2 * d)")
    d = last // 2
    leading = x.shape[:-1]

    if out is None:
        out = torch.empty(*leading, d, dtype=x.dtype, device=x.device)
    else:
        _validate_swiglu_tensor("out", out)
        if out.shape != (*leading, d):
            raise ValueError("out shape must match x with last dim halved")
        if out.dtype != x.dtype:
            raise TypeError(f"out must have dtype {x.dtype}")
        if out.device != x.device:
            raise ValueError(f"out must be on device {x.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        _view_writable_feature_rows("out", out)
        _reject_output_overlap((("out", out),), (("x", x),))

    n_rows = math.prod(leading)
    if x.numel() == 0:
        return out

    _LOGGER.info(f"fused_silu_mul: x={tuple(x.shape)} last_half={d} rows={n_rows}")

    gate = x[..., :d]
    up = x[..., d:]
    return _launch_fused_silu_mul(gate, up, out, eager_rounding=True)


# ── Fused SwiGLU ─────────────────────────────────────────────────────────


def swiglu_fwd(y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward: silu(y1) * y2 where y is split along last dim.

    This compatibility path keeps FP32 intermediates and reshapes noncontiguous
    inputs when needed before launching one Triton kernel.
    """
    _validate_swiglu_tensor("y", y)
    if y.size(-1) % 2 != 0:
        raise ValueError("last dimension must be even (2 * d)")

    half_cols = y.size(-1) // 2
    out = torch.empty(*y.shape[:-1], half_cols, dtype=y.dtype, device=y.device)
    if y.numel() == 0:
        return out
    flat_y = y.reshape(-1, y.size(-1))
    gate = flat_y[:, :half_cols]
    up = flat_y[:, half_cols:]
    out = out.view(-1, half_cols)
    _launch_fused_silu_mul(gate, up, out, eager_rounding=False)
    return out.reshape(*y.shape[:-1], half_cols)


def swiglu_bwd_split(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_gate: Optional[torch.Tensor] = None,
    grad_up: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute eager-equivalent SwiGLU gradients for separate inputs."""
    _validate_matching_swiglu_tensors(
        "gate", gate, ("up", up), ("grad_output", grad_output)
    )
    _view_feature_rows("gate", gate)
    _view_feature_rows("up", up)
    _view_feature_rows("grad_output", grad_output)

    explicit_outputs = []
    if grad_gate is None:
        grad_gate = torch.empty_like(gate, memory_format=torch.preserve_format)
    else:
        _validate_matching_swiglu_tensors("gate", gate, ("grad_gate", grad_gate))
        _view_writable_feature_rows("grad_gate", grad_gate)
        explicit_outputs.append(("grad_gate", grad_gate))
    if grad_up is None:
        grad_up = torch.empty_like(up, memory_format=torch.preserve_format)
    else:
        _validate_matching_swiglu_tensors("up", up, ("grad_up", grad_up))
        _view_writable_feature_rows("grad_up", grad_up)
        explicit_outputs.append(("grad_up", grad_up))

    if explicit_outputs:
        _reject_output_overlap(
            tuple(explicit_outputs),
            (("grad_output", grad_output), ("gate", gate), ("up", up)),
        )
    if gate.numel() == 0:
        return grad_gate, grad_up

    d = gate.size(-1)
    flat_grad = _view_feature_rows("grad_output", grad_output)
    flat_gate = _view_feature_rows("gate", gate)
    flat_up = _view_feature_rows("up", up)
    flat_dgate = _view_feature_rows("grad_gate", grad_gate)
    flat_dup = _view_feature_rows("grad_up", grad_up)
    n_rows = flat_gate.size(0)
    block_n = _pick_silu_block_n(d, n_rows)
    block_m = _pick_silu_block_m(n_rows, block_n, d)
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(d, block_n))
    _swiglu_bwd_kernel[grid](
        flat_grad,
        flat_gate,
        flat_up,
        flat_dgate,
        flat_dup,
        n_rows,
        d,
        flat_grad.stride(0),
        flat_grad.stride(1),
        flat_gate.stride(0),
        flat_gate.stride(1),
        flat_up.stride(0),
        flat_up.stride(1),
        flat_dgate.stride(0),
        flat_dgate.stride(1),
        flat_dup.stride(0),
        flat_dup.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        EAGER_ROUNDING=True,
        num_warps=_pick_silu_num_warps(n_rows, block_m, block_n),
        waves_per_eu=0,
    )
    return grad_gate, grad_up


def fused_silu_mul_backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    dgate: Optional[torch.Tensor] = None,
    dup: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility alias for :func:`swiglu_bwd_split`."""
    return swiglu_bwd_split(grad_output, gate, up, dgate, dup)


def swiglu_bwd(grad_output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU backward: computes gradient w.r.t. input y.

    This compatibility path keeps FP32 intermediates and reshapes noncontiguous
    inputs or gradients when needed before launching one Triton kernel.
    """
    _validate_swiglu_tensor("y", y)
    if y.size(-1) % 2 != 0:
        raise ValueError("last dimension must be even (2 * d)")
    half_cols = y.shape[-1] // 2
    expected_grad_shape = (*y.shape[:-1], half_cols)
    _validate_swiglu_tensor("grad_output", grad_output)
    if grad_output.shape != expected_grad_shape:
        raise ValueError(f"grad_output must have shape {expected_grad_shape}")
    if grad_output.dtype != y.dtype:
        raise TypeError(f"grad_output must have dtype {y.dtype}")
    if grad_output.device != y.device:
        raise ValueError(f"grad_output must be on device {y.device}")
    if y.numel() == 0:
        return torch.empty_like(y)

    flat_y = y.reshape(-1, y.size(-1))
    flat_grad = grad_output.reshape(-1, half_cols)
    d_input = torch.empty_like(flat_y)
    gate = flat_y[:, :half_cols]
    up = flat_y[:, half_cols:]
    dgate = d_input[:, :half_cols]
    dup = d_input[:, half_cols:]
    # Packed views have a wider row stride, which the shared tiled kernel accepts.
    rows = gate.shape[0]
    block_n = _pick_silu_block_n(half_cols, rows)
    block_m = _pick_silu_block_m(rows, block_n, half_cols)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(half_cols, block_n))
    _swiglu_bwd_kernel[grid](
        flat_grad,
        gate,
        up,
        dgate,
        dup,
        rows,
        half_cols,
        flat_grad.stride(0),
        flat_grad.stride(1),
        gate.stride(0),
        gate.stride(1),
        up.stride(0),
        up.stride(1),
        dgate.stride(0),
        dgate.stride(1),
        dup.stride(0),
        dup.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        EAGER_ROUNDING=False,
        num_warps=_pick_silu_num_warps(rows, block_m, block_n),
        waves_per_eu=0,
    )
    return d_input.reshape_as(y)
