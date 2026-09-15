# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.quant.dual_layout_mxfp4 import (
    _dual_layout_quant_mxfp4_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = ["dual_layout_quant_mxfp4"]


_LOGGER = AiterTritonLogger()
_MXFP4_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_MAX_PHILOX_COUNTER = (1 << 63) - 1
_SIGN_VALIDATED_VERSION = "_aiter_h16_sign_validated_version"


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except (AttributeError, RuntimeError):
        return None


def _prepare_h16_sign(
    sign_vector: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(sign_vector, torch.Tensor):
        raise TypeError("sign_vector must be a torch.Tensor")
    if sign_vector.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "sign_vector must have dtype torch.bfloat16 or torch.float32, "
            f"got {sign_vector.dtype}"
        )
    if sign_vector.device != reference.device:
        raise ValueError(
            "sign_vector must be on the same device as x, "
            f"got {sign_vector.device} and {reference.device}"
        )
    if sign_vector.numel() != _HADAMARD_SIZE:
        raise ValueError(
            f"sign_vector must contain {_HADAMARD_SIZE} elements, "
            f"got {sign_vector.numel()}"
        )

    version = _tensor_version(sign_vector)
    if (
        version is None
        or getattr(sign_vector, _SIGN_VALIDATED_VERSION, None) != version
    ):
        valid = torch.all(
            torch.isfinite(sign_vector) & ((sign_vector == 1) | (sign_vector == -1))
        )
        message = "sign_vector entries must be finite and equal to +1 or -1"
        if torch.cuda.is_current_stream_capturing():
            assert_async = getattr(torch, "_assert_async", None)
            if assert_async is None:
                raise RuntimeError(
                    "sign_vector must be validated before CUDA graph capture"
                )
            assert_async(valid, message)
        elif not valid.item():
            raise ValueError(message)
        if version is not None:
            setattr(sign_vector, _SIGN_VALIDATED_VERSION, version)

    return sign_vector.reshape(-1).contiguous()


def _philox_streams(
    M: int,
    N: int,
    use_sr_row: bool,
    use_sr_transposed: bool,
    philox_seed: int | None,
    philox_offset: int,
) -> tuple[int, int, int]:
    use_sr = use_sr_row or use_sr_transposed
    if not use_sr:
        if philox_seed is not None or philox_offset != 0:
            raise ValueError(
                "Philox arguments are only valid when stochastic rounding is enabled"
            )
        return 0, 0, 0

    if philox_seed is None:
        raise ValueError("philox_seed is required when stochastic rounding is enabled")
    if type(philox_seed) is not int or type(philox_offset) is not int:
        raise TypeError("philox_seed and philox_offset must be integers")
    if not 0 <= philox_seed <= _MAX_PHILOX_COUNTER:
        raise ValueError("philox_seed must be in [0, 2**63 - 1]")

    counters_per_layout = M * N // 8
    enabled_layouts = int(use_sr_row) + int(use_sr_transposed)
    reserved_counters = enabled_layouts * counters_per_layout
    if not 0 <= philox_offset <= _MAX_PHILOX_COUNTER - reserved_counters + 1:
        raise ValueError(
            "philox_offset must be non-negative and leave room for enabled layouts"
        )

    next_offset = philox_offset
    row_offset = next_offset if use_sr_row else 0
    if use_sr_row:
        next_offset += counters_per_layout
    transposed_offset = next_offset if use_sr_transposed else 0
    return philox_seed, row_offset, transposed_offset


def dual_layout_quant_mxfp4(
    x: torch.Tensor,
    sign_vector: torch.Tensor,
    *,
    use_sr_row: bool = False,
    use_sr_transposed: bool = False,
    philox_seed: int | None = None,
    philox_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``x`` and its transposed H16 rotation from one dense read.

    The first output pair is row-wise ``MXFP4(x)``. The second is row-wise
    ``MXFP4(H16(x.T))``, where each consecutive group of 16 values is first
    multiplied by ``sign_vector`` and then by a normalized Hadamard-16 matrix.
    Both layouts use one raw E8M0 scale per 32 consecutive values.

    Args:
        x: Contiguous 2-D CUDA tensor with dtype bfloat16 or float32. Both
            dimensions must be positive multiples of 32.
        sign_vector: CUDA tensor with 16 finite values, each exactly +1 or -1,
            on the same device as ``x``.
        use_sr_row: Stochastically round the row-wise E2M1 payload.
        use_sr_transposed: Stochastically round the rotated-transposed payload.
        philox_seed: Non-negative Philox seed, required when either SR flag is
            enabled.
        philox_offset: Starting Philox counter. Each SR-enabled layout consumes
            ``M*N/8`` counters. Enabled streams are assigned contiguously in row,
            then transposed order; disabled layouts consume no counters.

    Returns:
        ``(row_fp4, row_scales, transposed_fp4, transposed_scales)`` with
        shapes ``(M, N/2)``, ``(M, N/32)``, ``(N, M/2)``, and ``(N, M/32)``.
        All outputs are contiguous uint8 tensors in canonical logical order.

    Raises:
        TypeError: If an input dtype or Philox argument is invalid.
        ValueError: If an input shape, device, sign, or counter range is invalid.
        RuntimeError: If the input is not on a gfx950 device.
    """
    _LOGGER.info(
        "DUAL_LAYOUT_QUANT_MXFP4: "
        f"x={tuple(x.shape)} use_sr_row={use_sr_row} "
        f"use_sr_transposed={use_sr_transposed}"
    )
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D, got {x.dim()}-D")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"x must have dtype torch.bfloat16 or torch.float32, got {x.dtype}"
        )
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not isinstance(use_sr_row, bool) or not isinstance(use_sr_transposed, bool):
        raise TypeError("use_sr_row and use_sr_transposed must be bool")

    M, N = x.shape
    if M == 0 or N == 0:
        raise ValueError(f"x dimensions must be non-zero, got {tuple(x.shape)}")
    if M % _MXFP4_BLOCK_SIZE != 0 or N % _MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "x dimensions must both be divisible by "
            f"{_MXFP4_BLOCK_SIZE}, got {tuple(x.shape)}"
        )
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError("dual-layout MXFP4 quantization requires gfx950")

    sign_vector = _prepare_h16_sign(sign_vector, x)
    philox_seed, philox_offset_row, philox_offset_transposed = _philox_streams(
        M,
        N,
        use_sr_row,
        use_sr_transposed,
        philox_seed,
        philox_offset,
    )

    row_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    row_scales = torch.empty(
        (M, N // _MXFP4_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )
    transposed_fp4 = torch.empty((N, M // 2), dtype=torch.uint8, device=x.device)
    transposed_scales = torch.empty(
        (N, M // _MXFP4_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )

    # One program owns one semantic 32x32 quantization tile. Execution tuning
    # is intentionally left to Triton's defaults.
    grid = (M // _MXFP4_BLOCK_SIZE, N // _MXFP4_BLOCK_SIZE)
    _dual_layout_quant_mxfp4_kernel[grid](
        x,
        row_fp4,
        row_scales,
        transposed_fp4,
        transposed_scales,
        sign_vector,
        *x.stride(),
        *row_fp4.stride(),
        *row_scales.stride(),
        *transposed_fp4.stride(),
        *transposed_scales.stride(),
        M,
        N,
        philox_seed,
        philox_offset_row,
        philox_offset_transposed,
        BLOCK_SIZE=_MXFP4_BLOCK_SIZE,
        HADAMARD_SIZE=_HADAMARD_SIZE,
        USE_SR_ROW=use_sr_row,
        USE_SR_TRANSPOSED=use_sr_transposed,
    )
    return row_fp4, row_scales, transposed_fp4, transposed_scales
