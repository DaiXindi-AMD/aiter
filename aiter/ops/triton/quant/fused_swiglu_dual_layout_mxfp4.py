# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fused_swiglu_dual_layout_mxfp4 import (
    _fused_swiglu_dual_layout_mxfp4_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.shuffle import (
    MXFP4_SCALE_KCHUNK,
    MXFP4_SCALE_STRIPE,
    MXFP4_SHUFFLE_TILE_ROWS,
    MXFP4_SHUFFLE_UNIT_BYTES,
    mxfp4_data_shuffle_supported,
    mxfp4_scale_swizzle_supported,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.quant_config_utils import get_quant_config

__all__ = ["fused_swiglu_dual_layout_mxfp4"]


_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "FUSED-SWIGLU-DUAL-LAYOUT-MXFP4"
_QUANT_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_HADAMARD16_CACHE: dict[torch.device, torch.Tensor] = {}


def _hadamard16_values() -> tuple[tuple[float, ...], ...]:
    rows = ((1.0,),)
    while len(rows) < _HADAMARD_SIZE:
        rows = tuple(row + row for row in rows) + tuple(
            row + tuple(-value for value in row) for row in rows
        )
    return tuple(tuple(value * 0.25 for value in row) for row in rows)


_NORMALIZED_HADAMARD16 = _hadamard16_values()


def _get_hadamard16(device: torch.device) -> torch.Tensor:
    matrix = _HADAMARD16_CACHE.get(device)
    if matrix is None:
        matrix = torch.tensor(
            _NORMALIZED_HADAMARD16,
            dtype=torch.bfloat16,
            device=device,
        )
        _HADAMARD16_CACHE[device] = matrix
    return matrix


def _validate_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got {tensor.dim()}-D")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_config(config: dict) -> None:
    required = ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "waves_per_eu")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing fused SwiGLU MXFP4 config keys: {missing}")
    if any(type(config[key]) is not int for key in required):
        raise TypeError("Fused SwiGLU MXFP4 config values must be integers")

    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]
    if (
        block_m < _QUANT_BLOCK_SIZE
        or block_n < _QUANT_BLOCK_SIZE
        or block_m % _QUANT_BLOCK_SIZE
        or block_n % _QUANT_BLOCK_SIZE
        or block_m & (block_m - 1)
        or block_n & (block_n - 1)
    ):
        raise ValueError("BLOCK_M and BLOCK_N must be powers of two divisible by 32")
    if config["num_warps"] <= 0 or config["num_warps"] & (config["num_warps"] - 1):
        raise ValueError("num_warps must be a positive power of two")
    if config["num_stages"] <= 0 or config["waves_per_eu"] < 0:
        raise ValueError("num_stages must be positive and waves_per_eu non-negative")


def fused_swiglu_dual_layout_mxfp4(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swizzle_scale: bool = False,
    shuffle_col: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute BF16 SwiGLU and its row/rotated-column RTN MXFP4 layouts.

    ``activation_bf16`` exactly follows two BF16 rounding points:
    ``silu(gate.float()).bfloat16()`` followed by multiplication with ``up`` in
    FP32 and a second BF16 cast. Both quantized layouts use that stored value.
    The column layout is the transpose after a normalized deterministic H16
    transform (all signs are ``+1``).

    Args:
        gate: Contiguous 2-D BF16 CUDA tensor shaped ``(M, N)``.
        up: Matching contiguous 2-D BF16 CUDA tensor.
        swizzle_scale: Store both scale tensors in gfx950's 32x8 GEMM layout.
        shuffle_col: Store the column payload in AITER's 16-row B layout.

    Returns:
        ``(activation_bf16, row_packed, row_scale, col_packed, col_scale)``.
        Canonical shapes are ``(M, N)``, ``(M, N/2)``, ``(M, N/32)``,
        ``(N, M/2)``, and ``(N, M/32)``. Swizzled scale shapes are
        ``(M/32, N)`` and ``(N/32, M)``.
    """
    _validate_tensor("gate", gate)
    _validate_tensor("up", up)
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must have matching shapes, got {gate.shape} and {up.shape}"
        )
    if gate.device != up.device:
        raise ValueError(
            f"gate and up must be on the same device, got {gate.device} and {up.device}"
        )
    if not isinstance(swizzle_scale, bool):
        raise TypeError("swizzle_scale must be bool")
    if not isinstance(shuffle_col, bool):
        raise TypeError("shuffle_col must be bool")

    M, N = gate.shape
    if M == 0 or N == 0:
        raise ValueError(f"gate and up dimensions must be non-zero, got {(M, N)}")
    if M % _QUANT_BLOCK_SIZE or N % _QUANT_BLOCK_SIZE:
        raise ValueError(
            f"gate and up dimensions must be divisible by {_QUANT_BLOCK_SIZE}, "
            f"got {(M, N)}"
        )
    if not gate.is_cuda:
        raise ValueError("gate and up must be CUDA tensors")
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError("fused SwiGLU dual-layout MXFP4 requires gfx950")

    row_scale_cols = N // _QUANT_BLOCK_SIZE
    col_scale_cols = M // _QUANT_BLOCK_SIZE
    if swizzle_scale:
        if not mxfp4_scale_swizzle_supported(M, row_scale_cols):
            raise ValueError(
                f"row scales {(M, row_scale_cols)} must tile evenly for 32x8 swizzle"
            )
        if not mxfp4_scale_swizzle_supported(N, col_scale_cols):
            raise ValueError(
                f"column scales {(N, col_scale_cols)} must tile evenly for 32x8 swizzle"
            )
    if shuffle_col and not mxfp4_data_shuffle_supported(N, M // 2):
        raise ValueError(
            f"column payload shape {(N, M // 2)} does not tile for the B shuffle"
        )

    config = get_quant_config(_CONFIG_NAME, M)
    _validate_config(config)
    block_m = config["BLOCK_M"]
    block_n = config["BLOCK_N"]

    activation_bf16 = torch.empty_like(gate)
    row_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=gate.device)
    row_scale_shape = (
        (M // MXFP4_SCALE_STRIPE, row_scale_cols * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (M, row_scale_cols)
    )
    row_scale = torch.empty(row_scale_shape, dtype=torch.uint8, device=gate.device)
    col_packed = torch.empty((N, M // 2), dtype=torch.uint8, device=gate.device)
    col_scale_shape = (
        (N // MXFP4_SCALE_STRIPE, col_scale_cols * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (N, col_scale_cols)
    )
    col_scale = torch.empty(col_scale_shape, dtype=torch.uint8, device=gate.device)

    _LOGGER.info(
        "FUSED_SWIGLU_DUAL_LAYOUT_MXFP4: "
        f"shape={tuple(gate.shape)} swizzle_scale={swizzle_scale} "
        f"shuffle_col={shuffle_col}"
    )
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_swiglu_dual_layout_mxfp4_kernel[grid](
        gate,
        up,
        activation_bf16,
        row_packed,
        row_scale,
        col_packed,
        col_scale,
        _get_hadamard16(gate.device),
        *gate.stride(),
        *up.stride(),
        *activation_bf16.stride(),
        *row_packed.stride(),
        *row_scale.stride(),
        *col_packed.stride(),
        *col_scale.stride(),
        M,
        N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        HADAMARD_SIZE=_HADAMARD_SIZE,
        SWIZZLE_SCALE=swizzle_scale,
        SHUFFLE_COL=shuffle_col,
        SCALE_STRIPE=MXFP4_SCALE_STRIPE,
        SCALE_KCHUNK=MXFP4_SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=MXFP4_SHUFFLE_UNIT_BYTES,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
    )
    if shuffle_col:
        col_packed.is_shuffled = True
    return activation_bf16, row_packed, row_scale, col_packed, col_scale
