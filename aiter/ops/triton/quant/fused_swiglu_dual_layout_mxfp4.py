# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fused_swiglu_dual_layout_mxfp4 import (
    _dual_layout_quant_mxfp4_kernel,
    _fused_swiglu_bwd_dual_layout_mxfp4_kernel,
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

__all__ = [
    "dual_layout_quant_mxfp4",
    "fused_swiglu_dual_layout_mxfp4",
    "fused_swiglu_bwd_dual_layout_mxfp4",
]


_LOGGER = AiterTritonLogger()
_FWD_CONFIG_NAME = "FUSED-SWIGLU-DUAL-LAYOUT-MXFP4"
_DUAL_CONFIG_NAME = "DUAL-LAYOUT-MXFP4"
_BWD_CONFIG_NAME = "FUSED-SWIGLU-BWD-DUAL-LAYOUT-MXFP4"
_QUANT_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_PHILOX_ROUNDS = 7
_PHILOX_COLUMN_OFFSET = 0x9E3779B9
_PHILOX_TILE_M_CAP = 256
_PHILOX_TILE_N_CAP = 32
_MAX_PHILOX_COUNTER = (1 << 63) - 1
_SCALE_SWIZZLED_ATTR = "_mxfp4_scale_swizzled"
_DATA_SHUFFLED_ATTR = "_mxfp4_data_shuffled"
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


def _validate_matching(reference_name: str, reference: torch.Tensor, **others) -> None:
    _validate_tensor(reference_name, reference)
    for name, tensor in others.items():
        _validate_tensor(name, tensor)
        if tensor.shape != reference.shape:
            raise ValueError(
                f"{reference_name} and {name} must have matching shapes, got "
                f"{tuple(reference.shape)} and {tuple(tensor.shape)}"
            )
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} must be on {reference_name} device {reference.device}, "
                f"got {tensor.device}"
            )


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


def _dividing_block(dim: int, cap: int) -> int:
    block = min(cap, 1 << (dim.bit_length() - 1))
    while dim % block:
        block >>= 1
    if block < _QUANT_BLOCK_SIZE:
        raise ValueError(f"dimension {dim} has no supported block divisor")
    return block


def _get_config(config_name: str, M: int, N: int, *, exact_tiles: bool) -> dict:
    config = get_quant_config(config_name, M)
    _validate_config(config)
    if exact_tiles:
        config["BLOCK_M"] = _dividing_block(M, config["BLOCK_M"])
        config["BLOCK_N"] = _dividing_block(N, config["BLOCK_N"])
    return config


def _get_philox_tile(M: int, N: int) -> dict[str, int]:
    """Return the production RNG tile independently of launch tuning."""
    return {
        "BLOCK_M": _dividing_block(M, _PHILOX_TILE_M_CAP),
        "BLOCK_N": _dividing_block(N, _PHILOX_TILE_N_CAP),
    }


def _validate_layout(
    M: int,
    N: int,
    *,
    swizzle_scale: bool,
    shuffle_col: bool,
) -> None:
    if M == 0 or N == 0:
        raise ValueError(f"tensor dimensions must be non-zero, got {(M, N)}")
    if M % _QUANT_BLOCK_SIZE or N % _QUANT_BLOCK_SIZE:
        raise ValueError(
            f"tensor dimensions must be divisible by {_QUANT_BLOCK_SIZE}, got {(M, N)}"
        )

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


def _validate_device(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError("dual-layout MXFP4 requires gfx950")


def _philox_streams(
    M: int,
    N: int,
    use_sr: bool,
    philox_seed: int | None,
    philox_offset: int | None,
    *,
    name: str,
) -> tuple[int, int, int]:
    """Validate one branch's Philox pair and derive its column stream."""
    if not isinstance(use_sr, bool):
        raise TypeError("use_sr must be bool")
    if not use_sr:
        if philox_seed is not None or philox_offset is not None:
            raise ValueError("Philox arguments are only valid when use_sr=True")
        return 0, 0, 0

    if philox_seed is None or philox_offset is None:
        raise ValueError(f"{name} Philox seed and offset are required when use_sr=True")
    if type(philox_seed) is not int or type(philox_offset) is not int:
        raise TypeError(f"{name} Philox seed and offset must be integers")
    if not 0 <= philox_seed <= _MAX_PHILOX_COUNTER:
        raise ValueError(f"{name} Philox seed must be in [0, 2**63 - 1]")

    counters_per_layout = M * N // 8
    col_offset = philox_offset + _PHILOX_COLUMN_OFFSET
    last_start = _MAX_PHILOX_COUNTER - counters_per_layout + 1
    if not 0 <= philox_offset <= last_start or col_offset > last_start:
        raise ValueError(
            f"{name} Philox offset must leave room for row and column streams"
        )
    return philox_seed, philox_offset, col_offset


def _allocate_dual_layout(
    reference: torch.Tensor,
    *,
    swizzle_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = reference.shape
    row_scale_cols = N // _QUANT_BLOCK_SIZE
    col_scale_cols = M // _QUANT_BLOCK_SIZE
    row_packed = torch.empty((M, N // 2), dtype=torch.uint8, device=reference.device)
    row_scale_shape = (
        (M // MXFP4_SCALE_STRIPE, row_scale_cols * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (M, row_scale_cols)
    )
    row_scale = torch.empty(row_scale_shape, dtype=torch.uint8, device=reference.device)
    col_packed = torch.empty((N, M // 2), dtype=torch.uint8, device=reference.device)
    col_scale_shape = (
        (N // MXFP4_SCALE_STRIPE, col_scale_cols * MXFP4_SCALE_STRIPE)
        if swizzle_scale
        else (N, col_scale_cols)
    )
    col_scale = torch.empty(col_scale_shape, dtype=torch.uint8, device=reference.device)
    return row_packed, row_scale, col_packed, col_scale


def _mark_layout(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    swizzle_scale: bool,
    shuffle_col: bool,
) -> None:
    _, row_scale, col_packed, col_scale = outputs
    if swizzle_scale:
        setattr(row_scale, _SCALE_SWIZZLED_ATTR, True)
        setattr(col_scale, _SCALE_SWIZZLED_ATTR, True)
    if shuffle_col:
        setattr(col_packed, _DATA_SHUFFLED_ATTR, True)
        col_packed.is_shuffled = True


def _launch_dual_layout(
    x: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    config: dict,
    rng_config: dict,
    *,
    use_sr: bool,
    philox_seed: int,
    philox_offset_row: int,
    philox_offset_col: int,
    swizzle_scale: bool,
    shuffle_col: bool,
) -> None:
    M, N = x.shape
    row_packed, row_scale, col_packed, col_scale = outputs
    grid = (M // config["BLOCK_M"], N // config["BLOCK_N"])
    _dual_layout_quant_mxfp4_kernel[grid](
        x,
        row_packed,
        row_scale,
        col_packed,
        col_scale,
        _get_hadamard16(x.device),
        *x.stride(),
        *row_packed.stride(),
        *row_scale.stride(),
        *col_packed.stride(),
        *col_scale.stride(),
        M,
        N,
        philox_seed,
        philox_offset_row,
        philox_offset_col,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        HADAMARD_SIZE=_HADAMARD_SIZE,
        USE_SR=use_sr,
        PHILOX_ROUNDS=_PHILOX_ROUNDS,
        RNG_BLOCK_M=rng_config["BLOCK_M"],
        RNG_BLOCK_N=rng_config["BLOCK_N"],
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


def dual_layout_quant_mxfp4(
    x: torch.Tensor,
    *,
    use_sr: bool = False,
    philox_seed: int | None = None,
    philox_offset: int | None = None,
    swizzle_scale: bool = False,
    shuffle_col: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize BF16 ``x`` and normalized-H16 ``x.T`` into MXFP4 layouts.

    SR uses Philox4x32-7. Its row stream starts at ``philox_offset`` and its
    column stream at ``philox_offset + 0x9E3779B9``. The H16 signs are fixed to
    ``+1``, matching the current training contract.
    """
    _validate_tensor("x", x)
    if not isinstance(swizzle_scale, bool) or not isinstance(shuffle_col, bool):
        raise TypeError("swizzle_scale and shuffle_col must be bool")
    M, N = x.shape
    _validate_layout(M, N, swizzle_scale=swizzle_scale, shuffle_col=shuffle_col)
    seed, row_offset, col_offset = _philox_streams(
        M, N, use_sr, philox_seed, philox_offset, name="dual-layout"
    )
    _validate_device(x)

    config = _get_config(_DUAL_CONFIG_NAME, M, N, exact_tiles=True)
    rng_config = _get_philox_tile(M, N)
    outputs = _allocate_dual_layout(x, swizzle_scale=swizzle_scale)
    _LOGGER.info(
        f"DUAL_LAYOUT_QUANT_MXFP4: shape={tuple(x.shape)} use_sr={use_sr} "
        f"swizzle_scale={swizzle_scale} shuffle_col={shuffle_col}"
    )
    _launch_dual_layout(
        x,
        outputs,
        config,
        rng_config,
        use_sr=use_sr,
        philox_seed=seed,
        philox_offset_row=row_offset,
        philox_offset_col=col_offset,
        swizzle_scale=swizzle_scale,
        shuffle_col=shuffle_col,
    )
    _mark_layout(outputs, swizzle_scale=swizzle_scale, shuffle_col=shuffle_col)
    return outputs


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
    """Compute BF16 SwiGLU and row/normalized-H16-column RTN MXFP4 layouts."""
    _validate_matching("gate", gate, up=up)
    if not isinstance(swizzle_scale, bool):
        raise TypeError("swizzle_scale must be bool")
    if not isinstance(shuffle_col, bool):
        raise TypeError("shuffle_col must be bool")
    M, N = gate.shape
    _validate_layout(M, N, swizzle_scale=swizzle_scale, shuffle_col=shuffle_col)
    _validate_device(gate)

    config = _get_config(_FWD_CONFIG_NAME, M, N, exact_tiles=False)
    activation_bf16 = torch.empty_like(gate)
    outputs = _allocate_dual_layout(gate, swizzle_scale=swizzle_scale)
    row_packed, row_scale, col_packed, col_scale = outputs
    _LOGGER.info(
        f"FUSED_SWIGLU_DUAL_LAYOUT_MXFP4: shape={tuple(gate.shape)} "
        f"swizzle_scale={swizzle_scale} shuffle_col={shuffle_col}"
    )
    grid = (triton.cdiv(M, config["BLOCK_M"]), triton.cdiv(N, config["BLOCK_N"]))
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
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
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
    _mark_layout(outputs, swizzle_scale=swizzle_scale, shuffle_col=shuffle_col)
    return activation_bf16, row_packed, row_scale, col_packed, col_scale


def fused_swiglu_bwd_dual_layout_mxfp4(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    use_sr: bool = False,
    gate_philox_seed: int | None = None,
    gate_philox_offset: int | None = None,
    up_philox_seed: int | None = None,
    up_philox_offset: int | None = None,
    swizzle_scale: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Compute eager-compatible BF16 SwiGLU gradients and dual MXFP4 layouts.

    The flat return is ``(dgate, dup, dgate_row, dgate_row_scale, dgate_col,
    dgate_col_scale, dup_row, dup_row_scale, dup_col, dup_col_scale)``. SR uses
    explicit independent gate/up Philox pairs; each column stream adds
    ``0x9E3779B9`` to its branch's base offset. Column payloads stay unshuffled.
    """
    _validate_matching("grad_output", grad_output, gate=gate, up=up)
    if not isinstance(swizzle_scale, bool):
        raise TypeError("swizzle_scale must be bool")
    M, N = grad_output.shape
    _validate_layout(M, N, swizzle_scale=swizzle_scale, shuffle_col=False)
    gate_seed, gate_row_offset, gate_col_offset = _philox_streams(
        M,
        N,
        use_sr,
        gate_philox_seed,
        gate_philox_offset,
        name="gate",
    )
    up_seed, up_row_offset, up_col_offset = _philox_streams(
        M,
        N,
        use_sr,
        up_philox_seed,
        up_philox_offset,
        name="up",
    )
    _validate_device(grad_output)

    config = _get_config(_BWD_CONFIG_NAME, M, N, exact_tiles=True)
    rng_config = _get_philox_tile(M, N)
    dgate = torch.empty_like(gate)
    dup = torch.empty_like(up)
    dgate_layout = _allocate_dual_layout(gate, swizzle_scale=swizzle_scale)
    dup_layout = _allocate_dual_layout(up, swizzle_scale=swizzle_scale)
    dg_row, dg_row_scale, dg_col, dg_col_scale = dgate_layout
    du_row, du_row_scale, du_col, du_col_scale = dup_layout
    _LOGGER.info(
        f"FUSED_SWIGLU_BWD_DUAL_LAYOUT_MXFP4: shape={tuple(gate.shape)} "
        f"use_sr={use_sr} swizzle_scale={swizzle_scale}"
    )
    grid = (M // config["BLOCK_M"], N // config["BLOCK_N"])
    _fused_swiglu_bwd_dual_layout_mxfp4_kernel[grid](
        grad_output,
        gate,
        up,
        dgate,
        dup,
        dg_row,
        dg_row_scale,
        dg_col,
        dg_col_scale,
        du_row,
        du_row_scale,
        du_col,
        du_col_scale,
        _get_hadamard16(gate.device),
        *grad_output.stride(),
        *gate.stride(),
        *up.stride(),
        *dgate.stride(),
        *dup.stride(),
        *dg_row.stride(),
        *dg_row_scale.stride(),
        *dg_col.stride(),
        *dg_col_scale.stride(),
        *du_row.stride(),
        *du_row_scale.stride(),
        *du_col.stride(),
        *du_col_scale.stride(),
        M,
        N,
        gate_seed,
        gate_row_offset,
        gate_col_offset,
        up_seed,
        up_row_offset,
        up_col_offset,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        HADAMARD_SIZE=_HADAMARD_SIZE,
        USE_SR=use_sr,
        PHILOX_ROUNDS=_PHILOX_ROUNDS,
        RNG_BLOCK_M=rng_config["BLOCK_M"],
        RNG_BLOCK_N=rng_config["BLOCK_N"],
        SWIZZLE_SCALE=swizzle_scale,
        SCALE_STRIPE=MXFP4_SCALE_STRIPE,
        SCALE_KCHUNK=MXFP4_SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=MXFP4_SHUFFLE_UNIT_BYTES,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
    )
    _mark_layout(dgate_layout, swizzle_scale=swizzle_scale, shuffle_col=False)
    _mark_layout(dup_layout, swizzle_scale=swizzle_scale, shuffle_col=False)
    return (
        dgate,
        dup,
        dg_row,
        dg_row_scale,
        dg_col,
        dg_col_scale,
        du_row,
        du_row_scale,
        du_col,
        du_col_scale,
    )
