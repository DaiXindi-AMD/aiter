# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-Python scheduling helpers for the Triton RMSNorm kernels."""

_RMSNORM_BWD_WAVES_PER_SM = 2
_RMSNORM_PERSISTENT_MAX_N = 512
_RMSNORM_LARGE_M_THRESHOLD = 8192
_RMSNORM_LARGE_M_SMALL_N_MAX_N = 2048
_RMSNORM_BWD_TILE_ELEMS = 16384


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def _should_use_large_m_small_n(m: int, n: int) -> bool:
    return bool(
        m > _RMSNORM_LARGE_M_THRESHOLD
        and 0 < n <= _RMSNORM_LARGE_M_SMALL_N_MAX_N
    )


def _should_use_persistent_narrow_bwd(m: int, n: int) -> bool:
    """Use Lumen's bounded persistent backward schedule for narrow rows."""
    return bool(m > 0 and 0 < n <= _RMSNORM_PERSISTENT_MAX_N)


def _should_use_tiled_forward(m: int, n: int, is_training: bool) -> bool:
    """Use the tiled forward for narrow training or the existing large-M case."""
    return bool(
        _should_use_large_m_small_n(m, n)
        or (is_training and _should_use_persistent_narrow_bwd(m, n))
    )


def _rmsnorm_bwd_tile_shape(n: int) -> tuple[int, int]:
    block_n = _next_power_of_2(n)
    block_m = max(8, min(128, _RMSNORM_BWD_TILE_ELEMS // block_n))
    return block_m, block_n


def _rmsnorm_bwd_num_programs(m: int, block_m: int, num_sms: int) -> int:
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")

    num_tiles = _ceil_div(m, block_m)
    return min(num_tiles, num_sms * _RMSNORM_BWD_WAVES_PER_SM)


def _rmsnorm_bwd_schedule(
    m: int, n: int, num_sms: int | None = None
) -> tuple[int, int, int] | None:
    """Return ``(block_m, block_n, programs)`` for the tiled backward path."""
    persistent = _should_use_persistent_narrow_bwd(m, n)
    if not persistent and not _should_use_large_m_small_n(m, n):
        return None

    block_m, block_n = _rmsnorm_bwd_tile_shape(n)
    if persistent:
        if num_sms is None:
            raise ValueError("num_sms is required for the persistent RMSNorm schedule")
        num_programs = _rmsnorm_bwd_num_programs(m, block_m, num_sms)
    else:
        num_programs = _ceil_div(m, block_m)
    return block_m, block_n, num_programs
