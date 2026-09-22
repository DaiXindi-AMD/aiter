# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "aiter/ops/triton/normalization/_rmsnorm_schedule.py"
)
_SPEC = importlib.util.spec_from_file_location("_aiter_rmsnorm_schedule", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SCHEDULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCHEDULE)

_RMSNORM_PERSISTENT_MAX_N = _SCHEDULE._RMSNORM_PERSISTENT_MAX_N
_rmsnorm_bwd_num_programs = _SCHEDULE._rmsnorm_bwd_num_programs
_rmsnorm_bwd_schedule = _SCHEDULE._rmsnorm_bwd_schedule
_rmsnorm_bwd_tile_shape = _SCHEDULE._rmsnorm_bwd_tile_shape
_should_use_large_m_small_n = _SCHEDULE._should_use_large_m_small_n
_should_use_persistent_narrow_bwd = _SCHEDULE._should_use_persistent_narrow_bwd
_should_use_tiled_forward = _SCHEDULE._should_use_tiled_forward


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, (128, 1)),
        (128, (128, 128)),
        (129, (64, 256)),
        (256, (64, 256)),
        (512, (32, 512)),
        (1536, (8, 2048)),
    ],
)
def test_rmsnorm_bwd_tile_shape(n, expected):
    assert _rmsnorm_bwd_tile_shape(n) == expected


@pytest.mark.parametrize(
    "m, block_m, num_sms, expected",
    [
        (95, 32, 8, 3),
        (1024, 32, 16, 32),
        (4097, 32, 16, 32),
    ],
)
def test_rmsnorm_bwd_persistent_program_count(m, block_m, num_sms, expected):
    assert _rmsnorm_bwd_num_programs(m, block_m, num_sms) == expected


def test_rmsnorm_bwd_dispatch_boundaries():
    assert _RMSNORM_PERSISTENT_MAX_N == 512
    assert _should_use_persistent_narrow_bwd(1, 512)
    assert not _should_use_persistent_narrow_bwd(1, 513)
    assert not _should_use_persistent_narrow_bwd(0, 128)

    assert not _should_use_large_m_small_n(8192, 513)
    assert _should_use_large_m_small_n(8193, 513)
    assert _should_use_large_m_small_n(8193, 2048)
    assert not _should_use_large_m_small_n(8193, 2049)


def test_rmsnorm_forward_dispatch_preserves_narrow_training_policy():
    assert _should_use_tiled_forward(1, 512, is_training=True)
    assert not _should_use_tiled_forward(1, 512, is_training=False)
    assert _should_use_tiled_forward(8193, 512, is_training=False)
    assert not _should_use_tiled_forward(8192, 513, is_training=True)
    assert _should_use_tiled_forward(8193, 513, is_training=True)


def test_rmsnorm_bwd_schedule_selects_persistent_and_full_grid_paths():
    # Narrow rows cap the launch at two waves per SM and grid-stride the rest.
    assert _rmsnorm_bwd_schedule(4097, 128, num_sms=8) == (128, 128, 16)

    # The pre-existing wider specialization keeps one program per row tile.
    assert _rmsnorm_bwd_schedule(8193, 513) == (16, 1024, 513)

    # Shapes outside both predicates retain the generic backward kernel.
    assert _rmsnorm_bwd_schedule(8192, 513) is None
    assert _rmsnorm_bwd_schedule(8193, 2049) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: _rmsnorm_bwd_tile_shape(0),
        lambda: _rmsnorm_bwd_num_programs(0, 32, 8),
        lambda: _rmsnorm_bwd_num_programs(1, 0, 8),
        lambda: _rmsnorm_bwd_num_programs(1, 32, 0),
        lambda: _rmsnorm_bwd_schedule(1, 128),
    ],
)
def test_rmsnorm_bwd_schedule_rejects_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
