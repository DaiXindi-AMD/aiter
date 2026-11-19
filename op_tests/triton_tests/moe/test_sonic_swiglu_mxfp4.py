# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused SonicMoE producer preserves activation and MXFP4 bytes."""

import pytest
import torch

from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    grouped_mxfp4_wgrad,
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.moe.sonic_mxfp4 import _use_fused_swiglu_producer
from aiter.ops.triton.moe.sonic_swiglu_mxfp4 import fused_swiglu_mxfp4
from aiter.ops.triton.moe.sonicmoe import activation_fwd
from aiter.ops.triton.quant.grouped_mxfp4_colwise import (
    quantize_grouped_mxfp4_colwise,
)
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_dual,
    quantize_grouped_mxfp4_dual_v2,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_COUNTS = (0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 33, 47, 64) * 2
_FEATURES = 768


def _inputs(pad_multiple=None):
    torch.manual_seed(20261119)
    rows = sum(_COUNTS)
    h1 = torch.randn((rows, 2 * _FEATURES), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        _COUNTS, h1.device, pad_multiple=pad_multiple, backend="triton"
    )
    return h1, metadata


@pytest.mark.parametrize(
    "pad_multiple,dual",
    [
        (128, quantize_grouped_mxfp4_dual),
        (256, quantize_grouped_mxfp4_dual_v2),
    ],
    ids=["triton-wgrad", "gluon-wgrad"],
)
def test_fused_swiglu_mxfp4_byte_exact_forward(pad_multiple, dual):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    h1, metadata = _inputs(pad_multiple)

    expected_a = activation_fwd(h1, _FEATURES, "swiglu", concat_layout=True)
    expected_row, expected_col = dual(expected_a, metadata)
    actual_a, actual_row, actual_col = fused_swiglu_mxfp4(h1, metadata)

    assert torch.equal(actual_a, expected_a)
    for actual, expected in zip(
        (*actual_row, *actual_col), (*expected_row, *expected_col)
    ):
        assert torch.equal(actual, expected)
    assert actual_col.metadata is metadata
    assert actual_col.source is actual_a


def test_fused_swiglu_mxfp4_byte_exact_wgrad_contract():
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    h1, metadata = _inputs()
    activation, _row, activation_col = fused_swiglu_mxfp4(h1, metadata)
    dy = torch.randn(
        (activation.shape[0], 64), dtype=torch.bfloat16, device=activation.device
    )
    dy_col = quantize_grouped_mxfp4_colwise(dy, metadata)

    expected = grouped_mxfp4_wgrad(
        activation, dy, _COUNTS, metadata=metadata, backend="triton"
    )
    actual = grouped_mxfp4_wgrad(
        activation,
        dy,
        _COUNTS,
        metadata=metadata,
        quantized_operands=(activation_col, dy_col),
        backend="triton",
    )
    assert torch.equal(actual, expected)


def test_fused_swiglu_selection_is_default_off_and_native_only(monkeypatch):
    monkeypatch.delenv("SONIC_MOE_MXFP4_FUSED_PRODUCER", raising=False)
    assert not _use_fused_swiglu_producer("mxfp4")
    monkeypatch.setenv("SONIC_MOE_MXFP4_FUSED_PRODUCER", "1")
    assert _use_fused_swiglu_producer("mxfp4")
    assert not _use_fused_swiglu_producer("bf16")
    monkeypatch.setenv("SONIC_MOE_MXFP4_FUSED_PRODUCER", "invalid")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _use_fused_swiglu_producer("mxfp4")
