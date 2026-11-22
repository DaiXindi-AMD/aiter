# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Byte-exact contract of the one-read dual-orientation MXFP4 weight pack."""

import pytest
import torch

from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
    pack_grouped_mxfp4_weights,
    pack_pre_routed_weights_dual,
)
from aiter.ops.triton.moe.sonic_mxfp4 import pack_pre_routed_weights
from aiter.ops.triton.quant.dual_mxfp4_weight_pack import quantize_mxfp4_weight_dual
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(get_arch() != "gfx950", reason="gfx950 only")


def _assert_packs_equal(actual, expected):
    for (a_payload, a_scales), (e_payload, e_scales) in zip(actual, expected):
        assert a_payload.is_contiguous() and a_scales.is_contiguous()
        assert torch.equal(a_payload, e_payload)
        assert torch.equal(a_scales, e_scales)


@pytest.mark.parametrize(
    "e,h,i",
    [(32, 2048, 768), (2, 128, 64)],
)
def test_pre_routed_dual_pack_matches_four_pack_loop(e, h, i):
    torch.manual_seed(20261122)
    w1 = torch.randn((e, h, 2 * i), dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn((e, i, h), dtype=torch.bfloat16, device="cuda") * 0.02
    _assert_packs_equal(
        pack_pre_routed_weights_dual(w1, w2), pack_pre_routed_weights(w1, w2)
    )


def test_dual_pack_edge_values_match_single_orientation_quant():
    torch.manual_seed(7)
    weight = torch.randn((3, 128, 256), dtype=torch.float32, device="cuda")
    weight[0] = 0.0
    weight[1, :, :64] = -0.0
    weight[1, :32, 64:] *= 1e30
    weight[1, 32:64, 64:] *= 1e-30
    weight[2, 0, 0] = float("inf")
    weight[2, 1, 5] = float("-inf")
    weight[2, 40, 200] = float("nan")
    # +inf next to NaN, next to -inf, and on an odd (high-nibble) column.
    weight[2, 70, 33] = float("inf")
    weight[2, 70, 34] = float("nan")
    weight[2, 90, 96] = float("inf")
    weight[2, 90, 97] = float("-inf")
    weight[2, 100, 129] = float("inf")
    weight = weight.to(torch.bfloat16)
    row, col = quantize_mxfp4_weight_dual(weight)
    _assert_packs_equal(
        (row, col),
        (
            pack_grouped_mxfp4_weights(weight),
            pack_grouped_mxfp4_weights(weight.transpose(1, 2)),
        ),
    )


def test_dual_pack_leaves_input_unchanged():
    weight = torch.randn((2, 128, 128), dtype=torch.bfloat16, device="cuda")
    before = weight.clone()
    quantize_mxfp4_weight_dual(weight)
    assert torch.equal(weight, before)


@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.randn((2, 128, 128), dtype=torch.float16, device="cuda"),
        lambda: torch.randn(
            (2, 128, 128), dtype=torch.bfloat16, device="cuda"
        ).transpose(1, 2),
        lambda: torch.randn((128, 128), dtype=torch.bfloat16, device="cuda"),
        lambda: torch.randn((2, 48, 128), dtype=torch.bfloat16, device="cuda"),
        lambda: torch.randn((2, 128, 160), dtype=torch.bfloat16, device="cuda"),
    ],
)
def test_dual_pack_rejects_unsupported_inputs(make):
    with pytest.raises(ValueError):
        quantize_mxfp4_weight_dual(make())
