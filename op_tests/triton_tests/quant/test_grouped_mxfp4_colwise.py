# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Columnwise packs match the runtime's exact MXFP4 quantization recipe."""

import pytest
import torch

from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import prepare_mxfp4_wgrad_metadata
from aiter.ops.triton.quant.grouped_mxfp4_colwise import quantize_grouped_mxfp4_colwise
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant
from op_tests.triton_tests.utils.mxfp4_wgrad_reference import decode_columnwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("counts", [(0, 1, 31, 33, 129), (128, 256), (0, 0)])
@pytest.mark.parametrize("features", [63, 65, 127, 128, 129, 255, 257])
@pytest.mark.parametrize("zeros", [False, True])
def test_grouped_mxfp4_colwise_byte_exact(counts, features, zeros):
    torch.manual_seed(9)
    x = torch.randn((sum(counts), features), dtype=torch.bfloat16, device="cuda")
    if zeros:
        x.zero_()
    meta = prepare_mxfp4_wgrad_metadata(counts, x.device)
    actual = quantize_grouped_mxfp4_colwise(x, meta)
    real, padded = 0, 0
    decoded = decode_columnwise(actual)
    for count in counts:
        length = (
            (count + meta.pad_multiple - 1) // meta.pad_multiple
        ) * meta.pad_multiple
        if length:
            reference = torch.zeros(
                (features, length), dtype=torch.bfloat16, device="cuda"
            )
            reference[:, :count] = x[real : real + count].T
            payload, scales = dynamic_mxfp4_quant(reference)
            assert torch.equal(
                actual[0][:, padded // 2 : (padded + length) // 2], payload
            )
            assert torch.equal(
                actual[1][:, padded // 32 : (padded + length) // 32], scales
            )
            assert (
                torch.count_nonzero(decoded[:, padded + count : padded + length]) == 0
            )
        real += count
        padded += length
    assert actual[0].is_contiguous() and actual[1].is_contiguous()


def test_grouped_mxfp4_colwise_independent_expert_scales():
    x = torch.full((65, 63), 0.125, dtype=torch.bfloat16, device="cuda")
    x[33:] = 256.0
    meta = prepare_mxfp4_wgrad_metadata((33, 32), x.device)
    payload, scales = quantize_grouped_mxfp4_colwise(x, meta)
    assert torch.all(scales[:, 0] != scales[:, meta.pad_multiple // 32])
    assert torch.isfinite(decode_columnwise((payload, scales))).all()
