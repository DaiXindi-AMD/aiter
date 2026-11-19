# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Grouped dual producer matches independent row/column MXFP4 quantization."""

import pytest
import torch

from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import (
    quantize_grouped_mxfp4_colwise,
)
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_concurrent,
    quantize_grouped_mxfp4_dual,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("counts", [(0, 1, 31, 33, 129), (128, 256), (0, 0)])
@pytest.mark.parametrize("features", [128, 768, 1536, 2048])
@pytest.mark.parametrize("zeros", [False, True])
def test_grouped_mxfp4_dual_byte_exact(counts, features, zeros):
    torch.manual_seed(85)
    x = torch.randn((sum(counts), features), dtype=torch.bfloat16, device="cuda")
    if zeros:
        x.zero_()
    metadata = prepare_mxfp4_wgrad_metadata(
        counts, x.device, pad_multiple=256, backend="gluon"
    )

    row, col = quantize_grouped_mxfp4_dual(x, metadata)
    col_ref = quantize_grouped_mxfp4_colwise(x, metadata)

    if x.shape[0]:
        row_ref = dynamic_mxfp4_quant(x)
        assert torch.equal(row[0], row_ref[0])
        assert torch.equal(row[1], row_ref[1])
    else:
        assert row[0].shape == (0, features // 2)
        assert row[1].shape == (0, features // 32)
    assert torch.equal(col[0], col_ref[0])
    assert torch.equal(col[1], col_ref[1])
    assert row[0].is_contiguous() and row[1].is_contiguous()
    assert col[0].is_contiguous() and col[1].is_contiguous()

    concurrent_row, concurrent_col, ready = quantize_grouped_mxfp4_concurrent(
        x, metadata
    )
    torch.cuda.current_stream().wait_event(ready)
    torch.cuda.synchronize()
    assert torch.equal(concurrent_row[0], row[0])
    assert torch.equal(concurrent_row[1], row[1])
    assert torch.equal(concurrent_col[0], col[0])
    assert torch.equal(concurrent_col[1], col[1])
