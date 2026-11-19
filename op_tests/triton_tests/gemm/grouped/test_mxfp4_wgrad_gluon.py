# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon variable-K FP4 contract through the public wrapper."""

import pytest
import torch

from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    MXFP4_WGRAD_BACKENDS,
    check_mxfp4_wgrad_support,
    grouped_mxfp4_wgrad,
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import quantize_grouped_mxfp4_colwise
from aiter.ops.triton.utils.mxfp4_wgrad import (
    get_mxfp4_wgrad_stats,
    reset_mxfp4_wgrad_stats,
)
from op_tests.triton_tests.utils.mxfp4_wgrad_reference import (
    bf16_wgrad_reference,
    error_metrics,
    same_fp4_wgrad_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "counts,n,m",
    [
        ((0, 1, 31, 33, 129), 63, 95),
        ((0, 0, 0), 65, 97),
        ((256, 257, 0), 256, 256),
        ((7, 513, 2), 257, 255),
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_mxfp4_wgrad_gluon_same_operands(counts, n, m, transpose, record_property):
    torch.manual_seed(29)
    x = torch.randn((sum(counts), n), dtype=torch.bfloat16, device="cuda") * 0.05
    dy = torch.randn((sum(counts), m), dtype=torch.bfloat16, device="cuda") * 0.08
    meta = prepare_mxfp4_wgrad_metadata(counts, x.device, backend="gluon")
    assert meta.pad_multiple == 256 and "gluon" in MXFP4_WGRAD_BACKENDS
    xp, dyp = quantize_grouped_mxfp4_colwise(x, meta), quantize_grouped_mxfp4_colwise(
        dy, meta
    )
    shape = (len(counts), n, m) if transpose else (len(counts), m, n)
    out = torch.full(shape, float("nan"), dtype=torch.float32, device="cuda")
    reset_mxfp4_wgrad_stats()
    grouped_mxfp4_wgrad(
        x,
        dy,
        counts,
        out=out,
        metadata=meta,
        quantized_operands=(xp, dyp),
        transpose_output=transpose,
        backend="gluon",
    )
    expected = same_fp4_wgrad_reference(xp, dyp, meta, transpose_output=transpose)
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)
    assert torch.isfinite(out).all() and out.is_contiguous()
    if sum(counts):
        actual = get_mxfp4_wgrad_stats()["last_config"]
        assert actual["logical_shape"] == (len(counts), m, n)
        assert actual["physical_shape"] == (
            (len(counts), n, m) if transpose else (len(counts), m, n)
        )
        assert actual["config"]["BLOCK_SIZE_M"] == 256
        assert actual["config"]["BLOCK_SIZE_N"] == 128
        assert actual["config"]["BLOCK_SIZE_K"] == 256
        physical_m, physical_n = (n, m) if transpose else (m, n)
        tiles_per_expert = ((physical_m + 255) // 256) * ((physical_n + 127) // 128)
        record_property("physical_tiles_per_expert", tiles_per_expert)
        if m == 255 and n == 257 and transpose:
            assert (
                tiles_per_expert == 4
            )  # Logical pre-swap grid would issue only three tiles.
        assert actual["logical_transpose_output"] is transpose
        assert actual["physical_operand_roles"] == (
            ("X_colwise", "dY_colwise") if transpose else ("dY_colwise", "X_colwise")
        )
    triton = grouped_mxfp4_wgrad(
        x,
        dy,
        counts,
        metadata=meta,
        quantized_operands=(xp, dyp),
        transpose_output=transpose,
        backend="triton",
    )
    torch.testing.assert_close(out, triton, rtol=2e-4, atol=2e-5)
    for key, value in error_metrics(
        bf16_wgrad_reference(x, dy, counts, transpose_output=transpose), out
    ).items():
        record_property("bf16_" + key, value)
    if sum(counts):
        assert get_mxfp4_wgrad_stats()["backend_calls"]["gluon_e2m1_e2m1"] == 1


@pytest.mark.parametrize("transpose", [False, True])
def test_mxfp4_wgrad_gluon_two_mb_sentinel_and_zero(transpose):
    n, m = 65, 97
    shape = (3, n, m) if transpose else (3, m, n)
    out = torch.full(shape, 3.0, dtype=torch.float32, device="cuda")
    expected = out.clone()
    for counts in ((0, 33, 129), (17, 0, 257), (0, 0, 0)):
        x = torch.randn((sum(counts), n), dtype=torch.bfloat16, device="cuda") * 0.05
        dy = torch.randn((sum(counts), m), dtype=torch.bfloat16, device="cuda") * 0.08
        meta = prepare_mxfp4_wgrad_metadata(counts, x.device, backend="gluon")
        xp, dyp = quantize_grouped_mxfp4_colwise(
            x, meta
        ), quantize_grouped_mxfp4_colwise(dy, meta)
        before = out.clone()
        grouped_mxfp4_wgrad(
            x,
            dy,
            counts,
            out=out,
            accumulate=True,
            metadata=meta,
            quantized_operands=(xp, dyp),
            transpose_output=transpose,
            backend="gluon",
        )
        expected += same_fp4_wgrad_reference(xp, dyp, meta, transpose_output=transpose)
        torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)
        for expert, count in enumerate(counts):
            if not count:
                assert torch.equal(out[expert], before[expert])
    out.zero_()
    assert torch.count_nonzero(out) == 0


def test_mxfp4_wgrad_gluon_rejects_wrong_padding_before_write():
    x = torch.randn((33, 65), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((33, 97), dtype=torch.bfloat16, device="cuda")
    meta = prepare_mxfp4_wgrad_metadata((0, 33), x.device, pad_multiple=128)
    out = torch.full((2, 97, 65), 3.0, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="padding"):
        grouped_mxfp4_wgrad(
            x, dy, (0, 33), out=out, accumulate=True, metadata=meta, backend="gluon"
        )
    assert torch.all(out == 3.0)


def test_mxfp4_wgrad_gluon_rejects_backend_and_operand_abi():
    x = torch.randn((33, 65), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((33, 97), dtype=torch.bfloat16, device="cuda")
    out = torch.full((2, 97, 65), 3.0, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="backend"):
        check_mxfp4_wgrad_support(x.device, 97, 65, backend="unknown")
    with pytest.raises(ValueError, match="BF16"):
        grouped_mxfp4_wgrad(
            x.float(), dy, (0, 33), out=out, accumulate=True, backend="gluon"
        )
    assert torch.all(out == 3.0)
