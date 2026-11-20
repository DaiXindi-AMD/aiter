# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public MXFP4 Wgrad correctness and accumulation contract."""

import pytest
import torch

from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    grouped_mxfp4_wgrad,
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import quantize_grouped_mxfp4_colwise
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_dual_mapped,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.utils.mxfp4_wgrad_reference import (
    bf16_wgrad_reference,
    error_metrics,
    same_fp4_wgrad_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "counts", [(0, 1, 31, 33, 129), (0, 0, 0), (128, 256, 0), (7, 513, 2)]
)
@pytest.mark.parametrize("features", [(63, 95), (128, 128)])
@pytest.mark.parametrize("transpose", [False, True])
def test_mxfp4_wgrad_same_operands(counts, features, transpose, record_property):
    torch.manual_seed(7)
    n, m = features
    x = torch.randn((sum(counts), n), dtype=torch.bfloat16, device="cuda") * 0.05
    dy = torch.randn((sum(counts), m), dtype=torch.bfloat16, device="cuda") * 0.08
    metadata = prepare_mxfp4_wgrad_metadata(counts, x.device)
    xp = quantize_grouped_mxfp4_colwise(x, metadata)
    dyp = quantize_grouped_mxfp4_colwise(dy, metadata)
    out = grouped_mxfp4_wgrad(
        x,
        dy,
        counts,
        metadata=metadata,
        quantized_operands=(xp, dyp),
        transpose_output=transpose,
    )
    expected = same_fp4_wgrad_reference(xp, dyp, metadata, transpose_output=transpose)
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)
    assert out.dtype == torch.float32 and out.is_contiguous()
    assert torch.isfinite(out).all()
    bf16 = bf16_wgrad_reference(x, dy, counts, transpose_output=transpose)
    assert torch.isfinite(bf16).all()
    for key, value in error_metrics(bf16, out).items():
        record_property("bf16_" + key, value)
    for expert, count in enumerate(counts):
        if count == 0:
            assert torch.count_nonzero(out[expert]) == 0
    direct = grouped_mxfp4_wgrad(
        x, dy, counts, metadata=metadata, transpose_output=transpose
    )
    torch.testing.assert_close(out, direct, rtol=0, atol=0)


@pytest.mark.parametrize("transpose", [False, True])
def test_mxfp4_wgrad_two_microbatches_and_empty(transpose):
    n, m, e = 65, 97, 3
    shape = (e, n, m) if transpose else (e, m, n)
    main_grad = torch.randn(shape, dtype=torch.float32, device="cuda")
    expected = main_grad.clone()
    for counts in ((0, 33, 129), (17, 0, 256), (0, 0, 0)):
        x = torch.randn((sum(counts), n), dtype=torch.bfloat16, device="cuda") * 0.05
        dy = torch.randn((sum(counts), m), dtype=torch.bfloat16, device="cuda") * 0.08
        metadata = prepare_mxfp4_wgrad_metadata(counts, x.device)
        xp, dyp = quantize_grouped_mxfp4_colwise(
            x, metadata
        ), quantize_grouped_mxfp4_colwise(dy, metadata)
        before = main_grad.clone()
        result = grouped_mxfp4_wgrad(
            x,
            dy,
            counts,
            out=main_grad,
            accumulate=True,
            metadata=metadata,
            quantized_operands=(xp, dyp),
            transpose_output=transpose,
        )
        expected += same_fp4_wgrad_reference(
            xp, dyp, metadata, transpose_output=transpose
        )
        assert result is main_grad
        torch.testing.assert_close(main_grad, expected, rtol=2e-4, atol=2e-5)
        for expert, count in enumerate(counts):
            if not count:
                assert torch.equal(main_grad[expert], before[expert])
    main_grad.zero_()
    assert torch.count_nonzero(main_grad) == 0


def test_mxfp4_wgrad_rejects_bad_contract_before_write():
    x = torch.randn((33, 64), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((33, 64), dtype=torch.bfloat16, device="cuda")
    out = torch.full((2, 64, 64), 3.0, dtype=torch.float32, device="cuda")
    for bad_counts in ((1, 31), (True, 32), (-1, 34), ()):
        with pytest.raises((ValueError, TypeError)):
            grouped_mxfp4_wgrad(x, dy, bad_counts, out=out, accumulate=True)
        assert torch.all(out == 3.0)
    with pytest.raises(ValueError, match="initialized"):
        grouped_mxfp4_wgrad(x, dy, (1, 32), accumulate=True)
    with pytest.raises(ValueError, match="FP32"):
        grouped_mxfp4_wgrad(x, dy, (1, 32), out=out.bfloat16())
    stale = prepare_mxfp4_wgrad_metadata((2, 31), x.device)
    with pytest.raises(ValueError, match="routing"):
        grouped_mxfp4_wgrad(x, dy, (1, 32), out=out, accumulate=True, metadata=stale)
    assert torch.all(out == 3.0)


def test_mxfp4_wgrad_rejects_stale_bound_pack():
    x = torch.randn((33, 64), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((33, 96), dtype=torch.bfloat16, device="cuda")
    old = prepare_mxfp4_wgrad_metadata((1, 32), x.device)
    new = prepare_mxfp4_wgrad_metadata((32, 1), x.device)
    assert old.padded_rows == new.padded_rows
    xp, dyp = quantize_grouped_mxfp4_colwise(x, old), quantize_grouped_mxfp4_colwise(
        dy, old
    )
    out = torch.full((2, 96, 64), 3.0, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="producer metadata"):
        grouped_mxfp4_wgrad(
            x,
            dy,
            (32, 1),
            out=out,
            accumulate=True,
            metadata=new,
            quantized_operands=(xp, dyp),
        )
    with pytest.raises(ValueError, match="source tensor"):
        grouped_mxfp4_wgrad(
            x.clone(),
            dy,
            (1, 32),
            out=out,
            accumulate=True,
            metadata=old,
            quantized_operands=(xp, dyp),
        )
    x.add_(1)
    with pytest.raises(ValueError, match="source was mutated"):
        grouped_mxfp4_wgrad(
            x,
            dy,
            (1, 32),
            out=out,
            accumulate=True,
            metadata=old,
            quantized_operands=(xp, dyp),
        )
    assert torch.all(out == 3.0)


@pytest.mark.parametrize(
    "field", ["real_offsets", "padded_offsets", "prep_blocks", "compact_blocks"]
)
def test_mxfp4_wgrad_rejects_metadata_corruption(field):
    from dataclasses import replace

    x = torch.randn((33, 64), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((33, 64), dtype=torch.bfloat16, device="cuda")
    meta = prepare_mxfp4_wgrad_metadata((1, 32), x.device)
    value = getattr(meta, field)
    wrong = replace(meta, **{field: value.to(torch.int64)})
    out = torch.full((2, 64, 64), 3.0, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="int32 shape/device/layout"):
        grouped_mxfp4_wgrad(x, dy, (1, 32), out=out, accumulate=True, metadata=wrong)
    value.add_(1)
    with pytest.raises(ValueError, match="mutated"):
        grouped_mxfp4_wgrad(x, dy, (1, 32), out=out, accumulate=True, metadata=meta)
    assert torch.all(out == 3.0)


def test_mxfp4_wgrad_poison_overwrite():
    x = torch.randn((33, 65), dtype=torch.bfloat16, device="cuda") * 0.05
    dy = torch.randn((33, 97), dtype=torch.bfloat16, device="cuda") * 0.08
    out = torch.full((2, 97, 65), float("nan"), dtype=torch.float32, device="cuda")
    meta = prepare_mxfp4_wgrad_metadata((0, 33), x.device)
    xp, dyp = quantize_grouped_mxfp4_colwise(x, meta), quantize_grouped_mxfp4_colwise(
        dy, meta
    )
    grouped_mxfp4_wgrad(
        x, dy, (0, 33), out=out, metadata=meta, quantized_operands=(xp, dyp)
    )
    torch.testing.assert_close(
        out, same_fp4_wgrad_reference(xp, dyp, meta), rtol=2e-4, atol=2e-5
    )
    assert torch.isfinite(out).all() and torch.count_nonzero(out[0]) == 0
    out.fill_(float("nan"))
    grouped_mxfp4_wgrad(x[:0], dy[:0], (0, 0), out=out)
    assert torch.count_nonzero(out) == 0


def test_mxfp4_wgrad_padding_alignment_contract():
    counts = (0, 31, 33, 129)
    x = torch.randn((sum(counts), 65), dtype=torch.bfloat16, device="cuda")
    meta = prepare_mxfp4_wgrad_metadata(counts, x.device)
    payload, scales = quantize_grouped_mxfp4_colwise(x, meta)
    assert torch.all(meta.padded_offsets % meta.pad_multiple == 0)
    assert payload.stride(0) % (meta.pad_multiple // 2) == 0
    assert scales.stride(0) % (meta.pad_multiple // 32) == 0


def test_mxfp4_wgrad_accepts_mapped_producer_packs(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    monkeypatch.setenv("AITER_GROUPED_MXFP4_MAPPED", "1")
    counts = (0, 1, 32)
    source_map = (torch.arange(sum(counts), device="cuda") % 17).to(torch.int32)
    x = torch.randn((17, 768), dtype=torch.bfloat16, device="cuda") * 0.05
    dy = torch.randn((sum(counts), 768), dtype=torch.bfloat16, device="cuda") * 0.08
    metadata = prepare_mxfp4_wgrad_metadata(counts, x.device, pad_multiple=256)
    xp = quantize_grouped_mxfp4_dual_mapped(x, source_map, metadata)[1]
    dyp = quantize_grouped_mxfp4_colwise(dy, metadata)

    out = grouped_mxfp4_wgrad(
        x,
        dy,
        counts,
        metadata=metadata,
        quantized_operands=(xp, dyp),
    )
    expected = same_fp4_wgrad_reference(xp, dyp, metadata)
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)


def test_mxfp4_wgrad_mapped_snapshot_allows_source_reuse(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    monkeypatch.setenv("AITER_GROUPED_MXFP4_MAPPED", "1")
    counts = (0, 1, 32)
    source_map = (torch.arange(sum(counts), device="cuda") % 17).to(torch.int32)
    x = torch.randn((17, 768), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((sum(counts), 768), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(counts, x.device, pad_multiple=256)
    xp = quantize_grouped_mxfp4_dual_mapped(x, source_map, metadata)[1]
    dyp = quantize_grouped_mxfp4_colwise(dy, metadata)
    expected = same_fp4_wgrad_reference(xp, dyp, metadata)
    x.copy_(torch.randn_like(x))
    out = grouped_mxfp4_wgrad(
        x,
        dy,
        counts,
        metadata=metadata,
        quantized_operands=(xp, dyp),
    )
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)


def test_mxfp4_wgrad_rejects_mapped_map_mutation_before_write(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    monkeypatch.setenv("AITER_GROUPED_MXFP4_MAPPED", "1")
    counts = (0, 1, 32)
    source_map = (torch.arange(sum(counts), device="cuda") % 17).to(torch.int32)
    x = torch.randn((17, 768), dtype=torch.bfloat16, device="cuda")
    dy = torch.randn((sum(counts), 768), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(counts, x.device, pad_multiple=256)
    xp = quantize_grouped_mxfp4_dual_mapped(x, source_map, metadata)[1]
    dyp = quantize_grouped_mxfp4_colwise(dy, metadata)
    out = torch.full((3, 768, 768), 3.0, dtype=torch.float32, device="cuda")
    source_map.add_(1)
    with pytest.raises(ValueError, match="source map was mutated"):
        grouped_mxfp4_wgrad(
            x,
            dy,
            counts,
            out=out,
            accumulate=True,
            metadata=metadata,
            quantized_operands=(xp, dyp),
        )
    assert torch.all(out == 3.0)
