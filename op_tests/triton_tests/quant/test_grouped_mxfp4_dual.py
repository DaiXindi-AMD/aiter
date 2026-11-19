# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Grouped dual producer matches independent row/column MXFP4 quantization."""

import gc

import pytest
import torch

import aiter.ops.triton.quant.grouped_mxfp4_dual as dual_module
from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import (
    quantize_grouped_mxfp4_colwise,
)
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_concurrent,
    quantize_grouped_mxfp4_dual,
    quantize_grouped_mxfp4_dual_v2,
    quantize_grouped_mxfp4_dual_v3,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton.arch_info import get_arch

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
    assert concurrent_row[0].is_contiguous()
    assert concurrent_row[1].is_contiguous()
    assert torch.equal(concurrent_row[0], row[0])
    assert torch.equal(concurrent_row[1], row[1])
    assert torch.equal(concurrent_col[0], col[0])
    assert torch.equal(concurrent_col[1], col[1])


def test_grouped_mxfp4_concurrent_records_delayed_consumer_stream():
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is required for the allocator regression")

    torch.manual_seed(85)
    torch.cuda.empty_cache()
    x = torch.randn((33, 640), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        (33,), x.device, pad_multiple=256, backend="gluon"
    )
    _row, col, ready = quantize_grouped_mxfp4_concurrent(x, metadata)
    current = torch.cuda.current_stream(x.device)
    current.wait_event(ready)
    expected_payload = col.payload.clone()
    expected_scales = col.scales.clone()
    current.synchronize()

    blocker = torch.cuda.Stream(device=x.device)
    with torch.cuda.stream(blocker):
        torch.cuda._sleep(20_000_000)
        release = torch.cuda.Event()
        release.record(blocker)
    current.wait_event(release)
    actual_payload = torch.empty_like(col.payload)
    actual_scales = torch.empty_like(col.scales)
    actual_payload.copy_(col.payload)
    actual_scales.copy_(col.scales)
    col.record_stream(current)

    specs = (
        (col.payload.shape, col.payload.dtype),
        (col.scales.shape, col.scales.dtype),
    )
    del col
    gc.collect()
    side = dual_module._side_stream(x.device)
    with torch.cuda.stream(side):
        recycled = [
            torch.empty(shape, dtype=dtype, device=x.device) for shape, dtype in specs
        ]
        for tensor in recycled:
            tensor.fill_(0xA5)

    current.synchronize()
    assert torch.equal(actual_payload, expected_payload)
    assert torch.equal(actual_scales, expected_scales)


@pytest.mark.parametrize("features", [768, 1536, 2048])
@pytest.mark.parametrize("zeros", [False, True])
def test_grouped_mxfp4_dual_v2_byte_exact(features, zeros, monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    counts = (0, 31, 33, 129, 513)
    torch.manual_seed(91)
    x = torch.randn((sum(counts), features), dtype=torch.bfloat16, device="cuda")
    if zeros:
        x.zero_()
    metadata = prepare_mxfp4_wgrad_metadata(
        counts, x.device, pad_multiple=256, backend="gluon"
    )

    expected = quantize_grouped_mxfp4_dual(x, metadata)
    actual = quantize_grouped_mxfp4_dual_v2(x, metadata)
    for result, reference in zip(
        (*actual[0], *actual[1]), (*expected[0], *expected[1])
    ):
        assert torch.equal(result, reference)

    monkeypatch.setenv("AITER_GROUPED_MXFP4_DUAL_V2", "1")
    selected = quantize_grouped_mxfp4_dual(x, metadata)
    for result, reference in zip(
        (*selected[0], *selected[1]), (*expected[0], *expected[1])
    ):
        assert torch.equal(result, reference)


def test_grouped_mxfp4_dual_v2_rejects_unconfigured_shape():
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    x = torch.randn((32, 128), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        (32,), x.device, pad_multiple=256, backend="gluon"
    )
    with pytest.raises(ValueError, match="feature size"):
        quantize_grouped_mxfp4_dual_v2(x, metadata)


@pytest.mark.parametrize("features", [768, 1536, 2048])
@pytest.mark.parametrize("zeros", [False, True])
def test_grouped_mxfp4_dual_v3_byte_exact(features, zeros, monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    counts = (0, 31, 33, 129, 513)
    torch.manual_seed(97)
    x = torch.randn((sum(counts), features), dtype=torch.bfloat16, device="cuda")
    if zeros:
        x.zero_()
    metadata = prepare_mxfp4_wgrad_metadata(
        counts, x.device, pad_multiple=256, backend="gluon"
    )

    expected = quantize_grouped_mxfp4_dual(x, metadata)
    actual = quantize_grouped_mxfp4_dual_v3(x, metadata)
    for result, reference in zip(
        (*actual[0], *actual[1]), (*expected[0], *expected[1])
    ):
        assert torch.equal(result, reference)

    monkeypatch.setenv("AITER_GROUPED_MXFP4_DUAL_V3", "1")
    selected = quantize_grouped_mxfp4_dual(x, metadata)
    for result, reference in zip(
        (*selected[0], *selected[1]), (*expected[0], *expected[1])
    ):
        assert torch.equal(result, reference)


def test_grouped_mxfp4_dual_v3_rejects_unconfigured_shape():
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    x = torch.randn((32, 128), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        (32,), x.device, pad_multiple=256, backend="gluon"
    )
    with pytest.raises(ValueError, match="feature size"):
        quantize_grouped_mxfp4_dual_v3(x, metadata)


def test_grouped_mxfp4_dual_v3_portable_quant_compiles():
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    x = torch.zeros((1, 768), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        (1,), x.device, pad_multiple=256, backend="gluon"
    )

    expected = quantize_grouped_mxfp4_dual(x, metadata)
    actual = quantize_grouped_mxfp4_dual_v3(x, metadata)
    for result, reference in zip(
        (*actual[0], *actual[1]), (*expected[0], *expected[1])
    ):
        assert torch.equal(result, reference)


def test_grouped_mxfp4_dual_v3_rounding_edges():
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    edge = torch.tensor(
        [
            -6.0,
            -5.0,
            -4.0,
            -3.5,
            -3.0,
            -2.5,
            -2.0,
            -1.75,
            -1.5,
            -1.25,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            -0.0,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            0.125,
            -0.125,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    scale = torch.exp2(
        (torch.arange(33, device="cuda", dtype=torch.float32) % 7) - 3
    ).to(torch.bfloat16)
    x = edge.repeat(33, 24) * scale[:, None]
    metadata = prepare_mxfp4_wgrad_metadata(
        (0, 1, 32), x.device, pad_multiple=256, backend="gluon"
    )

    expected = quantize_grouped_mxfp4_dual(x, metadata)
    actual = quantize_grouped_mxfp4_dual_v3(x, metadata)
    for result, reference in zip(
        (*actual[0], *actual[1]), (*expected[0], *expected[1])
    ):
        assert torch.equal(result, reference)


def test_grouped_mxfp4_dual_rejects_multiple_candidates(monkeypatch):
    x = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        (0,), x.device, pad_multiple=256, backend="gluon"
    )
    monkeypatch.setenv("AITER_GROUPED_MXFP4_DUAL_V2", "1")
    monkeypatch.setenv("AITER_GROUPED_MXFP4_DUAL_V3", "1")
    with pytest.raises(ValueError, match="only one"):
        quantize_grouped_mxfp4_dual(x, metadata)
