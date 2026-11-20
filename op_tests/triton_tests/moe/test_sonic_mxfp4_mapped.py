# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mapped pre-routed MXFP4 experts match explicit expert-major materialization."""

import pytest
import torch

from aiter.ops.triton.moe.sonic_mxfp4 import (
    MAPPED_DUAL_QUANT_API_VERSION,
    moe_pre_routed_a4w4,
    moe_pre_routed_quantized_mapped,
    pack_pre_routed_weights,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _weights(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    w1 = (
        torch.randn(
            (3, 768, 768), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn(
            (3, 384, 768), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.02
    ).requires_grad_()
    return w1, w2


def _microbatch(source_rows, counts, gather_values, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    source = torch.randn(
        (source_rows, 768),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).requires_grad_()
    gather_map = torch.tensor(gather_values, device="cuda", dtype=torch.int32)
    scores = torch.rand(
        (sum(counts),), device="cuda", dtype=torch.float32, generator=generator
    ).requires_grad_()
    dout = torch.randn(
        (sum(counts), 768),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return source, scores, tuple(counts), gather_map, dout


def _mapped(source, scores, counts, gather_map, weights, packed, main_grads=None):
    return moe_pre_routed_quantized_mapped(
        source,
        scores,
        counts,
        gather_map,
        *weights,
        packed,
        recipe="a4w4",
        wgrad_mode="mxfp4",
        main_grads=main_grads,
        wgrad_fallback="error",
    )


def _explicit(source, scores, counts, gather_map, weights, packed, main_grads=None):
    expert_x = source.index_select(0, gather_map.long())
    return expert_x, moe_pre_routed_a4w4(
        expert_x,
        scores,
        counts,
        *weights,
        packed,
        wgrad_mode="mxfp4",
        main_grads=main_grads,
    )


def test_mapped_two_microbatches_match_explicit_materialization(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    monkeypatch.setenv("AITER_GROUPED_MXFP4_MAPPED", "1")
    monkeypatch.setenv("SONIC_MOE_MXFP4_DUAL_QUANT", "kernel")
    mapped_weights = _weights(901)
    reference_weights = tuple(
        weight.detach().clone().requires_grad_() for weight in mapped_weights
    )
    mapped_packed = pack_pre_routed_weights(*mapped_weights)
    reference_packed = pack_pre_routed_weights(*reference_weights)
    specifications = (
        (5, (0, 3, 5), (0, 1, 1, 4, 2, 2, 2, 3), 902),
        (4, (2, 0, 4), (3, 3, 0, 1, 1, 2), 903),
    )
    mapped_batches, reference_batches = [], []
    for source_rows, counts, gather_values, seed in specifications:
        source, scores, counts, gather_map, dout = _microbatch(
            source_rows, counts, gather_values, seed
        )
        reference_source = source.detach().clone().requires_grad_()
        reference_scores = scores.detach().clone().requires_grad_()
        mapped_out = _mapped(
            source,
            scores,
            counts,
            gather_map,
            mapped_weights,
            mapped_packed,
        )
        expert_x, reference_out = _explicit(
            reference_source,
            reference_scores,
            counts,
            gather_map,
            reference_weights,
            reference_packed,
        )
        torch.testing.assert_close(mapped_out, reference_out, rtol=0, atol=0)
        mapped_batches.append((source, scores, gather_map, mapped_out, dout))
        reference_batches.append(
            (
                reference_source,
                reference_scores,
                gather_map,
                expert_x,
                reference_out,
                dout,
            )
        )

    for source, *_ in mapped_batches:
        with torch.no_grad():
            source.copy_(torch.randn_like(source))
    sum((out * dout).sum() for _, _, _, out, dout in mapped_batches).backward()
    sum((out * dout).sum() for _, _, _, _, out, dout in reference_batches).backward()

    for mapped_batch, reference_batch in zip(mapped_batches, reference_batches):
        source, scores, _gather_map, _out, _dout = mapped_batch
        reference_source, reference_scores, _, _expert_x, _out, _dout = reference_batch
        torch.testing.assert_close(source.grad, reference_source.grad, rtol=0, atol=0)
        torch.testing.assert_close(scores.grad, reference_scores.grad, rtol=0, atol=0)
    for actual, expected in zip(mapped_weights, reference_weights):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_mapped_main_grad_and_source_overwrite_match_explicit(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    monkeypatch.setenv("AITER_GROUPED_MXFP4_MAPPED", "1")
    monkeypatch.setenv("SONIC_MOE_MXFP4_DUAL_QUANT", "kernel")
    mapped_weights = _weights(911)
    reference_weights = tuple(
        weight.detach().clone().requires_grad_() for weight in mapped_weights
    )
    source, scores, counts, gather_map, dout = _microbatch(
        5, (0, 3, 5), (0, 1, 1, 4, 2, 2, 2, 3), 912
    )
    reference_source = source.detach().clone().requires_grad_()
    reference_scores = scores.detach().clone().requires_grad_()
    mapped_grads = tuple(
        torch.full_like(weight, 3.0, dtype=torch.float32) for weight in mapped_weights
    )
    reference_grads = tuple(grad.clone() for grad in mapped_grads)
    mapped_out = _mapped(
        source,
        scores,
        counts,
        gather_map,
        mapped_weights,
        pack_pre_routed_weights(*mapped_weights),
        mapped_grads,
    )
    _expert_x, reference_out = _explicit(
        reference_source,
        reference_scores,
        counts,
        gather_map,
        reference_weights,
        pack_pre_routed_weights(*reference_weights),
        reference_grads,
    )
    with torch.no_grad():
        source.copy_(torch.randn_like(source))
    mapped_out.backward(dout)
    reference_out.backward(dout)
    for actual, expected in zip(mapped_grads, reference_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(weight.grad is None for weight in mapped_weights)


@pytest.mark.parametrize("mutation", ["map", "source_structure"])
def test_mapped_mutation_rejects_before_main_grad_write(monkeypatch, mutation):
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    monkeypatch.setenv("AITER_GROUPED_MXFP4_MAPPED", "1")
    weights = _weights(921)
    source, scores, counts, gather_map, dout = _microbatch(
        5, (0, 3, 5), (0, 1, 1, 4, 2, 2, 2, 3), 922
    )
    main_grads = tuple(
        torch.full_like(weight, 3.0, dtype=torch.float32) for weight in weights
    )
    before = tuple(grad.clone() for grad in main_grads)
    out = _mapped(
        source,
        scores,
        counts,
        gather_map,
        weights,
        pack_pre_routed_weights(*weights),
        main_grads,
    )
    if mutation == "map":
        gather_map.add_(1)
        error = "gather_map was mutated"
    else:
        with torch.no_grad():
            source.set_(torch.empty_like(source))
        error = "snapshot source structure changed"
    with pytest.raises(ValueError, match=error):
        out.backward(dout)
    for actual, expected in zip(main_grads, before):
        assert torch.equal(actual, expected)


def test_mapped_api_version_and_default_off(monkeypatch):
    assert MAPPED_DUAL_QUANT_API_VERSION == 1
    if get_arch() != "gfx950":
        pytest.skip("gfx950 required")
    monkeypatch.delenv("AITER_GROUPED_MXFP4_MAPPED", raising=False)
    weights = _weights(931)
    source, scores, counts, gather_map, _dout = _microbatch(1, (0, 0, 1), (0,), 932)
    with pytest.raises(RuntimeError, match="disabled"):
        moe_pre_routed_quantized_mapped(
            source,
            scores,
            counts,
            gather_map,
            *weights,
            recipe="a4w4",
            wgrad_mode="mxfp4",
        )
