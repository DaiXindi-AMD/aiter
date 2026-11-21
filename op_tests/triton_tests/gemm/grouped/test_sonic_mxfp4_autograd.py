# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerics and gradient checks for the pre-routed MXFP4 expert recipes."""

import pytest
import torch
from test_sonic_mxfp4 import decode

from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
    pack_grouped_mxfp4_weights,
    prepare_expert_offsets,
)
from aiter.ops.triton.moe.sonic_mxfp4 import (
    RECIPES as EXPERT_RECIPES,
)
from aiter.ops.triton.moe.sonic_mxfp4 import (
    moe_pre_routed_a4w4,
    moe_pre_routed_a8w4,
    moe_pre_routed_a8w4f,
    moe_pre_routed_w4a16,
    pack_pre_routed_weights,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant


def decode_mxfp8(payload, scales):
    """E4M3 values times their e8m0 1x32 block scale."""
    values = payload.to(torch.float32)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


def quantized_mxfp4(value):
    # The quantizer divides by the row count, so empty experts must be skipped.
    if value.numel() == 0:
        return value.float()
    payload, scales = dynamic_mxfp4_quant(value.to(torch.bfloat16).contiguous())
    return decode(payload, scales)


def quantized_mxfp8(value):
    if value.numel() == 0:
        return value.float()
    payload, scales = dynamic_mxfp8_quant(value.to(torch.bfloat16).contiguous())
    return decode_mxfp8(payload, scales)


def exact(value):
    return value.float()


_OPERATIONS = {
    "w4a16": moe_pre_routed_w4a16,
    "a4w4": moe_pre_routed_a4w4,
    "a8w4": moe_pre_routed_a8w4,
    "a8w4f": moe_pre_routed_a8w4f,
}
_ACTIVATION = {
    "bf16": exact,
    "mxfp4": quantized_mxfp4,
    "mxfp8": quantized_mxfp8,
}


def _recipe(name):
    forward_format, backward_format = EXPERT_RECIPES[name]
    return _OPERATIONS[name], _ACTIVATION[forward_format], _ACTIVATION[backward_format]


ENTRYPOINTS = sorted(_OPERATIONS)


def test_device_counts_build_identical_offsets():
    counts = (3, 0, 7, 5)
    device_counts = torch.tensor(counts, dtype=torch.int64, device="cuda")
    expected = torch.tensor((0, 3, 3, 10, 15), dtype=torch.int32, device="cuda")
    actual = prepare_expert_offsets(counts, device_counts.device, device_counts)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_device_counts_reject_invalid_contract():
    counts = (3, 0, 7, 5)
    device_counts = torch.tensor(counts[:-1], dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="device_counts"):
        prepare_expert_offsets(counts, device_counts.device, device_counts)


def test_pinned_host_counts_preserve_forward_and_backward():
    counts = (3, 0, 7, 5)
    torch.manual_seed(176)
    inputs = torch.randn((15, 64), dtype=torch.bfloat16, device="cuda")
    scores = torch.linspace(0, 1, 15, dtype=torch.float32, device="cuda")
    w1 = torch.randn((4, 64, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn((4, 32, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    dout = torch.randn_like(inputs)

    def run(count_arg):
        args = tuple(
            value.detach().clone().requires_grad_() for value in (inputs, w1, w2)
        )
        routed_scores = scores.detach().clone().requires_grad_()
        output = moe_pre_routed_a4w4(
            args[0], routed_scores, count_arg, args[1], args[2]
        )
        (output * dout).sum().backward()
        return output, tuple(
            value.grad for value in (args[0], routed_scores, *args[1:])
        )

    baseline = run(counts)
    candidate = run(torch.tensor(counts, dtype=torch.int64, pin_memory=True))
    for actual, expected in zip(candidate, baseline):
        if isinstance(actual, tuple):
            for actual_grad, expected_grad in zip(actual, expected):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _decoded(weight):
    return decode(*pack_grouped_mxfp4_weights(weight))


@pytest.mark.parametrize("recipe", ENTRYPOINTS)
def test_declared_surrogate_gradients_and_router(recipe):
    operation, activation_fwd, activation_bwd = _recipe(recipe)
    torch.manual_seed(174)
    counts = (3, 0, 7, 5)
    x = torch.randn((15, 64), dtype=torch.bfloat16, device="cuda").requires_grad_()
    w1 = (
        torch.randn((4, 64, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn((4, 32, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    ).requires_grad_()
    scores = torch.linspace(
        0, 1, 15, device="cuda", dtype=torch.float32
    ).requires_grad_()
    dy = torch.randn_like(x)
    out = operation(x, scores, counts, w1, w2)
    (out * dy).sum().backward()
    for p in (x, scores, w1, w2):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    with torch.no_grad():
        raw_y = operation(x, torch.ones_like(scores), counts, w1, w2)
        torch.testing.assert_close(
            scores.grad, (dy.float() * raw_y.float()).sum(-1), rtol=0, atol=0
        )
        wf1, wf2 = _decoded(w1.transpose(1, 2)), _decoded(w2.transpose(1, 2))
        wd1, wd2 = _decoded(w1), _decoded(w2)
        refs_out, refs_dx, refs_dw1, refs_dw2 = [], [], [], []
        start = 0
        for e, count in enumerate(counts):
            xx, dd = x[start : start + count].float(), dy[start : start + count].float()
            ss = scores[start : start + count, None]
            hh = (activation_fwd(xx) @ wf1[e].T).to(torch.bfloat16).float()
            gate, up = hh.chunk(2, dim=-1)
            sig = torch.sigmoid(gate)
            aa = (gate * sig * up).to(torch.bfloat16).float()
            yy = (activation_fwd(aa) @ wf2[e].T).to(torch.bfloat16).float()
            da = (activation_bwd(dd) @ wd2[e].T).to(torch.bfloat16).float() * ss
            dh = (
                torch.cat((da * up * sig * (1 + gate * (1 - sig)), da * gate * sig), -1)
                .to(torch.bfloat16)
                .float()
            )
            refs_out.append((yy * ss).to(torch.bfloat16))
            refs_dx.append((activation_bwd(dh) @ wd1[e].T).to(torch.bfloat16))
            refs_dw1.append((xx.T @ dh).to(torch.bfloat16))
            refs_dw2.append(
                (aa.T @ (dd * ss).to(torch.bfloat16).float()).to(torch.bfloat16)
            )
            start += count
        for actual, reference in (
            (out, torch.cat(refs_out)),
            (x.grad, torch.cat(refs_dx)),
            (w1.grad, torch.stack(refs_dw1)),
            (w2.grad, torch.stack(refs_dw2)),
        ):
            # Independent float32 oracle rounds at BF16 intermediate boundaries.
            relative = (
                actual.float() - reference.float()
            ).norm() / reference.float().norm().clamp_min(1e-20)
            assert relative < 0.03, (recipe, float(relative))
        assert torch.count_nonzero(w1.grad[1]) == 0
        assert torch.count_nonzero(w2.grad[1]) == 0
        assert torch.count_nonzero(x.grad[0]) == 0
        pointer = w2.data_ptr()
        w2.zero_()
        changed = operation(x, scores, counts, w1, w2)
        assert w2.data_ptr() == pointer
        assert torch.count_nonzero(changed) == 0
        assert not torch.equal(out, changed)


def test_recipe_table_matches_entrypoints():
    """Every public entry point must have a recipe with an explicit format pair."""
    assert set(ENTRYPOINTS) == set(EXPERT_RECIPES)


@pytest.mark.parametrize("recipe", ENTRYPOINTS)
def test_packed_weight_cache_is_bit_identical(recipe):
    """A caller-supplied packed cache must not change any output or gradient."""
    operation = _OPERATIONS[recipe]
    torch.manual_seed(31)
    counts = (5, 0, 9, 13, 2, 1)
    e = len(counts)
    x = torch.randn(
        (sum(counts), 64), dtype=torch.bfloat16, device="cuda"
    ).requires_grad_()
    w1 = (
        torch.randn((e, 64, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn((e, 32, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    ).requires_grad_()
    scores = torch.rand((sum(counts),), device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)
    plain = operation(x, scores, counts, w1, w2)
    packed = pack_pre_routed_weights(w1, w2)
    cached = operation(x, scores, counts, w1, w2, packed)
    assert torch.equal(plain, cached)
    plain_grad = torch.autograd.grad(
        (plain * dy).sum(), (x, w1, w2), retain_graph=False
    )
    cached_grad = torch.autograd.grad(
        (cached * dy).sum(), (x, w1, w2), retain_graph=False
    )
    for a, b in zip(plain_grad, cached_grad):
        assert torch.equal(a, b)
    # Reusing the same packed cache for a second call must also be bit-identical.
    again = operation(x, scores, counts, w1, w2, packed)
    assert torch.equal(plain, again)
