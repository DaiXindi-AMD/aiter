# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerics and gradient checks for the pre-routed MXFP4 expert recipes."""

import pytest
import torch
from test_sonic_mxfp4 import decode

from aiter.ops.triton.gemm.grouped import sonic_mxfp4 as sonic_gemm
from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
    pack_grouped_mxfp4_weights,
    prepare_expert_offsets,
    prepare_grouped_gemm_metadata,
    prepare_grouped_gemm_metadata_host,
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


_REAL_ROUTE_COUNTS = (
    1348, 860, 999, 1310, 870, 30, 881, 789, 598, 245, 0, 959, 2158, 471, 804, 0,
    783, 936, 991, 100, 4, 1301, 716, 1612, 921, 456, 870, 782, 2958, 1661, 780, 1691,
)  # fmt: skip


@pytest.mark.parametrize(
    "counts",
    [(3, 0, 7, 5), (0, 0, 1), (129, 128, 127, 0, 256), _REAL_ROUTE_COUNTS],
)
@pytest.mark.parametrize("block_m,block_n", [(128, 128), (64, 128), (32, 256)])
def test_host_metadata_matches_device_metadata(counts, block_m, block_n):
    ns = (1536, 2048, 768, 100)
    device = torch.device("cuda")
    expected = prepare_grouped_gemm_metadata(counts, device, block_m, block_n)
    actual = prepare_grouped_gemm_metadata_host(counts, device, block_m, block_n, ns)
    for key in ("offsets", "counts_dev"):
        assert actual[key].dtype == torch.int32 and actual[key].is_contiguous()
        torch.testing.assert_close(actual[key], expected[key].int(), rtol=0, atol=0)
    for n in ns:
        expected_base = sonic_gemm._expert_block_base(expected, n)
        actual_base = actual[("block_base", n)]
        assert actual_base.dtype == torch.int32 and actual_base.is_contiguous()
        torch.testing.assert_close(actual_base, expected_base, rtol=0, atol=0)
    # A shape outside ``ns`` still falls back to the lazy device build.
    torch.testing.assert_close(
        sonic_gemm._expert_block_base(actual, 384),
        sonic_gemm._expert_block_base(expected, 384),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("counts", [(), (3, -1), (2**31, 0)])
def test_host_metadata_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        prepare_grouped_gemm_metadata_host(
            counts, torch.device("cuda"), 128, 128, (64,)
        )


@pytest.mark.parametrize("count_kind", ["tuple", "pinned"])
def test_host_metadata_preserves_forward_and_backward(monkeypatch, count_kind):
    counts = (3, 0, 7, 5)
    torch.manual_seed(177)
    inputs = torch.randn((15, 64), dtype=torch.bfloat16, device="cuda")
    scores = torch.linspace(0, 1, 15, dtype=torch.float32, device="cuda")
    w1 = torch.randn((4, 64, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn((4, 32, 64), dtype=torch.bfloat16, device="cuda") * 0.02
    dout = torch.randn_like(inputs)

    def run(host_metadata):
        monkeypatch.setenv("AITER_SONIC_HOST_METADATA", host_metadata)
        args = tuple(
            value.detach().clone().requires_grad_() for value in (inputs, w1, w2)
        )
        routed_scores = scores.detach().clone().requires_grad_()
        count_arg = (
            counts
            if count_kind == "tuple"
            else torch.tensor(counts, dtype=torch.int64, pin_memory=True)
        )
        output = moe_pre_routed_a4w4(
            args[0], routed_scores, count_arg, args[1], args[2]
        )
        (output * dout).sum().backward()
        return (output, args[0].grad, routed_scores.grad, args[1].grad, args[2].grad)

    baseline = run("0")
    candidate = run("1")
    for actual, expected in zip(candidate, baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def test_bf16_wgrad_writes_and_accumulates_fp32_main_grad():
    counts = (3, 0, 7, 5)
    torch.manual_seed(177)
    weights = (
        torch.randn((4, 64, 64), dtype=torch.bfloat16, device="cuda") * 0.02,
        torch.randn((4, 32, 64), dtype=torch.bfloat16, device="cuda") * 0.02,
    )
    direct_weights = tuple(w.detach().clone().requires_grad_() for w in weights)
    reference_weights = tuple(w.detach().clone().requires_grad_() for w in weights)
    main_grads = tuple(
        torch.full_like(w, float("nan"), dtype=torch.float32) for w in direct_weights
    )
    for weight in direct_weights:
        weight.grad_added_to_main_grad = False

    expected = [torch.zeros_like(g) for g in main_grads]
    for seed in (178, 179):
        torch.manual_seed(seed)
        x = torch.randn((15, 64), dtype=torch.bfloat16, device="cuda")
        scores = torch.rand((15,), dtype=torch.float32, device="cuda")
        dout = torch.randn_like(x)
        reference = moe_pre_routed_a4w4(
            x.detach().clone().requires_grad_(),
            scores.detach().clone().requires_grad_(),
            counts,
            *reference_weights,
        )
        (reference * dout).sum().backward()
        for total, weight in zip(expected, reference_weights):
            total.add_(weight.grad.float())
            weight.grad = None

        direct = moe_pre_routed_a4w4(
            x.detach().clone().requires_grad_(),
            scores.detach().clone().requires_grad_(),
            counts,
            *direct_weights,
            wgrad_mode="bf16",
            main_grads=main_grads,
        )
        (direct * dout).sum().backward()
        assert all(w.grad is None and w.grad_added_to_main_grad for w in direct_weights)

    for actual, reference in zip(main_grads, expected):
        # Direct FP32 output avoids the control path's intermediate BF16 rounding.
        error = (actual - reference).float()
        relative = error.norm() / reference.float().norm().clamp_min(1e-20)
        snr_db = 20 * torch.log10(
            reference.float().norm().clamp_min(1e-20) / error.norm().clamp_min(1e-20)
        )
        assert relative < 0.0018
        assert snr_db > 55
        assert error.abs().max() < 4.5e-4
        assert torch.count_nonzero(actual[1]) == 0


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
