# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused OPUS chain must reproduce the Triton native-Wgrad recipe."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.sonic_mxfp4 import (
    PRE_ROUTED_MXFP4_WGRAD_BACKENDS,
    moe_pre_routed_a4w4,
    pack_pre_routed_weights,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="gfx950 required",
)

COUNTS = (
    (300, 0, 129, 75),
    (1, 255, 256, 257),
    (0, 0, 0, 0),
)


def _inputs(counts, h, i, seed=0):
    torch.manual_seed(seed)
    rows, e = sum(counts), len(counts)
    x = torch.randn((rows, h), dtype=torch.bfloat16, device="cuda")
    scores = torch.rand((rows,), dtype=torch.float32, device="cuda")
    w1 = torch.randn((e, h, 2 * i), dtype=torch.bfloat16, device="cuda") * 0.05
    w2 = torch.randn((e, i, h), dtype=torch.bfloat16, device="cuda") * 0.05
    dout = torch.randn((rows, h), dtype=torch.bfloat16, device="cuda")
    return x, scores, w1, w2, dout


def _run(backend, x, scores, counts, w1, w2, dout, packed, main_grads=None):
    x = x.detach().clone().requires_grad_()
    scores = scores.detach().clone().requires_grad_()
    w1 = w1.detach().clone().requires_grad_()
    w2 = w2.detach().clone().requires_grad_()
    out = moe_pre_routed_a4w4(
        x,
        scores,
        counts,
        w1,
        w2,
        packed,
        wgrad_mode="mxfp4",
        main_grads=main_grads,
        wgrad_backend=backend,
    )
    out.backward(dout)
    return out, x.grad, scores.grad, w1.grad, w2.grad


@pytest.mark.parametrize("counts", COUNTS)
@pytest.mark.parametrize("h,i", [(256, 128), (512, 256)])
def test_opus_matches_triton_native_wgrad(monkeypatch, counts, h, i):
    monkeypatch.setenv("SONIC_MOE_MXFP4_DUAL_QUANT", "kernel")
    x, scores, w1, w2, dout = _inputs(counts, h, i)
    packed = pack_pre_routed_weights(w1, w2)
    ref = _run("triton", x, scores, counts, w1, w2, dout, packed)
    got = _run("opus", x, scores, counts, w1, w2, dout, packed)
    for name, r, g in zip(("out", "dx", "ds"), ref[:3], got[:3]):
        assert torch.equal(r, g), name
    for r, g in zip(ref[3:], got[3:]):
        torch.testing.assert_close(g.float(), r.float(), rtol=1e-2, atol=1e-2)


def test_opus_main_grads_overwrite_then_accumulate(monkeypatch):
    monkeypatch.setenv("SONIC_MOE_MXFP4_DUAL_QUANT", "kernel")
    counts = (300, 0, 129, 75)
    x, scores, w1, w2, dout = _inputs(counts, 256, 128)
    packed = pack_pre_routed_weights(w1, w2)
    _, _, _, dw1, dw2 = _run("opus", x, scores, counts, w1, w2, dout, None)
    main_grads = tuple(
        torch.full(w.shape, 7.0, dtype=torch.float32, device="cuda") for w in (w1, w2)
    )
    weights = [w.detach().clone().requires_grad_() for w in (w1, w2)]
    for weight in weights:
        weight.grad_added_to_main_grad = False
    for _ in range(2):
        out = moe_pre_routed_a4w4(
            x,
            scores,
            counts,
            *weights,
            packed,
            wgrad_mode="mxfp4",
            main_grads=main_grads,
            wgrad_backend="opus",
        )
        out.backward(dout)
        assert all(w.grad is None and w.grad_added_to_main_grad for w in weights)
    for grad, ref in zip(main_grads, (dw1, dw2)):
        torch.testing.assert_close(grad, 2 * ref.float(), rtol=1e-2, atol=1e-2)


def test_opus_backend_contract():
    assert "opus" in PRE_ROUTED_MXFP4_WGRAD_BACKENDS
    x, scores, w1, w2, _ = _inputs((4, 4), 256, 128)
    with pytest.raises(ValueError, match="wgrad_mode=mxfp4"):
        moe_pre_routed_a4w4(x, scores, (4, 4), w1, w2, wgrad_backend="opus")
    x, scores, w1, w2, _ = _inputs((4, 4), 192, 128)
    with pytest.raises(NotImplementedError, match="multiples of 128"):
        moe_pre_routed_a4w4(
            x, scores, (4, 4), w1, w2, wgrad_mode="mxfp4", wgrad_backend="opus"
        )
