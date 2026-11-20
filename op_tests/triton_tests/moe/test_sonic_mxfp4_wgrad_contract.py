# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native main_grad must not silently clear an unrelated PyTorch gradient."""

import pytest
import torch

from aiter.ops.triton.moe.sonic_mxfp4 import (
    _main_grad_accumulate,
    _resolve_dual_quant_mode,
    _validate_main_grads,
    moe_pre_routed_a4w4,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_native_wgrad_defaults_to_fused_dual_quant(monkeypatch):
    monkeypatch.delenv("SONIC_MOE_MXFP4_DUAL_QUANT", raising=False)
    assert _resolve_dual_quant_mode("mxfp4") == "kernel"
    assert _resolve_dual_quant_mode("bf16") == "0"
    monkeypatch.setenv("SONIC_MOE_MXFP4_DUAL_QUANT", "0")
    assert _resolve_dual_quant_mode("mxfp4") == "0"


def test_main_grad_first_write_then_accumulate():
    weights = (
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device="cuda"),
        torch.empty((2, 32, 64), dtype=torch.bfloat16, device="cuda"),
    )
    main_grads = tuple(
        torch.zeros_like(weight, dtype=torch.float32) for weight in weights
    )
    for weight in weights:
        weight.grad_added_to_main_grad = False
    assert not _main_grad_accumulate(weights, main_grads)
    for weight in weights:
        weight.grad_added_to_main_grad = True
    assert _main_grad_accumulate(weights, main_grads)


def test_main_grad_first_write_state_must_match():
    weights = (
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device="cuda"),
        torch.empty((2, 32, 64), dtype=torch.bfloat16, device="cuda"),
    )
    main_grads = tuple(
        torch.zeros_like(weight, dtype=torch.float32) for weight in weights
    )
    weights[0].grad_added_to_main_grad = False
    weights[1].grad_added_to_main_grad = True
    with pytest.raises(ValueError, match="first-write state"):
        _main_grad_accumulate(weights, main_grads)


def test_main_grad_without_megatron_bookkeeping_remains_additive():
    weights = (
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device="cuda"),
        torch.empty((2, 32, 64), dtype=torch.bfloat16, device="cuda"),
    )
    main_grads = tuple(
        torch.zeros_like(weight, dtype=torch.float32) for weight in weights
    )
    assert _main_grad_accumulate(weights, main_grads)


@pytest.mark.parametrize("existing", [0, 1])
def test_sonic_mxfp4_rejects_existing_grad(existing):
    x = torch.randn((33, 64), dtype=torch.bfloat16, device="cuda")
    scores = torch.ones((33,), dtype=torch.float32, device="cuda")
    weights = (
        torch.randn(
            (2, 64, 64), dtype=torch.bfloat16, device="cuda", requires_grad=True
        ),
        torch.randn(
            (2, 32, 64), dtype=torch.bfloat16, device="cuda", requires_grad=True
        ),
    )
    weights[existing].grad = torch.ones_like(weights[existing])
    previous = weights[existing].grad.clone()
    main_grads = tuple(torch.full_like(w, 3.0, dtype=torch.float32) for w in weights)
    with pytest.raises(ValueError, match="weight.grad=None"):
        moe_pre_routed_a4w4(
            x, scores, (1, 32), *weights, wgrad_mode="mxfp4", main_grads=main_grads
        )
    assert torch.equal(weights[existing].grad, previous)
    assert all(torch.all(g == 3.0) for g in main_grads)


def test_main_grads_accept_disjoint_views_of_one_distributed_optimizer_buffer():
    weights = (
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device="cuda"),
        torch.empty((2, 32, 64), dtype=torch.bfloat16, device="cuda"),
    )
    sizes = [weight.numel() for weight in weights]
    flat = torch.zeros(sum(sizes), dtype=torch.float32, device="cuda")
    main_grads = (
        flat[: sizes[0]].view_as(weights[0]),
        flat[sizes[0] :].view_as(weights[1]),
    )
    assert torch._C._overlaps(*main_grads), "regression requires shared storage"
    _validate_main_grads(main_grads, weights)


def test_main_grads_reject_actual_byte_range_overlap():
    weights = (
        torch.empty((2, 64, 64), dtype=torch.bfloat16, device="cuda"),
        torch.empty((2, 32, 64), dtype=torch.bfloat16, device="cuda"),
    )
    sizes = [weight.numel() for weight in weights]
    overlap = 128
    flat = torch.zeros(sum(sizes), dtype=torch.float32, device="cuda")
    main_grads = (
        flat[: sizes[0]].view_as(weights[0]),
        flat[sizes[0] - overlap : sum(sizes) - overlap].view_as(weights[1]),
    )
    with pytest.raises(ValueError, match="overlap each other"):
        _validate_main_grads(main_grads, weights)
