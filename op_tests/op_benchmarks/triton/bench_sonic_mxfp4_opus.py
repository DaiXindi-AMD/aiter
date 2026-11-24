# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023
"""Per-layer A4W4 forward+backward with native MXFP4 Wgrad: Triton chain vs fused OPUS chain.

Uses the Qwen3-30B-A3B EP4 layer of bench_sonic_mxfp4_autograd with main_grads, the
way Megatron calls it. ``host_ms`` is the CPU enqueue time per layer.
"""

import os
import time

import torch
from bench_sonic_mxfp4_autograd import make_inputs, realistic_counts, wall_ms

from aiter.ops.triton.moe.sonic_mxfp4 import (
    moe_pre_routed_a4w4,
    pack_pre_routed_weights,
)


def main():
    os.environ.setdefault("SONIC_MOE_MXFP4_DUAL_QUANT", "kernel")
    counts = realistic_counts()
    x, scores, w1, w2, dy = make_inputs()
    w1, w2 = w1.detach(), w2.detach()
    packed = pack_pre_routed_weights(w1, w2)
    main_grads = tuple(torch.zeros_like(w, dtype=torch.float32) for w in (w1, w2))
    weights = [w.clone().requires_grad_() for w in (w1, w2)]

    def step(backend):
        out = moe_pre_routed_a4w4(
            x,
            scores,
            counts,
            *weights,
            packed,
            wgrad_mode="mxfp4",
            main_grads=main_grads,
            wgrad_backend=backend,
        )
        torch.autograd.grad(out, x, dy)

    for backend in ("triton", "opus"):
        gpu = wall_ms(lambda: step(backend))
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            step(backend)
        host = (time.perf_counter() - start) * 1e3 / 50
        torch.cuda.synchronize()
        print(f"{backend:7s} fwd+bwd {gpu:7.3f} ms  host_ms {host:7.3f}", flush=True)


if __name__ == "__main__":
    main()
