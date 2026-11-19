# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023
"""Per-layer forward+backward cost of each pre-routed expert recipe vs production BF16.

This is the number that predicts an end-to-end step delta: the model has 48 MoE
layers and 2 microbatches per step, so `delta * 96` is the step-time change of
swapping the expert path, holding everything else fixed.

The deployed configuration supplies the packed-weight cache, so the *cached*
numbers are the ones that project to end-to-end; the uncached numbers are kept to
show what the cache is worth.
"""

import json
import time

import torch
from aiter.ops.triton.sonicmoe import SonicMoEActivationType, moe_pre_routed_inputs

from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
    _resolve_cfg,
    block_shape,
    family_for,
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

E = 32
H = 2048
I = 768
ROWS = 32768  # 4 sequences x 4096 tokens x top-8 / 4 EP ranks
LAYERS = 48
MICROBATCHES = 2
BASELINE_STEP_MS = 1558.9

_OPERATIONS = {
    "w4a16": moe_pre_routed_w4a16,
    "a4w4": moe_pre_routed_a4w4,
    "a8w4": moe_pre_routed_a8w4,
    "a8w4f": moe_pre_routed_a8w4f,
}


def realistic_counts(rows=ROWS, experts=E, seed=20261112):
    """Ragged per-expert token counts that sum to `rows` with a plausible skew."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.rand(experts, generator=generator) ** 1.6 + 0.05
    raw = (weights / weights.sum() * rows).round().to(torch.int64)
    raw[-1] += rows - int(raw.sum())
    assert int(raw.sum()) == rows and int(raw.min()) > 0
    return tuple(int(v) for v in raw)


def wall_ms(fn, warmup=8, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def cpu_ms(fn, rep=200):
    fn()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - start) * 1e3 / rep


def make_inputs():
    torch.manual_seed(1234)
    x = torch.randn((ROWS, H), dtype=torch.bfloat16, device="cuda").requires_grad_()
    scores = torch.rand((ROWS,), dtype=torch.float32, device="cuda")
    w1 = (
        torch.randn((E, H, 2 * I), dtype=torch.bfloat16, device="cuda") * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn((E, I, H), dtype=torch.bfloat16, device="cuda") * 0.02
    ).requires_grad_()
    dy = torch.randn((ROWS, H), dtype=torch.bfloat16, device="cuda")
    return x, scores, w1, w2, dy


def fwd_bwd(operation, x, scores, counts, w1, w2, dy, packed=None):
    out = operation(x, scores, counts, w1, w2, packed)
    torch.autograd.grad((out * dy).sum(), (x, w1, w2), retain_graph=False)
    return out


def main():
    counts = realistic_counts()
    assert len(counts) == E and sum(counts) == ROWS
    x, scores, w1, w2, dy = make_inputs()
    counts_gpu = torch.tensor(counts, dtype=torch.int32)
    stream = torch.cuda.current_stream().cuda_stream

    records = {
        "E": E,
        "H": H,
        "I": I,
        "rows": ROWS,
        "counts": counts,
        "layers": LAYERS,
        "microbatches": MICROBATCHES,
        "baseline_step_ms": BASELINE_STEP_MS,
        "recipes": {},
    }

    # Config resolution cost: the bucketed key must make this a pure cache hit.
    families = {}
    for name in _OPERATIONS:
        forward_format = EXPERT_RECIPES[name][0]
        families[name] = family_for(forward_format)
    records["config_lookup"] = {
        "cached_cpu_us": cpu_ms(
            lambda: _resolve_cfg("GEMM-SONIC-A8W4", 2048, 1536, 2048, E)
        )
        * 1e3,
        "block_shape_cached_cpu_us": cpu_ms(
            lambda: block_shape("GEMM-SONIC-A8W4", 900, 1536, 2048, E)
        )
        * 1e3,
    }
    for name, family in families.items():
        shapes = ((2 * I, H), (H, I), (I, H), (H, 2 * I))
        blocks = {block_shape(family, 900, n, k, E) for n, k in shapes}
        records.setdefault("block_shapes", {})[name] = {
            "family": family,
            "shapes": [f"N{n}_K{k}" for n, k in shapes],
            "unique_block_shapes": sorted(blocks),
        }

    def production():
        return moe_pre_routed_inputs(
            x,
            scores,
            counts_gpu,
            w1,
            None,
            w2,
            None,
            stream,
            SonicMoEActivationType.SWIGLU,
            False,
            True,
        )[0]

    records["recipes"]["production_bf16"] = {
        "fwd_ms": wall_ms(lambda: production()),
        "fwd_bwd_ms": wall_ms(
            lambda: torch.autograd.grad((production() * dy).sum(), (x, w1, w2))
        ),
    }

    for name, operation in _OPERATIONS.items():
        entry = {
            "recipe": EXPERT_RECIPES[name],
            "fwd_ms": wall_ms(lambda: operation(x, scores, counts, w1, w2)),
            "fwd_bwd_ms": wall_ms(
                lambda: fwd_bwd(operation, x, scores, counts, w1, w2, dy)
            ),
        }
        packed = pack_pre_routed_weights(w1, w2)
        entry["pack_only_ms"] = wall_ms(
            lambda: pack_pre_routed_weights(w1, w2), warmup=3, rep=10
        )
        entry["fwd_cached_ms"] = wall_ms(
            lambda: operation(x, scores, counts, w1, w2, packed)
        )
        entry["fwd_bwd_cached_ms"] = wall_ms(
            lambda: fwd_bwd(operation, x, scores, counts, w1, w2, dy, packed)
        )
        records["recipes"][name] = entry

    production_cached = records["recipes"]["production_bf16"]["fwd_bwd_ms"]
    for name, entry in records["recipes"].items():
        entry["delta_per_layer_cached_ms"] = (
            entry.get("fwd_bwd_cached_ms", entry["fwd_bwd_ms"]) - production_cached
        )
        entry["delta_per_layer_uncached_ms"] = entry["fwd_bwd_ms"] - production_cached
    print(json.dumps(records, indent=1), flush=True)

    print(
        "\nprojected step time, DEPLOYED config (cached packs), "
        f"baseline {BASELINE_STEP_MS:.1f} ms:",
        flush=True,
    )
    for name, entry in records["recipes"].items():
        delta = entry["delta_per_layer_cached_ms"] * LAYERS * MICROBATCHES
        print(
            f"  {name:16s} per-layer {entry.get('fwd_bwd_cached_ms', 0):7.3f} ms"
            f"  step {BASELINE_STEP_MS + delta:7.1f} ms  ({BASELINE_STEP_MS / (BASELINE_STEP_MS + delta):.3f}x)"
        )
    print("\nuncached (to show what the packed-weight cache is worth):", flush=True)
    for name, entry in records["recipes"].items():
        delta = entry["delta_per_layer_uncached_ms"] * LAYERS * MICROBATCHES
        print(
            f"  {name:16s} per-layer {entry['fwd_bwd_ms']:7.3f} ms  step delta {delta:+8.1f} ms"
        )


if __name__ == "__main__":
    main()
