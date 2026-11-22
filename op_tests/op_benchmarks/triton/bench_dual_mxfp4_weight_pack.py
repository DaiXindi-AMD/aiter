#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark the four-call pre-routed weight pack against the one-read dual pack."""

import argparse
import json
import statistics

import torch

from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import pack_pre_routed_weights_dual
from aiter.ops.triton.moe.sonic_mxfp4 import pack_pre_routed_weights


def _time(operation, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        result = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    e, h, i = args.experts, args.hidden, args.intermediate
    torch.manual_seed(20261122)
    w1 = torch.randn((e, h, 2 * i), dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn((e, i, h), dtype=torch.bfloat16, device="cuda") * 0.02
    providers = {
        "four_call": lambda: pack_pre_routed_weights(w1, w2),
        "dual": lambda: pack_pre_routed_weights_dual(w1, w2),
    }
    for operation in providers.values():
        operation()
    torch.cuda.synchronize()
    samples = {name: [] for name in providers}
    last = {}
    for repeat in range(args.repeats):
        names = list(providers) if repeat % 2 == 0 else list(reversed(providers))
        for name in names:
            elapsed, last[name] = _time(providers[name], args.iterations)
            samples[name].append(elapsed)
    exact = all(
        torch.equal(a, b)
        for pa, pb in zip(last["dual"], last["four_call"])
        for a, b in zip(pa, pb)
    )
    weight_bytes = (w1.numel() + w2.numel()) * 2
    # Minimum traffic: read BF16 once, write two FP4 payloads and two scale sets.
    min_bytes = weight_bytes + 2 * (weight_bytes // 4 + weight_bytes // 64)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    print(
        json.dumps(
            {
                "shape": {"E": e, "H": h, "I": i},
                "byte_exact": exact,
                "median_ms": medians,
                "speedup": medians["four_call"] / medians["dual"],
                "dual_effective_tbps": min_bytes / (medians["dual"] * 1e-3) / 1e12,
                "samples_ms": samples,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
