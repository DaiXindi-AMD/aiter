#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark staged SwiGLU plus dual MXFP4 quant against the fused producer."""

import argparse
import json
import statistics

import torch

from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.moe.sonicmoe import activation_fwd
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_dual_v2,
    swiglu_grouped_mxfp4_dual,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

EXPERTS = 32
HIDDEN = 2048
INTERMEDIATE = 768
DEFAULT_ROWS = 32768


def _time(operation):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result


def _balanced_counts(rows):
    base, extra = divmod(rows, EXPERTS)
    return tuple(base + (expert < extra) for expert in range(EXPERTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if get_arch() != "gfx950":
        raise RuntimeError("fused SwiGLU grouped MXFP4 dual requires gfx950")

    counts = _balanced_counts(args.rows)
    h = torch.randn((args.rows, 2 * INTERMEDIATE), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        counts, h.device, pad_multiple=256, backend="gluon"
    )

    def staged():
        output = activation_fwd(h, INTERMEDIATE, "swiglu", concat_layout=True)
        row, col = quantize_grouped_mxfp4_dual_v2(output, metadata)
        return output, row, col

    def fused():
        return swiglu_grouped_mxfp4_dual(h, metadata)

    providers = (("staged", staged), ("fused", fused))
    for _ in range(5):
        for _, operation in providers:
            operation()
    torch.cuda.synchronize()

    samples = {name: [] for name, _ in providers}
    last = {}
    for index in range(args.repeats):
        order = providers[index % 2 :] + providers[: index % 2]
        for name, operation in order:
            elapsed, last[name] = _time(operation)
            samples[name].append(elapsed)

    staged_output, staged_row, staged_col = last["staged"]
    fused_output, fused_row, fused_col = last["fused"]
    exact = all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (fused_output, *fused_row, *fused_col),
            (staged_output, *staged_row, *staged_col),
        )
    )
    staged_ms = statistics.median(samples["staged"])
    fused_ms = statistics.median(samples["fused"])

    rows, features, padded = args.rows, INTERMEDIATE, metadata.padded_rows
    activation_input = rows * 2 * features * 2
    activation_output = rows * features * 2
    quant_outputs = (
        rows * features // 2
        + rows * features // 32
        + features * padded // 2
        + features * padded // 32
    )
    fused_bytes = activation_input + activation_output + quant_outputs
    staged_bytes = fused_bytes + activation_output
    result = {
        "gpu": torch.cuda.get_device_name(),
        "arch": get_arch(),
        "shape": {
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "rows": rows,
            "padded_rows": padded,
        },
        "staged_median_ms": staged_ms,
        "fused_median_ms": fused_ms,
        "speedup": staged_ms / fused_ms,
        "byte_exact": exact,
        "expected_launches": {"staged": 2, "fused": 1},
        "logical_hbm_bytes": {"staged": staged_bytes, "fused": fused_bytes},
        "logical_byte_reduction": 1.0 - fused_bytes / staged_bytes,
        "samples_ms": samples,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
