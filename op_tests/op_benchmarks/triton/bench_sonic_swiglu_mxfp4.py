#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark activation plus grouped dual quant against the fused producer."""

import argparse
import json
import statistics

import torch

from aiter import logger
from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.moe.sonic_swiglu_mxfp4 import fused_swiglu_mxfp4
from aiter.ops.triton.moe.sonicmoe import activation_fwd
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_dual_v2,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

_COUNTS = (1536, 1280, 1152, 1024, 960, 896, 768, 640) * 4
_FEATURES = 768


def _time(operation):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--peak-gbps", type=float)
    args = parser.parse_args()
    if get_arch() != "gfx950":
        raise RuntimeError("the fused producer benchmark requires gfx950")

    rows = sum(_COUNTS)
    h1 = torch.randn((rows, 2 * _FEATURES), dtype=torch.bfloat16, device="cuda")
    metadata = prepare_mxfp4_wgrad_metadata(
        _COUNTS, h1.device, pad_multiple=256, backend="gluon"
    )

    def separate():
        activation = activation_fwd(h1, _FEATURES, "swiglu", concat_layout=True)
        row, col = quantize_grouped_mxfp4_dual_v2(activation, metadata)
        return activation, row, col

    def fused():
        return fused_swiglu_mxfp4(h1, metadata)

    providers = (("activation_plus_dual", separate), ("fused", fused))
    for _ in range(5):
        for _name, operation in providers:
            operation()
    torch.cuda.synchronize()

    samples = {name: [] for name, _operation in providers}
    last = {}
    for index in range(args.repeats):
        order = providers if index % 2 == 0 else providers[::-1]
        for name, operation in order:
            elapsed, last[name] = _time(operation)
            samples[name].append(elapsed)

    expected = last["activation_plus_dual"]
    actual = last["fused"]
    byte_exact = all(
        torch.equal(result, reference)
        for result, reference in zip(
            (actual[0], *actual[1], *actual[2]),
            (expected[0], *expected[1], *expected[2]),
        )
    )
    separate_ms = statistics.median(samples["activation_plus_dual"])
    fused_ms = statistics.median(samples["fused"])

    h1_bytes = h1.numel() * h1.element_size()
    activation_bytes = rows * _FEATURES * h1.element_size()
    output_bytes = (
        activation_bytes
        + rows * _FEATURES // 2
        + rows * _FEATURES // 32
        + metadata.padded_rows * _FEATURES // 2
        + metadata.padded_rows * _FEATURES // 32
    )
    logical_bytes = {
        "activation_plus_dual": h1_bytes + activation_bytes + output_bytes,
        "fused": h1_bytes + output_bytes,
    }
    result = {
        "arch": get_arch(),
        "experts": len(_COUNTS),
        "rows": rows,
        "features": _FEATURES,
        "ragged_counts": _COUNTS,
        "repeats": args.repeats,
        "activation_plus_dual_median_ms": separate_ms,
        "fused_median_ms": fused_ms,
        "speedup": separate_ms / fused_ms,
        "byte_exact": byte_exact,
        "launches": {"activation_plus_dual": 2, "fused": 1},
        "logical_input_read_bytes": {
            "activation_plus_dual": h1_bytes + activation_bytes,
            "fused": h1_bytes,
        },
        "logical_input_read_reduction": activation_bytes
        / (h1_bytes + activation_bytes),
        "logical_total_bytes": logical_bytes,
        "logical_gbps": {
            "activation_plus_dual": logical_bytes["activation_plus_dual"]
            / separate_ms
            / 1e6,
            "fused": logical_bytes["fused"] / fused_ms / 1e6,
        },
        "samples_ms": samples,
    }
    if args.peak_gbps is not None:
        result["peak_gbps"] = args.peak_gbps
        result["logical_bandwidth_utilization_pct"] = {
            name: 100.0 * bandwidth / args.peak_gbps
            for name, bandwidth in result["logical_gbps"].items()
        }
    logger.info("%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
