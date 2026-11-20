#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark independent row+column quant versus the grouped dual producer."""

import argparse
import json
import os
import statistics

import torch

from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import (
    quantize_grouped_mxfp4_colwise,
)
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_concurrent,
    quantize_grouped_mxfp4_dual,
    quantize_grouped_mxfp4_dual_mapped,
    quantize_grouped_mxfp4_dual_v2,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton.arch_info import get_arch


def _time(callable_):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    counts = (1536, 1280, 1152, 1024, 960, 896, 768, 640) * 4
    rows = sum(counts)
    source_rows = (rows + 3) // 4
    source = torch.randn(
        (source_rows, args.features), dtype=torch.bfloat16, device="cuda"
    )
    source_map = (torch.arange(rows, device=source.device) % source_rows).to(
        torch.int32
    )
    source_map_long = source_map.long()
    x = source.index_select(0, source_map_long)
    metadata = prepare_mxfp4_wgrad_metadata(
        counts, x.device, pad_multiple=256, backend="gluon"
    )

    def separate():
        return dynamic_mxfp4_quant(x), quantize_grouped_mxfp4_colwise(x, metadata)

    def dual():
        return quantize_grouped_mxfp4_dual(x, metadata)

    def concurrent():
        row, col, ready = quantize_grouped_mxfp4_concurrent(x, metadata)
        torch.cuda.current_stream().wait_event(ready)
        return row, col

    def explicit_gather_v2():
        gathered = source.index_select(0, source_map_long)
        return quantize_grouped_mxfp4_dual_v2(gathered, metadata)

    def mapped():
        return quantize_grouped_mxfp4_dual_mapped(source, source_map, metadata)

    providers = [
        ("separate", separate),
        ("dual", dual),
        ("concurrent", concurrent),
    ]
    if get_arch() == "gfx950":
        providers.extend(
            [
                ("v2", lambda: quantize_grouped_mxfp4_dual_v2(x, metadata)),
                ("explicit_gather_v2", explicit_gather_v2),
            ]
        )
        if os.environ.get("AITER_GROUPED_MXFP4_MAPPED") == "1":
            providers.append(("mapped", mapped))
    for _ in range(5):
        for _, operation in providers:
            operation()
    torch.cuda.synchronize()

    samples = {f"{name}_ms": [] for name, _ in providers}
    last = {}
    for index in range(args.repeats):
        arms = tuple(providers)
        shift = index % len(arms)
        order = arms[shift:] + arms[:shift]
        for name, operation in order:
            elapsed, last[name] = _time(operation)
            samples[f"{name}_ms"].append(elapsed)

    row_ref, col_ref = last["separate"]
    exact = {}
    for name, _ in providers[1:]:
        row, col = last[name]
        exact[name] = all(
            torch.equal(actual, expected)
            for actual, expected in zip((*row, *col), (*row_ref, *col_ref))
        )
    separate_ms = statistics.median(samples["separate_ms"])
    dual_ms = statistics.median(samples["dual_ms"])
    concurrent_ms = statistics.median(samples["concurrent_ms"])
    result = {
        "rows": rows,
        "features": args.features,
        "repeats": args.repeats,
        "separate_median_ms": separate_ms,
        "dual_median_ms": dual_ms,
        "concurrent_median_ms": concurrent_ms,
        "dual_speedup": separate_ms / dual_ms,
        "concurrent_speedup": separate_ms / concurrent_ms,
        "byte_exact": exact,
        "samples": samples,
    }
    if "v2_ms" in samples:
        v2_ms = statistics.median(samples["v2_ms"])
        result["v2_median_ms"] = v2_ms
        result["v2_speedup"] = separate_ms / v2_ms
        result["v2_vs_dual"] = dual_ms / v2_ms
    if "mapped_ms" in samples:
        gathered_ms = statistics.median(samples["explicit_gather_v2_ms"])
        mapped_ms = statistics.median(samples["mapped_ms"])
        result["explicit_gather_v2_median_ms"] = gathered_ms
        result["mapped_median_ms"] = mapped_ms
        result["mapped_vs_explicit_gather_speedup"] = gathered_ms / mapped_ms
        gather_bytes = rows * args.features * 2
        result["removed_bf16_gather_buffer_bytes"] = gather_bytes
        result["avoided_logical_intermediate_traffic_bytes"] = 2 * gather_bytes
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
