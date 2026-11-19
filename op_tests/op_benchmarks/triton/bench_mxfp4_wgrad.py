# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wgrad stage costs and quantized-reference errors; training timing is separate."""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import triton

from aiter import logger
from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    grouped_mxfp4_wgrad,
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import quantize_grouped_mxfp4_colwise
from aiter.ops.triton.utils.mxfp4_wgrad import get_mxfp4_wgrad_stats
from op_tests.triton_tests.utils.mxfp4_wgrad_reference import (
    bf16_wgrad_reference,
    error_metrics,
    same_fp4_wgrad_reference,
)


def _measure(fn, warmup, iterations, repeats, capture=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        fn = graph.replay
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "batch_mean_p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "sample_semantics": "event sample is an iterations-call mean; not per-call p95",
        "spread_ms": max(samples) - min(samples),
        "samples_ms": samples,
    }


def _cases(args):
    if args.counts_json:
        value = json.loads(Path(args.counts_json).read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [value]
        samples = value.get("samples", [])
        if samples:
            return [sample["counts"] for sample in samples[: args.max_cases]]
        if "counts" in value:
            return [value["counts"]]
        raise ValueError("counts JSON must contain counts or samples[].counts")
    return [[0, 1, 31, 33, 129], [513, 0, 7, 2048]]


def _bench_case(counts, n, m, args):
    _wgrad = partial(grouped_mxfp4_wgrad, backend=args.backend)
    _prepare = partial(prepare_mxfp4_wgrad_metadata, backend=args.backend)
    x = torch.randn((sum(counts), n), dtype=torch.bfloat16, device=args.device) * 0.05
    dy = torch.randn((sum(counts), m), dtype=torch.bfloat16, device=args.device) * 0.08
    metadata = _prepare(tuple(counts), x.device)
    xp, dyp = quantize_grouped_mxfp4_colwise(
        x, metadata
    ), quantize_grouped_mxfp4_colwise(dy, metadata)
    out = torch.zeros((len(counts), n, m), dtype=torch.float32, device=x.device)

    def launch():
        return _wgrad(
            x,
            dy,
            tuple(counts),
            out=out,
            metadata=metadata,
            quantized_operands=(xp, dyp),
            transpose_output=True,
        )

    launch()
    same_fp4 = same_fp4_wgrad_reference(xp, dyp, metadata, transpose_output=True)
    bf16 = bf16_wgrad_reference(x, dy, counts, transpose_output=True)
    torch.testing.assert_close(out, same_fp4, rtol=2e-4, atol=2e-5)
    accuracy = {
        "same_fp4": error_metrics(same_fp4, out),
        "bf16": error_metrics(bf16, out),
    }

    def prep():
        return (
            quantize_grouped_mxfp4_colwise(x, metadata),
            quantize_grouped_mxfp4_colwise(dy, metadata),
        )

    def full():
        return _wgrad(x, dy, tuple(counts), out=out, transpose_output=True)

    def prepared_full():
        return _wgrad(
            x, dy, tuple(counts), out=out, metadata=metadata, transpose_output=True
        )

    settings = (args.warmup, args.iterations, args.repeats)
    timings = {
        "consumer_public_no_quant": _measure(launch, *settings),
        "consumer_cuda_graph_single_kernel": _measure(launch, *settings, capture=True),
        "two_operand_quant_public": _measure(prep, *settings),
        "quant_plus_gemm_reused_metadata": _measure(prepared_full, *settings),
        "metadata_plus_quant_plus_gemm": _measure(full, *settings),
    }
    bytes_min = (m + n) * metadata.padded_rows * (0.5 + 1 / 32) + 4 * len(
        counts
    ) * m * n
    flops = 2 * sum(counts) * m * n
    consumer_ms = timings["consumer_cuda_graph_single_kernel"]["median_ms"]
    return {
        "counts": counts,
        "M_dy_features": m,
        "N_x_features": n,
        "real_rows": metadata.total_rows,
        "padded_rows": metadata.padded_rows,
        "padding_ratio": metadata.padded_rows / max(metadata.total_rows, 1),
        "accuracy": accuracy,
        "timings": timings,
        "logical_flops": flops,
        "minimum_consumer_bytes_estimate": bytes_min,
        "arithmetic_intensity_estimate": flops / bytes_min,
        "achieved_tflops": flops / consumer_ms / 1e9,
        "minimum_bytes_effective_tbps_estimate": bytes_min / consumer_ms / 1e9,
        "stats": get_mxfp4_wgrad_stats(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-json")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--shapes", default="2048x1536,768x2048")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("triton", "gluon"), default="triton")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(11)
    shapes = [tuple(map(int, shape.split("x"))) for shape in args.shapes.split(",")]
    report = {
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(args.device),
        },
        "boundary": "isolated Wgrad; not R75 matched BF16 or complete expert/training step",
        "method": "warm event medians/spread; captured single-consumer graph avoids host gaps",
        "roofline": "minimum-byte estimates only; empirical roof and PMC remain required",
        "settings": vars(args),
        "results": [],
    }
    for counts in _cases(args):
        for n, m in shapes:
            row = _bench_case(counts, n, m, args)
            report["results"].append(row)
            logger.info(
                "Wgrad N=%d M=%d rows=%d consumer=%s",
                n,
                m,
                sum(counts),
                row["timings"]["consumer_public_no_quant"],
            )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
