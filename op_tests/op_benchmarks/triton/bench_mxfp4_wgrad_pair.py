# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Matched R75 BF16 and native MXFP4 Wgrad on real two-microbatch windows."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import triton

from aiter.ops.triton.gemm.grouped.bf16_wgrad import grouped_bf16_wgrad
from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    check_mxfp4_wgrad_support,
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


def _event(fn, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _summary(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "batch_mean_p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min_ms": min(values),
        "max_ms": max(values),
        "spread_ms": max(values) - min(values),
        "samples_ms": values,
    }


def _paired(a, b, args):
    for _ in range(args.warmup):
        a()
        b()
    torch.cuda.synchronize()
    samples = {"bf16": [], "mxfp4": []}
    chronological = []
    for repeat in range(args.repeats):
        order = [("bf16", a), ("mxfp4", b), ("mxfp4", b), ("bf16", a)]
        if repeat % 2:
            order = [("mxfp4", b), ("bf16", a), ("bf16", a), ("mxfp4", b)]
        for position, (name, fn) in enumerate(order):
            value = _event(fn, args.iterations)
            samples[name].append(value)
            chronological.append(
                {
                    "repeat": repeat,
                    "position": position,
                    "variant": name,
                    "batch_mean_ms": value,
                    "windows_in_batch": args.iterations,
                }
            )
    result = {name: _summary(value) for name, value in samples.items()}
    result["chronological"] = chronological
    result["sample_semantics"] = (
        "Each event sample is the mean of iterations two-MB windows; p95 is not per-window p95."
    )
    result["bf16_over_mxfp4"] = (
        result["bf16"]["median_ms"] / result["mxfp4"]["median_ms"]
    )
    return result


def _stage(fn, args):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    return _summary([_event(fn, args.iterations) for _ in range(args.repeats)])


def _window(rows, n, m, args):
    config, is_tuned = check_mxfp4_wgrad_support(
        torch.device(args.device), m, n, backend=args.backend, transpose_output=True
    )
    _wgrad = partial(grouped_mxfp4_wgrad, backend=args.backend)
    _prepare = partial(prepare_mxfp4_wgrad_metadata, backend=args.backend)
    if args.config_file:
        expected = json.loads(Path(args.config_file).read_text(encoding="utf-8"))["any"]
        if config != expected:
            raise ValueError("active shared-loader config differs from --config-file")
    prepared = []
    for row in rows:
        counts = tuple(row["counts"])
        x = (
            torch.randn((sum(counts), n), dtype=torch.bfloat16, device=args.device)
            * 0.05
        )
        dy = (
            torch.randn((sum(counts), m), dtype=torch.bfloat16, device=args.device)
            * 0.08
        )
        meta = _prepare(counts, x.device)
        xp, dyp = quantize_grouped_mxfp4_colwise(
            x, meta
        ), quantize_grouped_mxfp4_colwise(dy, meta)
        prepared.append((x, dy, counts, meta, xp, dyp))
    shape = (len(rows[0]["counts"]), n, m)
    native = torch.zeros(shape, dtype=torch.float32, device=args.device)
    baseline = torch.zeros_like(native)

    def bf16():
        baseline.zero_()
        for mb, (x, dy, counts, meta, _, _) in enumerate(prepared):
            grouped_bf16_wgrad(
                x,
                dy,
                counts,
                out=baseline,
                offsets=meta.real_offsets,
                accumulate=bool(mb),
                transpose_output=True,
            )

    def full_native():
        native.zero_()
        for x, dy, counts, _, _, _ in prepared:
            _wgrad(x, dy, counts, out=native, accumulate=True, transpose_output=True)

    def reused_native():
        native.zero_()
        for x, dy, counts, meta, _, _ in prepared:
            _wgrad(
                x,
                dy,
                counts,
                out=native,
                metadata=meta,
                accumulate=True,
                transpose_output=True,
            )

    def consumer():
        native.zero_()
        for x, dy, counts, meta, xp, dyp in prepared:
            _wgrad(
                x,
                dy,
                counts,
                out=native,
                metadata=meta,
                quantized_operands=(xp, dyp),
                accumulate=True,
                transpose_output=True,
            )

    def metadata():
        for x, _, counts, _, _, _ in prepared:
            _prepare(counts, x.device)

    def quant():
        for x, dy, _, meta, _, _ in prepared:
            quantize_grouped_mxfp4_colwise(x, meta)
            quantize_grouped_mxfp4_colwise(dy, meta)

    bf16()
    consumer()
    same = sum(
        same_fp4_wgrad_reference(xp, dyp, meta, transpose_output=True)
        for _, _, _, meta, xp, dyp in prepared
    )
    original = sum(
        bf16_wgrad_reference(x, dy, counts, transpose_output=True)
        for x, dy, counts, _, _, _ in prepared
    )
    torch.testing.assert_close(native, same, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(baseline, original, rtol=2e-4, atol=2e-5)
    errors = {
        "same_fp4": error_metrics(same, native),
        "bf16": error_metrics(original, native),
    }
    stages = {
        "metadata_two_mb": _stage(metadata, args),
        "colwise_quant_four_operands": _stage(quant, args),
        "consumer_two_mb_plus_zero": _stage(consumer, args),
    }
    pairs = {
        "complete_wgrad_metadata_quant_gemm_zero_vs_bf16": _paired(
            bf16, full_native, args
        ),
        "reused_metadata_quant_gemm_zero_vs_bf16": _paired(bf16, reused_native, args),
    }
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            bf16()
            full_native()
            torch.cuda.synchronize()
        profile.export_chrome_trace(
            str(
                Path(args.output).with_suffix(
                    f'.layer{rows[0]["layer"]}.{n}x{m}.trace.json'
                )
            )
        )
    return {
        "routing": rows,
        "N_x": n,
        "M_dy": m,
        "active_consumer_config": config,
        "is_tuned": is_tuned,
        "input_values": "synthetic BF16 with seed19; routing counts are real",
        "accuracy": errors,
        "stages": stages,
        "pairs": pairs,
        "native_stats": get_mxfp4_wgrad_stats(),
        "measurement_boundary": "two MB Wgrad incl FP32 zero/accum; excludes F/D and optimizer",
        "kernel_evidence": "trace and compiled HSACO must be checked separately; enqueue stats are not evidence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-jsonl", required=True)
    parser.add_argument("--layers", default="0,27")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--shapes", default="2048x1536,768x2048")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("triton", "gluon"), default="triton")
    parser.add_argument("--config-file")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(19)
    rows = [
        json.loads(line)
        for line in Path(args.route_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(args.device),
        "settings": vars(args),
        "windows": [],
    }
    if args.config_file:
        result["config_file_sha256"] = hashlib.sha256(
            Path(args.config_file).read_bytes()
        ).hexdigest()
    for layer in map(int, args.layers.split(",")):
        window = sorted(
            (
                r
                for r in rows
                if r["rank"] == 0
                and r["step"] == args.step
                and r["layer"] == layer
                and r["mb"] in (0, 1)
            ),
            key=lambda r: r["mb"],
        )
        if len(window) != 2 or [r["mb"] for r in window] != [0, 1]:
            raise ValueError(
                "need two real microbatches from the same rank, step and layer"
            )
        for n, m in (tuple(map(int, s.split("x"))) for s in args.shapes.split(",")):
            result["windows"].append(_window(window, n, m, args))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
