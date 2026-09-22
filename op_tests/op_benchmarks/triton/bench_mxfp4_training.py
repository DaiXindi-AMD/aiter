# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the migrated gfx950 MXFP4 training quantization paths."""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant.mxfp4 import convert_to_mxfp4
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)


def _get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _quantize(x: torch.Tensor, mode: str):
    if mode == "sr":
        return convert_to_mxfp4(x, use_sr=True)
    if mode == "rtn_swizzle":
        return convert_to_mxfp4(x, swizzle_scale=True)
    raise ValueError(f"unsupported mode: {mode}")


def run_benchmark(args) -> None:
    shapes = [tuple(args.shape)] if args.shape else [(32, 256), (256, 4096)]
    modes = args.mode.split(",")
    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=shapes,
        line_arg="mode",
        line_vals=modes,
        line_names=modes,
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Bandwidth (GB/s)",
        plot_name=get_caller_name_no_ext(),
        args={"dtype": args.dtype},
    )

    @triton.testing.perf_report([benchmark])
    def bench_mxfp4_training(M, N, mode, dtype, **_):
        if M % 32 or N % 256:
            raise ValueError("migrated swizzled shapes require M % 32 == N % 256 == 0")
        x = torch.randn((M, N), dtype=_get_dtype(dtype), device="cuda")
        ms = triton.testing.do_bench(
            lambda: _quantize(x, mode), warmup=25, rep=100
        )
        total_bytes = x.numel() * x.element_size() + M * (N // 2) + M * (N // 32)
        return total_bytes / (ms * 1e-3) * 1e-9

    bench_mxfp4_training.run(
        save_path="." if args.output else None,
        print_data=True,
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark migrated MXFP4 training quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shape", type=int, nargs=2, metavar=("M", "N"))
    parser.add_argument(
        "--mode",
        default="sr,rtn_swizzle",
        help="Comma-separated modes: sr,rtn_swizzle.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args(args)


def main(args=None) -> None:
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    sys.exit(main())
