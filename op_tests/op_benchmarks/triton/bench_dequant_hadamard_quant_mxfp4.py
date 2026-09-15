# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant import (
    dequant_hadamard_quant_mxfp4,
    dynamic_mxfp4_quant,
)
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32


def get_default_shapes() -> list[tuple[int, int]]:
    return [
        (512, 4096),
        (2048, 4096),
        (4096, 4096),
        (6144, 4096),
        (4096, 12288),
        (12288, 4096),
    ]


def _hadamard16(device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < 16:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * 0.25


def _decomposed_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    sign: torch.Tensor,
    hadamard: torch.Tensor,
):
    values = mxfp4_to_f32(packed)
    scale = e8m0_to_f32(scales).repeat_interleave(32, dim=1)
    transposed = (values * scale).to(torch.bfloat16).t().contiguous()
    K, M = transposed.shape
    rotated = (
        transposed.float().reshape(K, M // 16, 16) * sign.reshape(1, 1, 16)
    ) @ hadamard
    return dynamic_mxfp4_quant(rotated.reshape(K, M))


def run_benchmark(args):
    shapes = [tuple(args.shape)] if args.shape else get_default_shapes()
    providers = args.provider.split(",")

    benchmark = triton.testing.Benchmark(
        x_names=["M", "K"],
        x_vals=shapes,
        line_arg="provider",
        line_vals=providers,
        line_names=providers,
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Time (ms)" if args.metric == "time" else "Bandwidth (GB/s)",
        plot_name="dequant-hadamard-quant-mxfp4",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench(M, K, provider, metric):
        torch.manual_seed(17)
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        packed, scales = dynamic_mxfp4_quant(x)
        sign = torch.ones(16, dtype=torch.float32, device=x.device)
        hadamard = _hadamard16(x.device)

        if provider == "fused":

            def fn():
                return dequant_hadamard_quant_mxfp4(packed, scales, sign)

        elif provider == "decomposed":

            def fn():
                return _decomposed_reference(packed, scales, sign, hadamard)

        else:
            raise ValueError(f"unknown provider: {provider}")

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if metric == "time":
            return ms
        input_bytes = (
            packed.numel() + scales.numel() + sign.numel() * sign.element_size()
        )
        output_bytes = K * (M // 2) + K * (M // 32)
        return (input_bytes + output_bytes) / (ms * 1e-3) * 1e-9

    bench.run(save_path="." if args.output else None, print_data=True)


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark fused MXFP4 dequant-transpose-H16-requant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "K"),
        help="single logical input shape",
    )
    parser.add_argument(
        "--provider",
        default="fused,decomposed",
        help="comma-separated providers: fused,decomposed",
    )
    parser.add_argument(
        "--metric",
        choices=("time", "bandwidth"),
        default="time",
    )
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args(args)


def main(args: list[str] | None = None):
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    sys.exit(main())
