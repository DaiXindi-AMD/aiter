# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark two-source 32x32 block-scaled MXFP4 weight construction."""

import argparse
import csv
import gc
from pathlib import Path
import subprocess

import torch
import triton

import aiter
from aiter.ops.triton.quant import (
    dynamic_mxfp4_quant_2way,
    dynamic_mxfp4_quant_blockscale,
)

_BLOCK_SIZE = 32
_DEFAULT_SHAPES = [
    (12288, 12288, 4096),
    (6144, 6144, 4096),
    (3072, 3072, 4096),
    (4096, 8192, 4096),
]
_PROVIDERS = ("cat_quant", "two_source", "separate_compact_cat", "precat_quant")


def _build_provider(
    provider: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    shuffle_data: bool,
):
    if provider == "cat_quant":
        return lambda: dynamic_mxfp4_quant_blockscale(
            torch.cat((x0, x1), dim=0), shuffle_data=shuffle_data
        )
    if provider == "two_source":
        return lambda: dynamic_mxfp4_quant_2way(x0, x1, shuffle_data=shuffle_data)
    if provider == "separate_compact_cat":

        def separate_compact_cat():
            packed0, scales0 = dynamic_mxfp4_quant_blockscale(
                x0, shuffle_data=shuffle_data
            )
            packed1, scales1 = dynamic_mxfp4_quant_blockscale(
                x1, shuffle_data=shuffle_data
            )
            return torch.cat((packed0, packed1), dim=0), torch.cat(
                (scales0, scales1), dim=0
            )

        return separate_compact_cat
    if provider == "precat_quant":
        concatenated = torch.cat((x0, x1), dim=0)
        return lambda: dynamic_mxfp4_quant_blockscale(
            concatenated, shuffle_data=shuffle_data
        )
    raise ValueError(f"unknown provider: {provider}")


def _measure_peak_increment(fn) -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak_increment = torch.cuda.max_memory_allocated() - baseline
    del result
    return peak_increment


def _semantic_bytes(rows0: int, rows1: int, cols: int) -> int:
    rows = rows0 + rows1
    input_bytes = rows * cols * torch.tensor([], dtype=torch.bfloat16).element_size()
    packed_bytes = rows * cols // 2
    scale_bytes = rows * cols // (_BLOCK_SIZE * _BLOCK_SIZE)
    return input_bytes + packed_bytes + scale_bytes


def _validate_shape(shape: tuple[int, int, int], shuffle_data: bool) -> None:
    rows0, rows1, cols = shape
    if any(dim <= 0 or dim % _BLOCK_SIZE != 0 for dim in shape):
        raise ValueError(f"shape dimensions must be positive multiples of 32: {shape}")
    if shuffle_data and cols % 64 != 0:
        raise ValueError(f"shuffled output requires K divisible by 64, got {cols}")


def _revision() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_benchmark(args) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    shapes = [tuple(args.shape)] if args.shape else _DEFAULT_SHAPES
    layouts = [False, True] if args.layout == "all" else [args.layout == "shuffled"]
    providers = tuple(args.providers.split(","))
    unknown = set(providers) - set(_PROVIDERS)
    if unknown:
        raise ValueError(f"unknown providers: {sorted(unknown)}")

    rows = []
    for shape in shapes:
        rows0, rows1, cols = shape
        for shuffle_data in layouts:
            _validate_shape(shape, shuffle_data)
            x0 = torch.randn((rows0, cols), dtype=torch.bfloat16, device=args.device)
            x1 = torch.randn((rows1, cols), dtype=torch.bfloat16, device=args.device)
            semantic_bytes = _semantic_bytes(rows0, rows1, cols)

            for provider in providers:
                fn = _build_provider(provider, x0, x1, shuffle_data)
                result = fn()
                torch.cuda.synchronize()
                del result
                latency_ms = triton.testing.do_bench(
                    fn, warmup=args.warmup, rep=args.repetitions
                )
                peak_increment = _measure_peak_increment(fn)
                bandwidth_gbps = semantic_bytes / (latency_ms * 1e-3) * 1e-9
                rows.append(
                    {
                        "M0": rows0,
                        "M1": rows1,
                        "K": cols,
                        "layout": "shuffled" if shuffle_data else "row_major",
                        "provider": provider,
                        "latency_ms": latency_ms,
                        "effective_bandwidth_gbps": bandwidth_gbps,
                        "peak_increment_mib": peak_increment / (1024**2),
                    }
                )
    return rows


def _print_metadata(args) -> None:
    device = torch.device(args.device)
    print(f"device={torch.cuda.get_device_name(device)}")
    print(
        f"torch={torch.__version__} hip={torch.version.hip} triton={triton.__version__}"
    )
    print(f"aiter={aiter.__file__} commit={_revision()}")
    print(f"warmup_ms={args.warmup} repetitions_ms={args.repetitions}")


def _print_rows(rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    print(",".join(fields))
    for row in rows:
        print(
            ",".join(
                (
                    f"{row[field]:.6f}"
                    if isinstance(row[field], float)
                    else str(row[field])
                )
                for field in fields
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark two-source 32x32 MXFP4 quantization."
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        metavar=("M0", "M1", "K"),
        help="Run one shape instead of the default Qwen and TP cases.",
    )
    parser.add_argument(
        "--layout",
        choices=("all", "row-major", "shuffled"),
        default="all",
    )
    parser.add_argument(
        "--providers",
        default=",".join(_PROVIDERS),
        help=f"Comma-separated subset of {','.join(_PROVIDERS)}.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup duration in ms.")
    parser.add_argument(
        "--repetitions", type=int, default=100, help="Measurement duration in ms."
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    rows = run_benchmark(args)
    _print_metadata(args)
    _print_rows(rows)
    if args.csv:
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
