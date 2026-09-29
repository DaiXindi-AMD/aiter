# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark fused SwiGLU plus row/rotated-column MXFP4 construction.

The legacy unfused provider includes a Torch H16 temporary. Fair two-stage
providers instead call the public split SwiGLU and dual-layout quantizer APIs.
Fused variants are compared with the matching two-stage output layout.
"""

from __future__ import annotations

import argparse
import csv
import gc
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import aiter
import torch
import triton

from aiter._version import __version__ as AITER_VERSION
from aiter.ops.triton import quant as quant_ops
from aiter.ops.triton.activation import swiglu_fwd_split
from aiter.ops.triton.quant import dual_layout_quant_mxfp4, dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton import arch_info

_FUSED_API_NAME = "fused_swiglu_dual_layout_mxfp4"
_HADAMARD_SIZE = 16
_BLOCK_SIZE = 32
_DEFAULT_SHAPES = ((16384, 4096), (16384, 12288), (16384, 24576))
_PROVIDERS = (
    "unfused",
    "two-stage-canonical",
    "two-stage-swizzle",
    "two-stage-swizzle-shuffle",
    "fused-canonical",
    "fused-swizzle",
    "fused-swizzle-shuffle",
)
_PROVIDER_CONTRACTS = {
    "unfused": "split SwiGLU + row quant + Torch H16/transpose + col quant",
    "two-stage-canonical": "split SwiGLU + public dual-layout quantizer",
    "two-stage-swizzle": "two-stage canonical + row/col scale swizzle",
    "two-stage-swizzle-shuffle": "two-stage swizzle + col B-payload shuffle",
    "fused-canonical": "one fused call, canonical row/col payload and scales",
    "fused-swizzle": "fused canonical + row/col scale swizzle",
    "fused-swizzle-shuffle": "fused swizzle + col B-payload shuffle",
}


@dataclass(frozen=True)
class BenchmarkResult:
    m: int
    d: int
    provider: str
    p20_ms: float
    median_ms: float
    p80_ms: float
    speedup_vs_unfused: float | None
    speedup_vs_two_stage: float | None
    contract_gbps: float
    peak_increment_mib: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _fused_api():
    return getattr(quant_ops, _FUSED_API_NAME)


def _normalized_hadamard16(device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < _HADAMARD_SIZE:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / (_HADAMARD_SIZE**0.5)


def _unfused(
    gate: torch.Tensor,
    up: torch.Tensor,
    hadamard: torch.Tensor,
):
    activation = swiglu_fwd_split(gate, up)
    row_packed, row_scale = dynamic_mxfp4_quant(activation)
    m, d = activation.shape
    transposed_blocks = activation.T.float().reshape(
        d,
        m // _HADAMARD_SIZE,
        _HADAMARD_SIZE,
    )
    rotated = (transposed_blocks @ hadamard).reshape(d, m)
    col_packed, col_scale = dynamic_mxfp4_quant(rotated)
    return activation, row_packed, row_scale, col_packed, col_scale


def _two_stage(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swizzle_scale: bool,
    shuffle_col: bool,
):
    activation = swiglu_fwd_split(gate, up)
    return (
        activation,
        *dual_layout_quant_mxfp4(
            activation,
            swizzle_scale=swizzle_scale,
            shuffle_col=shuffle_col,
        ),
    )


def _build_provider(
    provider: str,
    gate: torch.Tensor,
    up: torch.Tensor,
    hadamard: torch.Tensor,
) -> Callable[[], object]:
    if provider == "unfused":
        return lambda: _unfused(gate, up, hadamard)
    if provider == "two-stage-canonical":
        return lambda: _two_stage(gate, up, swizzle_scale=False, shuffle_col=False)
    if provider == "two-stage-swizzle":
        return lambda: _two_stage(gate, up, swizzle_scale=True, shuffle_col=False)
    if provider == "two-stage-swizzle-shuffle":
        return lambda: _two_stage(gate, up, swizzle_scale=True, shuffle_col=True)

    fused = _fused_api()
    if provider == "fused-canonical":
        return lambda: fused(gate, up)
    if provider == "fused-swizzle":
        return lambda: fused(gate, up, swizzle_scale=True)
    if provider == "fused-swizzle-shuffle":
        return lambda: fused(
            gate,
            up,
            swizzle_scale=True,
            shuffle_col=True,
        )
    raise ValueError(f"unknown provider: {provider}")


def _contract_bytes(m: int, d: int) -> int:
    elements = m * d
    bf16_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
    input_bytes = 2 * elements * bf16_bytes
    activation_bytes = elements * bf16_bytes
    packed_bytes = elements
    scale_bytes = 2 * elements // _BLOCK_SIZE
    return input_bytes + activation_bytes + packed_bytes + scale_bytes


def _measure_peak_increment(
    function: Callable[[], object],
    device: torch.device,
) -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = function()
    torch.cuda.synchronize(device)
    peak_increment = torch.cuda.max_memory_allocated(device) - baseline
    del result
    return peak_increment


def _validate_shape(shape: tuple[int, int], providers: Sequence[str]) -> None:
    m, d = shape
    if m <= 0 or d <= 0 or m % _BLOCK_SIZE or d % _BLOCK_SIZE:
        raise ValueError(f"shape must contain positive multiples of 32: {shape}")
    if any("swizzle" in provider for provider in providers):
        if m % 256 or d % 256:
            raise ValueError(
                "scale-swizzled providers require M and D divisible by 256, "
                f"got {shape}"
            )
    if any(provider.endswith("swizzle-shuffle") for provider in providers) and m % 64:
        raise ValueError(f"column B shuffle requires M divisible by 64, got {m}")


def _assert_exact(actual, expected, provider: str) -> None:
    names = ("activation", "row_packed", "row_scale", "col_packed", "col_scale")
    for name, result, reference in zip(names, actual, expected, strict=True):
        torch.testing.assert_close(
            result,
            reference,
            atol=0,
            rtol=0,
            msg=lambda message: f"{provider} {name} mismatch: {message}",
        )


def _git_repository(source_file: str | None) -> Path | None:
    if source_file is None:
        return None
    source_path = Path(source_file).resolve()
    return next(
        (parent for parent in source_path.parents if (parent / ".git").exists()),
        None,
    )


def _git_commit(source_file: str | None) -> str:
    repository = _git_repository(source_file)
    if repository is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip()


def _git_dirty(source_file: str | None) -> str:
    repository = _git_repository(source_file)
    if repository is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return "yes" if result.stdout else "no"


def _print_environment(args) -> None:
    device = torch.device(args.device)
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", arch_info.get_arch())
    print("Environment")
    print(f"  Device: {properties.name} ({architecture})")
    print(f"  ROCm: {torch.version.hip or 'not available'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Triton: {triton.__version__}")
    print(f"  AITER version: {AITER_VERSION}")
    print(f"  AITER commit: {_git_commit(aiter.__file__)}")
    print(f"  AITER tree dirty: {_git_dirty(aiter.__file__)}")
    print(f"  AITER source: {Path(aiter.__file__).resolve()}")
    print(f"  Warmup: {args.warmup} ms")
    print(f"  Repetitions: {args.repetitions} ms")
    print("  Unfused baseline includes Torch H16 transform and temporaries")
    print("Providers")
    for provider in args.providers.split(","):
        print(f"  {provider}: {_PROVIDER_CONTRACTS[provider]}")


def run_benchmark(args) -> list[BenchmarkResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"--device must name a CUDA device, got {device}")
    torch.cuda.set_device(device)
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError("fused SwiGLU dual-layout MXFP4 requires gfx950")

    providers = tuple(args.providers.split(","))
    unknown = set(providers) - set(_PROVIDERS)
    if unknown:
        raise ValueError(f"unknown providers: {sorted(unknown)}")
    shapes = [tuple(args.shape)] if args.shape else list(_DEFAULT_SHAPES)
    results = []

    for shape in shapes:
        _validate_shape(shape, providers)
        m, d = shape
        torch.manual_seed(args.seed)
        gate = torch.randn((m, d), dtype=torch.bfloat16, device=device)
        up = torch.randn_like(gate)
        hadamard = _normalized_hadamard16(device)
        contract_bytes = _contract_bytes(m, d)
        shape_results = []

        for suffix in ("canonical", "swizzle", "swizzle-shuffle"):
            fused_provider = f"fused-{suffix}"
            if fused_provider not in providers:
                continue
            reference = _build_provider(f"two-stage-{suffix}", gate, up, hadamard)()
            actual = _build_provider(fused_provider, gate, up, hadamard)()
            torch.cuda.synchronize(device)
            _assert_exact(actual, reference, fused_provider)
            del actual, reference

        for provider in providers:
            function = _build_provider(provider, gate, up, hadamard)
            result = function()
            torch.cuda.synchronize(device)
            del result
            median_ms, p20_ms, p80_ms = triton.testing.do_bench(
                function,
                warmup=args.warmup,
                rep=args.repetitions,
                quantiles=[0.5, 0.2, 0.8],
            )
            peak_increment = _measure_peak_increment(function, device)
            shape_results.append(
                {
                    "provider": provider,
                    "p20_ms": float(p20_ms),
                    "median_ms": float(median_ms),
                    "p80_ms": float(p80_ms),
                    "contract_gbps": contract_bytes / (median_ms * 1e-3) * 1e-9,
                    "peak_increment_mib": peak_increment / (1024**2),
                }
            )

        baseline = next(
            (row["median_ms"] for row in shape_results if row["provider"] == "unfused"),
            None,
        )
        two_stage_baselines = {
            row["provider"].removeprefix("two-stage-"): row["median_ms"]
            for row in shape_results
            if row["provider"].startswith("two-stage-")
        }
        for row in shape_results:
            speedup = None if baseline is None else baseline / row["median_ms"]
            suffix = row["provider"].removeprefix("fused-")
            two_stage_baseline = two_stage_baselines.get(suffix)
            speedup_vs_two_stage = (
                None
                if two_stage_baseline is None
                else two_stage_baseline / row["median_ms"]
            )
            results.append(
                BenchmarkResult(
                    m=m,
                    d=d,
                    provider=row["provider"],
                    p20_ms=row["p20_ms"],
                    median_ms=row["median_ms"],
                    p80_ms=row["p80_ms"],
                    speedup_vs_unfused=speedup,
                    speedup_vs_two_stage=speedup_vs_two_stage,
                    contract_gbps=row["contract_gbps"],
                    peak_increment_mib=row["peak_increment_mib"],
                )
            )
        del gate, up, hadamard
        gc.collect()
        torch.cuda.empty_cache()
    return results


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    fields = (
        "M",
        "D",
        "provider",
        "p20_ms",
        "median_ms",
        "p80_ms",
        "speedup_vs_unfused",
        "speedup_vs_two_stage",
        "contract_gbps",
        "peak_increment_mib",
    )
    print(",".join(fields))
    for result in results:
        print(
            ",".join(
                (
                    str(result.m),
                    str(result.d),
                    result.provider,
                    f"{result.p20_ms:.6f}",
                    f"{result.median_ms:.6f}",
                    f"{result.p80_ms:.6f}",
                    _format_optional(result.speedup_vs_unfused),
                    _format_optional(result.speedup_vs_two_stage),
                    f"{result.contract_gbps:.3f}",
                    f"{result.peak_increment_mib:.3f}",
                )
            )
        )


def _write_csv(path: Path, results: Sequence[BenchmarkResult]) -> None:
    fieldnames = list(BenchmarkResult.__dataclass_fields__)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: getattr(result, field) for field in fieldnames})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark fused SwiGLU dual-layout MXFP4 construction."
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=_positive_int,
        metavar=("M", "D"),
        help="Run one shape instead of 16384x4096/12288/24576.",
    )
    parser.add_argument(
        "--providers",
        default=",".join(_PROVIDERS),
        help=f"Comma-separated subset of {','.join(_PROVIDERS)}.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--warmup", type=_positive_int, default=25)
    parser.add_argument("--repetitions", type=_positive_int, default=100)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    providers = tuple(args.providers.split(","))
    unknown = set(providers) - set(_PROVIDERS)
    if unknown:
        raise ValueError(f"unknown providers: {sorted(unknown)}")
    results = run_benchmark(args)
    _print_environment(args)
    _print_results(results)
    if args.csv:
        _write_csv(args.csv, results)


if __name__ == "__main__":
    main()
