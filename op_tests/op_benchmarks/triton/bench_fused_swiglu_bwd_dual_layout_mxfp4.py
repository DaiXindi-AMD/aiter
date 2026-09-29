# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark split backward plus two dual quantizers against the fused path."""

import argparse
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import aiter
import torch
import triton

from aiter._version import __version__ as AITER_VERSION
from aiter.ops.triton.activation import swiglu_bwd_split
from aiter.ops.triton.quant import (
    dual_layout_quant_mxfp4,
    fused_swiglu_bwd_dual_layout_mxfp4,
)
from aiter.ops.triton.utils._triton import arch_info

_BLOCK_SIZE = 32
_DEFAULT_SHAPES = ((16384, 4096), (16384, 12288), (16384, 24576))
_PROVIDERS = (
    "unfused-canonical",
    "fused-canonical",
    "unfused-swizzle",
    "fused-swizzle",
)
_GATE_SEED = 0x1234
_GATE_OFFSET = 0x100000
_UP_SEED = 0x5678
_UP_OFFSET = 0x200000


@dataclass(frozen=True)
class BenchmarkResult:
    m: int
    d: int
    provider: str
    p20_ms: float
    median_ms: float
    p80_ms: float
    speedup_vs_matching_unfused: float | None
    contract_gbps: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _unfused(
    grad: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swizzle_scale: bool,
):
    dgate, dup = swiglu_bwd_split(grad, gate, up)
    dgate_layout = dual_layout_quant_mxfp4(
        dgate,
        use_sr=True,
        philox_seed=_GATE_SEED,
        philox_offset=_GATE_OFFSET,
        swizzle_scale=swizzle_scale,
    )
    dup_layout = dual_layout_quant_mxfp4(
        dup,
        use_sr=True,
        philox_seed=_UP_SEED,
        philox_offset=_UP_OFFSET,
        swizzle_scale=swizzle_scale,
    )
    return dgate, dup, *dgate_layout, *dup_layout


def _fused(
    grad: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swizzle_scale: bool,
):
    return fused_swiglu_bwd_dual_layout_mxfp4(
        grad,
        gate,
        up,
        use_sr=True,
        gate_philox_seed=_GATE_SEED,
        gate_philox_offset=_GATE_OFFSET,
        up_philox_seed=_UP_SEED,
        up_philox_offset=_UP_OFFSET,
        swizzle_scale=swizzle_scale,
    )


def _build_provider(
    provider: str,
    grad: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> Callable[[], object]:
    swizzle_scale = provider.endswith("swizzle")
    if provider.startswith("unfused"):
        return lambda: _unfused(grad, gate, up, swizzle_scale=swizzle_scale)
    if provider.startswith("fused"):
        return lambda: _fused(grad, gate, up, swizzle_scale=swizzle_scale)
    raise ValueError(f"unknown provider: {provider}")


def _validate_shape(shape: tuple[int, int], providers: Sequence[str]) -> None:
    m, d = shape
    if m <= 0 or d <= 0 or m % _BLOCK_SIZE or d % _BLOCK_SIZE:
        raise ValueError(f"shape must contain positive multiples of 32: {shape}")
    if any(provider.endswith("swizzle") for provider in providers):
        if m % 256 or d % 256:
            raise ValueError(
                "scale-swizzled providers require M and D divisible by 256, "
                f"got {shape}"
            )


def _contract_bytes(m: int, d: int) -> int:
    elements = m * d
    input_and_gradient_bf16 = 5 * elements * 2
    packed_payloads = 2 * elements
    scales = 4 * elements // _BLOCK_SIZE
    return input_and_gradient_bf16 + packed_payloads + scales


def _git_commit(source_file: str | None) -> str:
    if source_file is None:
        return "unknown"
    source_path = Path(source_file).resolve()
    repository = next(
        (parent for parent in source_path.parents if (parent / ".git").exists()),
        None,
    )
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


def run_benchmark(args) -> list[BenchmarkResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"--device must name a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError("fused SwiGLU backward dual-layout MXFP4 requires gfx950")

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
        grad = torch.randn(shape, dtype=torch.bfloat16, device=device)
        gate = torch.randn_like(grad)
        up = torch.randn_like(grad)
        contract_bytes = _contract_bytes(m, d)
        shape_results = []

        for provider in providers:
            function = _build_provider(provider, grad, gate, up)
            result = function()
            torch.cuda.synchronize(device)
            del result
            median_ms, p20_ms, p80_ms = triton.testing.do_bench(
                function,
                warmup=args.warmup,
                rep=args.repetitions,
                quantiles=[0.5, 0.2, 0.8],
            )
            shape_results.append(
                {
                    "provider": provider,
                    "p20_ms": float(p20_ms),
                    "median_ms": float(median_ms),
                    "p80_ms": float(p80_ms),
                    "contract_gbps": contract_bytes / (median_ms * 1e-3) * 1e-9,
                }
            )

        baselines = {
            row["provider"].removeprefix("unfused-"): row["median_ms"]
            for row in shape_results
            if row["provider"].startswith("unfused-")
        }
        for row in shape_results:
            layout = row["provider"].split("-", 1)[1]
            baseline = baselines.get(layout)
            results.append(
                BenchmarkResult(
                    m=m,
                    d=d,
                    provider=row["provider"],
                    p20_ms=row["p20_ms"],
                    median_ms=row["median_ms"],
                    p80_ms=row["p80_ms"],
                    speedup_vs_matching_unfused=(
                        None if baseline is None else baseline / row["median_ms"]
                    ),
                    contract_gbps=row["contract_gbps"],
                )
            )
        del grad, gate, up
    return results


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
    print(f"  AITER source: {Path(aiter.__file__).resolve()}")
    print("  SR: Philox4x32-7, explicit gate/up streams")
    print(f"  Warmup: {args.warmup} ms")
    print(f"  Repetitions: {args.repetitions} ms")


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    print(
        "M,D,provider,p20_ms,median_ms,p80_ms,"
        "speedup_vs_matching_unfused,contract_gbps"
    )
    for result in results:
        speedup = (
            "n/a"
            if result.speedup_vs_matching_unfused is None
            else f"{result.speedup_vs_matching_unfused:.4f}"
        )
        print(
            f"{result.m},{result.d},{result.provider},{result.p20_ms:.6f},"
            f"{result.median_ms:.6f},{result.p80_ms:.6f},{speedup},"
            f"{result.contract_gbps:.3f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact-shape split-backward plus two production-equivalent "
            "dual-layout quantizers against the fused AITER candidate."
        )
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=_positive_int,
        metavar=("M", "D"),
        help="Run one exact shape instead of the default training shapes.",
    )
    parser.add_argument(
        "--providers",
        default=",".join(_PROVIDERS),
        help=f"Comma-separated subset of {','.join(_PROVIDERS)}.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--warmup", type=_positive_int, default=25)
    parser.add_argument("--repetitions", type=_positive_int, default=100)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    results = run_benchmark(args)
    _print_environment(args)
    _print_results(results)


if __name__ == "__main__":
    main()
