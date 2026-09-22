# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the public split and packed AITER SwiGLU APIs.

``torch-eager`` and ``aiter-split`` use separate ``(M, D)`` gate/up tensors and
the eager dtype-rounding cut points. ``aiter-packed`` uses a prebuilt ``(M, 2D)``
tensor and legacy FP32-intermediate arithmetic, so it is a diagnostic result,
not a same-semantics speedup comparison; packing traffic is excluded.

The eager combined case recomputes SiLU for backward, matching the work done by
separate forward and backward calls. Logical bandwidth counts three tensor
equivalents for forward, five for backward, and eight combined.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import aiter
import torch
import triton

from aiter._version import __version__ as AITER_VERSION
from aiter.ops.triton.activation import (
    swiglu_bwd,
    swiglu_bwd_split,
    swiglu_fwd,
    swiglu_fwd_split,
)

DEFAULT_M_VALUES = (16384, 32768)
DEFAULT_D_VALUES = (12288,)
PROVIDERS = ("torch-eager", "aiter-split", "aiter-packed")
STAGES = ("forward", "backward", "combined")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
LOGICAL_TENSOR_COUNTS = {"forward": 3, "backward": 5, "combined": 8}
PROVIDER_CONTRACTS = {
    "torch-eager": "separate (M,D), eager dtype-rounding semantics",
    "aiter-split": "separate (M,D), eager dtype-rounding semantics",
    "aiter-packed": "prepacked (M,2D), legacy FP32-intermediate semantics",
}


@dataclass
class Workspace:
    gate: torch.Tensor
    up: torch.Tensor
    grad_output: torch.Tensor
    packed: torch.Tensor | None


@dataclass(frozen=True)
class BenchmarkResult:
    m: int
    d: int
    dtype: str
    stage: str
    provider: str
    p20_ms: float
    median_ms: float
    p80_ms: float
    logical_gbps: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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


def _print_environment(device: torch.device) -> None:
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "unknown")
    print("Environment")
    print(f"  Device: {properties.name} ({architecture})")
    print(f"  ROCm: {torch.version.hip or 'not available'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Triton: {triton.__version__}")
    print(f"  AITER version: {AITER_VERSION}")
    print(f"  AITER commit: {_git_commit(aiter.__file__)}")
    print(f"  AITER tree dirty: {_git_dirty(aiter.__file__)}")
    print(f"  AITER source: {Path(aiter.__file__).resolve()}")


def _make_workspace(
    m: int,
    d: int,
    dtype: torch.dtype,
    device: torch.device,
    need_packed: bool,
) -> Workspace:
    gate = torch.randn((m, d), dtype=dtype, device=device)
    up = torch.randn_like(gate)
    grad_output = torch.randn_like(gate)
    packed = torch.cat((gate, up), dim=-1) if need_packed else None
    return Workspace(
        gate=gate,
        up=up,
        grad_output=grad_output,
        packed=packed,
    )


def _torch_forward(workspace: Workspace) -> torch.Tensor:
    silu = torch.ops.aten.silu.default(workspace.gate)
    return torch.mul(silu, workspace.up)


def _torch_backward(workspace: Workspace) -> tuple[torch.Tensor, torch.Tensor]:
    silu = torch.ops.aten.silu.default(workspace.gate)
    grad_silu = torch.mul(
        workspace.grad_output,
        workspace.up,
    )
    grad_gate = torch.ops.aten.silu_backward.default(
        grad_silu,
        workspace.gate,
    )
    grad_up = torch.mul(
        workspace.grad_output,
        silu,
    )
    return grad_gate, grad_up


def _torch_combined(
    workspace: Workspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_silu = torch.ops.aten.silu.default(workspace.gate)
    output = torch.mul(forward_silu, workspace.up)
    grad_silu = torch.mul(
        workspace.grad_output,
        workspace.up,
    )
    grad_gate = torch.ops.aten.silu_backward.default(
        grad_silu,
        workspace.gate,
    )
    backward_silu = torch.ops.aten.silu.default(workspace.gate)
    grad_up = torch.mul(
        workspace.grad_output,
        backward_silu,
    )
    return output, grad_gate, grad_up


def _benchmark_callable(
    provider: str,
    stage: str,
    workspace: Workspace,
) -> Callable[[], object]:
    if provider == "torch-eager":
        if stage == "forward":
            return lambda: _torch_forward(workspace)
        if stage == "backward":
            return lambda: _torch_backward(workspace)
        return lambda: _torch_combined(workspace)

    if provider == "aiter-split":
        if stage == "forward":
            return lambda: swiglu_fwd_split(
                workspace.gate,
                workspace.up,
            )
        if stage == "backward":
            return lambda: swiglu_bwd_split(
                workspace.grad_output,
                workspace.gate,
                workspace.up,
            )

        def split_combined() -> object:
            output = swiglu_fwd_split(
                workspace.gate,
                workspace.up,
            )
            gradients = swiglu_bwd_split(
                workspace.grad_output,
                workspace.gate,
                workspace.up,
            )
            return output, gradients

        return split_combined

    if workspace.packed is None:
        raise ValueError("packed workspace is required for aiter-packed")
    if stage == "forward":
        return lambda: swiglu_fwd(workspace.packed)
    if stage == "backward":
        return lambda: swiglu_bwd(workspace.grad_output, workspace.packed)

    def packed_combined() -> object:
        output = swiglu_fwd(workspace.packed)
        gradients = swiglu_bwd(workspace.grad_output, workspace.packed)
        return output, gradients

    return packed_combined


def _logical_bytes(
    m: int,
    d: int,
    dtype: torch.dtype,
    stage: str,
) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return m * d * element_size * LOGICAL_TENSOR_COUNTS[stage]


def _run_shape(
    m: int,
    d: int,
    dtype_name: str,
    providers: Sequence[str],
    stages: Sequence[str],
    device: torch.device,
    warmup: int,
    rep: int,
) -> list[BenchmarkResult]:
    dtype = DTYPES[dtype_name]
    workspace = _make_workspace(
        m,
        d,
        dtype,
        device,
        need_packed="aiter-packed" in providers,
    )
    results = []
    for stage in stages:
        logical_bytes = _logical_bytes(m, d, dtype, stage)
        for provider in providers:
            function = _benchmark_callable(provider, stage, workspace)
            median_ms, p20_ms, p80_ms = triton.testing.do_bench(
                function,
                warmup=warmup,
                rep=rep,
                quantiles=[0.5, 0.2, 0.8],
            )
            results.append(
                BenchmarkResult(
                    m=m,
                    d=d,
                    dtype=dtype_name,
                    stage=stage,
                    provider=provider,
                    p20_ms=float(p20_ms),
                    median_ms=float(median_ms),
                    p80_ms=float(p80_ms),
                    logical_gbps=logical_bytes / (float(median_ms) * 1e6),
                )
            )
    return results


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    header = (
        f"{'M':>8} {'D':>8} {'dtype':>6} {'stage':>9} {'provider':>14} "
        f"{'p20_ms':>11} {'median_ms':>11} {'p80_ms':>11} {'logical_GB/s':>14}"
    )
    print("\nResults")
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.m:8d} {result.d:8d} {result.dtype:>6} "
            f"{result.stage:>9} {result.provider:>14} "
            f"{result.p20_ms:11.6f} {result.median_ms:11.6f} "
            f"{result.p80_ms:11.6f} {result.logical_gbps:14.3f}"
        )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--m",
        dest="m_values",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_M_VALUES),
        help="row counts to benchmark",
    )
    parser.add_argument(
        "--d",
        dest="d_values",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_D_VALUES),
        help="features per gate/up input",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bf16",
        help="input and output dtype",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="implementations to benchmark",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="SwiGLU stages to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=_positive_int,
        default=25,
        help="warmup duration passed to triton.testing.do_bench in milliseconds",
    )
    parser.add_argument(
        "--rep",
        type=_positive_int,
        default=100,
        help="measurement duration passed to triton.testing.do_bench in milliseconds",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", type=int, default=20260920, help="random seed")
    return parser.parse_args(args)


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("SwiGLU benchmark requires a CUDA/ROCm device")

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    _print_environment(device)
    print(
        "  Timing: triton.testing.do_bench GPU events; "
        f"warmup={args.warmup} ms, rep={args.rep} ms"
    )
    print("  Packed diagnostic: inputs are pre-packed; concatenation is not timed")
    print(
        "  Scope: direct public API latency; backward/combined do not include "
        "a framework autograd graph"
    )
    print("  Provider contracts:")
    for provider in args.providers:
        print(f"    {provider}: {PROVIDER_CONTRACTS[provider]}")
    print(
        f"  Inputs: M={args.m_values}, D={args.d_values}, dtype={args.dtype}, "
        f"providers={args.providers}, stages={args.stages}, seed={args.seed}"
    )

    results = []
    for m in args.m_values:
        for d in args.d_values:
            shape_results = _run_shape(
                m,
                d,
                args.dtype,
                args.providers,
                args.stages,
                device,
                args.warmup,
                args.rep,
            )
            results.extend(shape_results)
            del shape_results
            gc.collect()
            torch.cuda.empty_cache()

    _print_results(results)
    return results


def main(args: Sequence[str] | None = None) -> None:
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    main()
