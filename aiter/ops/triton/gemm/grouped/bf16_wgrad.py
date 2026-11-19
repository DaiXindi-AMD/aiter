# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public FP32 Wgrad control using the existing Sonic grouped GEMM backend."""

import torch

from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import prepare_expert_offsets
from aiter.ops.triton.moe.sonicmoe import grouped_gemm
from aiter.ops.triton.utils.mxfp4_wgrad import normalize_counts

__all__ = ["grouped_bf16_wgrad"]


def grouped_bf16_wgrad(
    x,
    dy,
    counts,
    *,
    out=None,
    accumulate=False,
    offsets=None,
    transpose_output=False,
):
    """Compute dY.T @ X into FP32 [E,M,N], or directly [E,N,M] when transposed.

    X[R,N]/dY[R,M] are contiguous CUDA BF16. Reuse the F/D real offsets when
    supplied. The existing backend owns config and fused FP32 accumulation.
    Empty experts give zero in overwrite mode and no contribution when adding.
    """
    for value in (x, dy):
        if (
            value.ndim != 2
            or not value.is_cuda
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
        ):
            raise ValueError("operands must be contiguous CUDA BF16 matrices")
    if x.device != dy.device or x.shape[0] != dy.shape[0]:
        raise ValueError("operands must have the same rows and device")
    counts = normalize_counts(counts, x.shape[0])
    shape = (
        (len(counts), x.shape[1], dy.shape[1])
        if transpose_output
        else (len(counts), dy.shape[1], x.shape[1])
    )
    if out is None:
        if accumulate:
            raise ValueError("accumulate requires initialized FP32 out")
        out = torch.empty(shape, dtype=torch.float32, device=x.device)
    if (
        out.shape != shape
        or out.dtype != torch.float32
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous FP32 with the requested shape/device")
    if any(torch._C._overlaps(out, value) for value in (x, dy)):
        raise ValueError("out must not overlap an operand")
    if not sum(counts):
        if not accumulate:
            out.zero_()
        return out
    if offsets is None:
        offsets = prepare_expert_offsets(counts, x.device)
    if (
        offsets.shape != (len(counts) + 1,)
        or offsets.dtype != torch.int32
        or offsets.device != x.device
    ):
        raise ValueError("offsets must be CUDA int32 [E+1] from the routing")
    lhs, rhs = (x, dy) if transpose_output else (dy, x)
    return grouped_gemm(
        lhs, rhs, offsets, out=out, A_is_transposed=True, accumulate=accumulate
    )
