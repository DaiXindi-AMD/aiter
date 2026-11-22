# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Row-wise FP32 score scaling fused into one memory pass.

``out = (x.to(fp32) * score[row]).to(x.dtype)`` is the routed-expert weighting. In
PyTorch it expands into an upcast, a broadcast multiply and a downcast, i.e. four
times the traffic of the fused form and three extra launches. The arithmetic is
per-element, so fusing it is bit-exact with the unfused chain.
"""

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

__all__ = ["row_dot_fp32", "row_scale_fp32"]

_row_scale_repr = make_kernel_repr("_row_scale_kernel", ["BLOCK_N"])
_row_dot_repr = make_kernel_repr("_row_dot_kernel", ["BLOCK_N"])


@triton.jit(repr=_row_scale_repr)
def _row_scale_kernel(
    x_ptr,
    score_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    value = tl.load(x_ptr + row.to(tl.int64) * N + col, mask=mask, other=0.0).to(
        tl.float32
    )
    score = tl.load(score_ptr + row)
    # fp32 multiply then a single round to the destination dtype: identical to
    # (x.float() * score[:, None]).to(x.dtype), minus the float32 temporaries.
    tl.store(
        out_ptr + row.to(tl.int64) * N + col,
        (value * score).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(repr=_row_dot_repr)
def _row_dot_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        col = start + tl.arange(0, BLOCK_N)
        mask = col < N
        a = tl.load(a_ptr + row * N + col, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * N + col, mask=mask, other=0.0).to(tl.float32)
        acc += a * b
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


def row_dot_fp32(a, b, block_n=2048):
    """Return ``(a.float() * b.float()).sum(-1)`` without FP32 temporaries.

    Products and accumulation are FP32 as in the eager form; only the summation
    order differs, so results agree to FP32 rounding rather than bit-for-bit.
    """
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("expected a[R,N] and b[R,N]")
    if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
        raise ValueError("inputs must share a BF16 or FP16 dtype")
    if b.device != a.device or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("inputs must be contiguous on one device")
    rows, n = a.shape
    out = torch.empty((rows,), dtype=torch.float32, device=a.device)
    if rows == 0:
        return out
    if n == 0:
        return out.zero_()
    _row_dot_kernel[(rows,)](a, b, out, n, BLOCK_N=block_n)
    return out


def row_scale_fp32(x, score, out=None, block_n=2048):
    """Return ``(x.float() * score[:, None]).to(x.dtype)`` for contiguous 2-D inputs."""
    if x.ndim != 2 or score.ndim != 1 or score.shape[0] != x.shape[0]:
        raise ValueError("expected x[R,N] and score[R]")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x must be BF16 or FP16")
    if score.dtype != torch.float32 or score.device != x.device:
        raise ValueError("score must be FP32 on x.device")
    if not x.is_contiguous() or not score.is_contiguous():
        raise ValueError("inputs must be contiguous")
    rows, n = x.shape
    if out is None:
        out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    elif out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError("out must be contiguous with x's shape and dtype")
    if rows == 0:
        return out
    grid = (rows, triton.cdiv(n, block_n))
    _row_scale_kernel[grid](x, score, out, n, BLOCK_N=block_n)
    return out
