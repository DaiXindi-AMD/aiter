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

__all__ = ["row_scale_fp32"]

_row_scale_repr = make_kernel_repr("_row_scale_kernel", ["BLOCK_N"])


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
