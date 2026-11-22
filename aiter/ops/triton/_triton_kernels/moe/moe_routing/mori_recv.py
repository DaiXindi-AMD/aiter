# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Receive-side local expert ids and per-expert counts after a MORI EP dispatch."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_mori_recv_repr = make_kernel_repr(
    "_mori_recv_local_routing_kernel",
    ["TOPK", "NUM_LOCAL", "NUM_BINS", "BLOCK"],
)


@triton.jit(repr=_mori_recv_repr)
def _mori_recv_local_routing_kernel(
    indices_ptr,
    num_valid_ptr,
    local_ptr,
    counts_ptr,
    total,
    local_start,
    TOPK: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Map BLOCK flat [rows, TOPK] global ids to local ids (-1 elsewhere) and add their histogram."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < total
    expert = tl.load(indices_ptr + offs, mask=in_range, other=-1).to(tl.int64)
    num_valid = tl.load(num_valid_ptr).to(tl.int64)
    local = expert - local_start
    keep = in_range & (offs // TOPK < num_valid) & (local >= 0) & (local < NUM_LOCAL)
    local = tl.where(keep, local, -1)
    tl.store(local_ptr + offs, local, mask=in_range)
    bins = tl.arange(0, NUM_BINS)
    hist = tl.sum((local[:, None] == bins[None, :]).to(tl.int32), axis=0).to(tl.int64)
    tl.atomic_add(counts_ptr + bins, hist, mask=(bins < NUM_LOCAL) & (hist != 0), sem="relaxed")
