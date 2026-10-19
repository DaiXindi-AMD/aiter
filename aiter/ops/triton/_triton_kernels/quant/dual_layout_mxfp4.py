# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Extend the BF16-source dual layout from PR #6458, a0c59410117168654d6888a63dfe2060d2044477.
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.mxfp4 import (
    finite_v1,
    normalized_hadamard16_v1,
    quantize_groups_v1,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_dual_layout_quant_mxfp4_repr = make_kernel_repr(
    "_dual_layout_quant_mxfp4_kernel",
    [
        "CONFIG_ID",
        "BLOCK_SIZE",
        "HADAMARD_SIZE",
        "USE_SR_ROW",
        "USE_SR_TRANSPOSED",
        "num_warps",
        "num_stages",
    ],
)


@triton.jit(repr=_dual_layout_quant_mxfp4_repr)
def _dual_layout_quant_mxfp4_kernel(
    x_ptr,
    row_ptr,
    row_scale_ptr,
    transposed_ptr,
    transposed_scale_ptr,
    status_ptr,
    M,
    N,
    P,
    row_seed,
    row_base,
    t_seed,
    t_base,
    BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    USE_SR_ROW: tl.constexpr,
    USE_SR_TRANSPOSED: tl.constexpr,
    CONFIG_ID: tl.constexpr,
):
    """Read a source tile once; pack its row and independent padded H16T."""
    tl.static_assert(BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * 32 + tl.arange(0, 32)
    cols = pid_n * 32 + tl.arange(0, 32)
    x = tl.load(
        x_ptr + rows[:, None] * N + cols[None, :], mask=rows[:, None] < M, other=0.0
    ).to(tl.float32)
    row_draw = row_base.to(tl.uint64) + rows[:, None].to(tl.uint64) * N + cols[None, :]
    row_packed, row_scales, row_poison = quantize_groups_v1(
        x, row_draw, row_seed, USE_SR_ROW, 32, 32
    )
    packed_cols = pid_n * 16 + tl.arange(0, 16)
    tl.store(
        row_ptr + rows[:, None] * (N // 2) + packed_cols[None, :],
        row_packed,
        mask=rows[:, None] < M,
    )
    tl.store(
        row_scale_ptr + rows * (N // 32) + pid_n, row_scales.reshape(32), mask=rows < M
    )

    source_poison = tl.max((~finite_v1(x)).to(tl.int32), 0) != 0
    safe = tl.where(finite_v1(x), x, 0.0)
    transposed, overflow16 = normalized_hadamard16_v1(
        tl.trans(safe).reshape(64, 16), 64
    )
    transposed = transposed.reshape(32, 32)
    overflow = tl.max(overflow16.reshape(32, 2).to(tl.int32), 1) != 0
    overflow = overflow & ~source_poison
    poison = source_poison | overflow
    t_draw = t_base.to(tl.uint64) + cols[:, None].to(tl.uint64) * P + rows[None, :]
    t_packed, t_scales, _ = quantize_groups_v1(
        tl.where(poison[:, None], 0.0, transposed),
        t_draw,
        t_seed,
        USE_SR_TRANSPOSED,
        32,
        32,
    )
    t_packed = tl.where(poison[:, None], 0, t_packed).to(tl.uint8)
    t_scales = tl.where(poison[:, None], 255, t_scales).to(tl.uint8)
    packed_rows = pid_m * 16 + tl.arange(0, 16)
    tl.store(transposed_ptr + cols[:, None] * (P // 2) + packed_rows[None, :], t_packed)
    tl.store(transposed_scale_ptr + cols * (P // 32) + pid_m, t_scales.reshape(32))
    input_bad = tl.sum(tl.sum(row_poison.to(tl.int32), 1), 0) != 0
    h_bad = tl.sum(overflow.to(tl.int32), 0) != 0
    flags = input_bad.to(tl.int32) | (h_bad.to(tl.int32) << 1)
    tl.atomic_or(status_ptr, flags, sem="relaxed", scope="gpu")
