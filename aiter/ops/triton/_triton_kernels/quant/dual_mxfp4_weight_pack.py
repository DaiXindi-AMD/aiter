# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-read row- and column-axis MXFP4 pack of a dense batched weight."""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_dual_weight_pack_repr = make_kernel_repr(
    "_dual_mxfp4_weight_pack_kernel",
    ["M", "N", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "USE_ASM"],
)


@triton.jit
def _saturate_positive_inf(values, payload, ROWS: tl.constexpr, COLS: tl.constexpr):
    """Encode +inf as +6 (0x7) like ``dynamic_mxfp4_quant``; the shared pack op flips its sign."""
    is_inf = (values == float("inf")).to(tl.int8).reshape(ROWS, COLS // 2, 2)
    even, odd = tl.split(is_inf)
    payload = tl.where(even != 0, (payload & 0xF0) | 0x07, payload)
    return tl.where(odd != 0, (payload & 0x0F) | 0x70, payload)


@triton.jit(repr=_dual_weight_pack_repr)
def _dual_mxfp4_weight_pack_kernel(
    w_ptr,
    row_payload_ptr,
    row_scale_ptr,
    col_payload_ptr,
    col_scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    USE_ASM: tl.constexpr = False,
):
    """Quantize one [BLOCK_SIZE_M, BLOCK_SIZE_N] tile of contiguous [E, M, N] along N and M.

    Row outputs are [E, M, N/2] payload and [E, M, N/32] scales; column outputs
    are [E, N, M/2] and [E, N, M/32]. Each axis gets its own 1x32 E8M0 scales,
    so no BF16 transpose is materialized. M and N are multiples of the tile.
    """
    tl.static_assert(BLOCK_SIZE_M % 32 == 0)
    tl.static_assert(BLOCK_SIZE_N % 32 == 0)
    tl.static_assert(M % BLOCK_SIZE_M == 0)
    tl.static_assert(N % BLOCK_SIZE_N == 0)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    expert = tl.program_id(2).to(tl.int64)

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    values = tl.load(
        w_ptr + expert * (M * N) + rows[:, None] * N + cols[None, :],
        cache_modifier=".cg",
    ).to(tl.float32)

    row_payload, row_scales = _mxfp4_quant_op(
        values, BLOCK_SIZE_N, BLOCK_SIZE_M, 32, USE_ASM=USE_ASM
    )
    has_inf = tl.max((values == float("inf")).to(tl.int32)) != 0
    if has_inf:
        row_payload = _saturate_positive_inf(
            values, row_payload, BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    row_pb = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    row_sb = pid_n * (BLOCK_SIZE_N // 32) + tl.arange(0, BLOCK_SIZE_N // 32)
    tl.store(
        row_payload_ptr
        + expert * (M * N // 2)
        + rows[:, None] * (N // 2)
        + row_pb[None, :],
        row_payload,
    )
    tl.store(
        row_scale_ptr
        + expert * (M * N // 32)
        + rows[:, None] * (N // 32)
        + row_sb[None, :],
        row_scales,
    )

    transposed = tl.trans(values)
    col_payload, col_scales = _mxfp4_quant_op(
        transposed, BLOCK_SIZE_M, BLOCK_SIZE_N, 32, USE_ASM=USE_ASM
    )
    if has_inf:
        col_payload = _saturate_positive_inf(
            transposed, col_payload, BLOCK_SIZE_N, BLOCK_SIZE_M
        )
    col_pb = pid_m * (BLOCK_SIZE_M // 2) + tl.arange(0, BLOCK_SIZE_M // 2)
    col_sb = pid_m * (BLOCK_SIZE_M // 32) + tl.arange(0, BLOCK_SIZE_M // 32)
    tl.store(
        col_payload_ptr
        + expert * (M * N // 2)
        + cols[:, None] * (M // 2)
        + col_pb[None, :],
        col_payload,
    )
    tl.store(
        col_scale_ptr
        + expert * (M * N // 32)
        + cols[:, None] * (M // 32)
        + col_sb[None, :],
        col_scales,
    )
