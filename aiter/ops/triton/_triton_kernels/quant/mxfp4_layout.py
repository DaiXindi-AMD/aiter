# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Private packed-MXFP4 layout kernels.

These permutations match the layouts consumed by AITER's gfx950 MXFP4 GEMMs.
They live separately from quantization so layout-only operations do not need to
carry the quantizer's rounding machinery.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


MXFP4_SCALE_STRIPE = 32
MXFP4_SCALE_KCHUNK = 8
MXFP4_SCALE_STRIPE_C = tl.constexpr(MXFP4_SCALE_STRIPE)
MXFP4_SCALE_KCHUNK_C = tl.constexpr(MXFP4_SCALE_KCHUNK)

MXFP4_SHUFFLE_TILE_ROWS = 16
MXFP4_SHUFFLE_UNIT_BYTES = 8
MXFP4_SHUFFLE_UNITS_PER_GROUP = 4
MXFP4_SHUFFLE_GROUP_BYTES = (
    MXFP4_SHUFFLE_UNIT_BYTES * MXFP4_SHUFFLE_UNITS_PER_GROUP
)
MXFP4_SHUFFLE_TILE_ROWS_C = tl.constexpr(MXFP4_SHUFFLE_TILE_ROWS)
MXFP4_SHUFFLE_UNIT_BYTES_C = tl.constexpr(MXFP4_SHUFFLE_UNIT_BYTES)


@triton.jit
def _shuffled_fp4_offsets(
    rows,
    byte_cols,
    num_byte_cols,
    TILE_ROWS: tl.constexpr,
    UNIT: tl.constexpr,
):
    """Flat byte offsets for AITER's ``layout=(16, 16)`` B operand."""
    UNITS_PER_GROUP: tl.constexpr = 4
    i = rows // TILE_ROWS
    r = rows % TILE_ROWS
    unit = byte_cols // UNIT
    within_unit = byte_cols % UNIT
    j = unit // UNITS_PER_GROUP
    rest = unit % UNITS_PER_GROUP
    p = rest // 2
    q = rest % 2
    groups = num_byte_cols // (UNIT * UNITS_PER_GROUP)
    dst_unit = ((i * groups + j) * 2 + p) * (TILE_ROWS * 2) + r * 2 + q
    return dst_unit * UNIT + within_unit


@triton.jit
def _swizzled_scale_offsets(
    rows,
    cols,
    num_cols,
    STRIPE: tl.constexpr,
    KCHUNK: tl.constexpr,
):
    """Flat offsets for gfx950's 32-row by 8-K-chunk scale layout."""
    s = rows // STRIPE
    a = (rows % STRIPE) // (STRIPE // 2)
    b = rows % (STRIPE // 2)
    k = cols // KCHUNK
    c = (cols % KCHUNK) // (KCHUNK // 2)
    d = cols % (KCHUNK // 2)
    within = (k * (KCHUNK // 2) + d) * (STRIPE * 2) + b * 4 + c * 2 + a
    return s * (num_cols * STRIPE) + within


_transpose_packed_fp4_kernel_repr = make_kernel_repr(
    "_transpose_packed_fp4_kernel",
    [
        "N_packed",
        "BLOCK_M",
        "BLOCK_N_PACKED",
        "SHUFFLE_DATA",
        "NUM_PACKED_COLS",
        "IN_SHUFFLED",
    ],
)


@triton.jit(repr=_transpose_packed_fp4_kernel_repr)
def _transpose_packed_fp4_kernel(
    in_ptr,
    out_ptr,
    M,
    N_packed: tl.constexpr,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N_PACKED: tl.constexpr,
    SHUFFLE_DATA: tl.constexpr = False,
    NUM_PACKED_COLS: tl.constexpr = 0,
    IN_SHUFFLED: tl.constexpr = False,
):
    """Transpose packed FP4 ``(M, N/2) -> (N, M/2)``."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    N = N_packed * 2
    BLOCK_N: tl.constexpr = BLOCK_N_PACKED * 2
    BLOCK_M_HALF: tl.constexpr = BLOCK_M // 2

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_packed = pid_n * BLOCK_N_PACKED + tl.arange(0, BLOCK_N_PACKED)
    mask = (rm[:, None] < M) & (rn_packed[None, :] < N_packed)

    if IN_SHUFFLED:
        in_offs = _shuffled_fp4_offsets(
            rm[:, None],
            rn_packed[None, :],
            N_packed,
            TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS_C,
            UNIT=MXFP4_SHUFFLE_UNIT_BYTES_C,
        )
    else:
        in_offs = rm[:, None] * stride_im + rn_packed[None, :] * stride_in
    packed = tl.load(in_ptr + in_offs, mask=mask, other=0).to(tl.uint8)

    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    unpacked = tl.reshape(tl.join(even, odd), (BLOCK_M, BLOCK_N))
    transposed = tl.trans(unpacked)
    reshaped = tl.reshape(transposed, (BLOCK_N, BLOCK_M_HALF, 2))
    t_even, t_odd = tl.split(reshaped)
    repacked = t_even | (t_odd << 4)

    rn_full = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_packed = pid_m * BLOCK_M_HALF + tl.arange(0, BLOCK_M_HALF)
    out_mask = (rn_full[:, None] < N) & (rm_packed[None, :] < (M // 2))
    if SHUFFLE_DATA:
        tl.store(
            out_ptr
            + _shuffled_fp4_offsets(
                rn_full[:, None],
                rm_packed[None, :],
                NUM_PACKED_COLS,
                TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS_C,
                UNIT=MXFP4_SHUFFLE_UNIT_BYTES_C,
            ),
            repacked,
            mask=out_mask,
        )
    else:
        tl.store(
            out_ptr
            + rn_full[:, None] * stride_om
            + rm_packed[None, :] * stride_on,
            repacked,
            mask=out_mask,
        )


_swizzle_mxfp4_scale_gfx950_kernel_repr = make_kernel_repr(
    "_swizzle_mxfp4_scale_gfx950_kernel",
    ["STRIPE", "KCHUNK", "BLOCK_K"],
)


@triton.jit(repr=_swizzle_mxfp4_scale_gfx950_kernel_repr)
def _swizzle_mxfp4_scale_gfx950_kernel(
    src_ptr,
    dst_ptr,
    cols,
    stride_sm,
    STRIPE: tl.constexpr,
    KCHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """E8M0 scales ``(rows, cols)`` to gfx950's GEMM scale layout."""
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)

    S: tl.constexpr = STRIPE
    KW: tl.constexpr = KCHUNK
    TILE_C: tl.constexpr = BLOCK_K * KW

    offs_r = tl.arange(0, S)
    offs_c = pid_k * TILE_C + tl.arange(0, TILE_C)
    x = tl.load(
        src_ptr + (pid_s * S + offs_r)[:, None] * stride_sm + offs_c[None, :],
        mask=offs_c[None, :] < cols,
        other=0,
    )
    x = tl.reshape(x, (2, S // 2, BLOCK_K, 2, KW // 2))
    x = tl.permute(x, (2, 4, 1, 3, 0))
    x = tl.reshape(x, (TILE_C * S,))

    offs_o = pid_k * (TILE_C * S) + tl.arange(0, TILE_C * S)
    tl.store(
        dst_ptr + pid_s * (cols * S) + offs_o,
        x,
        mask=offs_o < cols * S,
    )


_swizzle_expanded_2d_scale_kernel_repr = make_kernel_repr(
    "_swizzle_expanded_2d_scale_kernel",
    ["QUANT_BLOCK_SIZE", "STRIPE", "KCHUNK", "BLOCK_K"],
)


@triton.jit(repr=_swizzle_expanded_2d_scale_kernel_repr)
def _swizzle_expanded_2d_scale_kernel(
    src_ptr,
    dst_ptr,
    cols,
    stride_tile_row,
    stride_tile_col,
    QUANT_BLOCK_SIZE: tl.constexpr,
    STRIPE: tl.constexpr,
    KCHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Expand block-row scales while storing them in gfx950 GEMM order."""
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)

    S: tl.constexpr = STRIPE
    TILE_C: tl.constexpr = BLOCK_K * KCHUNK

    rows = pid_s * S + tl.arange(0, S)
    offs_c = pid_k * TILE_C + tl.arange(0, TILE_C)
    mask = offs_c[None, :] < cols
    x = tl.load(
        src_ptr
        + (rows // QUANT_BLOCK_SIZE)[:, None] * stride_tile_row
        + offs_c[None, :] * stride_tile_col,
        mask=mask,
        other=0,
    )
    tl.store(
        dst_ptr
        + _swizzled_scale_offsets(
            rows[:, None],
            offs_c[None, :],
            cols,
            STRIPE=STRIPE,
            KCHUNK=KCHUNK,
        ),
        x,
        mask=mask,
    )
