# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

MXFP4_SHUFFLE_TILE_ROWS = 16
MXFP4_SHUFFLE_UNIT_BYTES = 8
MXFP4_SHUFFLE_GROUP_BYTES = 4 * MXFP4_SHUFFLE_UNIT_BYTES
MXFP4_SCALE_STRIPE = 32
MXFP4_SCALE_KCHUNK = 8


def mxfp4_data_shuffle_supported(rows: int, packed_cols: int) -> bool:
    """Return whether packed FP4 data tiles evenly in AITER's 16x16 layout."""
    return (
        rows % MXFP4_SHUFFLE_TILE_ROWS == 0
        and packed_cols % MXFP4_SHUFFLE_GROUP_BYTES == 0
    )


def mxfp4_scale_swizzle_supported(rows: int, cols: int) -> bool:
    """Return whether a scale matrix evenly tiles for the gfx950 layout."""
    return rows % MXFP4_SCALE_STRIPE == 0 and cols % MXFP4_SCALE_KCHUNK == 0


@triton.jit
def _mxfp4_shuffled_offsets(
    rows,
    byte_cols,
    num_byte_cols,
    TILE_ROWS: tl.constexpr,
    UNIT_BYTES: tl.constexpr,
):
    """Map row-major packed-FP4 coordinates to AITER's 16x16 B layout."""
    UNITS_PER_GROUP: tl.constexpr = 4
    tile_row = rows // TILE_ROWS
    row = rows % TILE_ROWS
    unit = byte_cols // UNIT_BYTES
    byte = byte_cols % UNIT_BYTES
    tile_col = unit // UNITS_PER_GROUP
    unit_in_group = unit % UNITS_PER_GROUP
    pair = unit_in_group // 2
    half = unit_in_group % 2
    groups = num_byte_cols // (UNIT_BYTES * UNITS_PER_GROUP)
    dst_unit = (
        ((tile_row * groups + tile_col) * 2 + pair) * (TILE_ROWS * 2) + row * 2 + half
    )
    return dst_unit * UNIT_BYTES + byte


@triton.jit
def _mxfp4_swizzled_scale_offsets(
    rows,
    cols,
    num_cols,
    STRIPE: tl.constexpr,
    KCHUNK: tl.constexpr,
):
    """Map logical E8M0 coordinates to gfx950's 32x8 scale layout."""
    stripe = rows // STRIPE
    row_half = (rows % STRIPE) // (STRIPE // 2)
    row_lane = rows % (STRIPE // 2)
    chunk = cols // KCHUNK
    col_half = (cols % KCHUNK) // (KCHUNK // 2)
    col_lane = cols % (KCHUNK // 2)
    within = (
        (chunk * (KCHUNK // 2) + col_lane) * (STRIPE * 2)
        + row_lane * 4
        + col_half * 2
        + row_half
    )
    return stripe * (num_cols * STRIPE) + within
