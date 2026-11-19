# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-read grouped row/column MXFP4 activation producer."""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_dual_repr = make_kernel_repr(
    "_grouped_mxfp4_dual_kernel",
    [
        "F",
        "PAD_MULTIPLE",
        "BLOCK_SIZE_T",
        "BLOCK_SIZE_F",
        "CONFIG_NUM_WARPS",
        "CONFIG_NUM_STAGES",
    ],
)


@triton.jit(
    repr=_dual_repr,
    do_not_specialize=[
        "row_payload_stride",
        "row_scale_stride",
        "col_payload_stride",
        "col_scale_stride",
    ],
)
def _grouped_mxfp4_dual_kernel(
    input_ptr,
    row_payload_ptr,
    row_scale_ptr,
    col_payload_ptr,
    col_scale_ptr,
    real_offsets,
    padded_offsets,
    prep_blocks,
    row_payload_stride,
    row_scale_stride,
    col_payload_stride,
    col_scale_stride,
    F: tl.constexpr,
    PAD_MULTIPLE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
    CONFIG_NUM_WARPS: tl.constexpr,
    CONFIG_NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_T % 32 == 0)
    tl.static_assert(BLOCK_SIZE_F % 32 == 0)
    tl.static_assert(PAD_MULTIPLE % BLOCK_SIZE_T == 0)
    block = tl.program_id(0)
    feature_base = tl.program_id(1) * BLOCK_SIZE_F
    feature = feature_base + tl.arange(0, BLOCK_SIZE_F)
    expert = tl.load(prep_blocks + block * 2).to(tl.int32)
    local = tl.load(prep_blocks + block * 2 + 1).to(tl.int64) * PAD_MULTIPLE
    start = tl.load(real_offsets + expert).to(tl.int64)
    end = tl.load(real_offsets + expert + 1).to(tl.int64)
    padded = tl.load(padded_offsets + expert).to(tl.int64)
    for part in tl.static_range(PAD_MULTIPLE // BLOCK_SIZE_T):
        token = start + local + part * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
        valid = (token[:, None] < end) & (feature[None, :] < F)
        values = tl.load(
            input_ptr + token[:, None] * F + feature[None, :],
            mask=valid,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

        row_payload, row_scales = _mxfp4_quant_op(
            values, BLOCK_SIZE_F, BLOCK_SIZE_T, 32
        )
        row_pb = feature_base // 2 + tl.arange(0, BLOCK_SIZE_F // 2)
        row_sb = feature_base // 32 + tl.arange(0, BLOCK_SIZE_F // 32)
        tl.store(
            row_payload_ptr + token[:, None] * row_payload_stride + row_pb[None, :],
            row_payload,
            mask=(token[:, None] < end) & (row_pb[None, :] < F // 2),
        )
        tl.store(
            row_scale_ptr + token[:, None] * row_scale_stride + row_sb[None, :],
            row_scales,
            mask=(token[:, None] < end) & (row_sb[None, :] < F // 32),
        )

        col_payload, col_scales = _mxfp4_quant_op(
            tl.trans(values), BLOCK_SIZE_T, BLOCK_SIZE_F, 32
        )
        col_pb = (padded + local + part * BLOCK_SIZE_T) // 2 + tl.arange(
            0, BLOCK_SIZE_T // 2
        )
        col_sb = (padded + local + part * BLOCK_SIZE_T) // 32 + tl.arange(
            0, BLOCK_SIZE_T // 32
        )
        tl.store(
            col_payload_ptr
            + feature[:, None].to(tl.int64) * col_payload_stride
            + col_pb[None, :],
            col_payload,
            mask=feature[:, None] < F,
        )
        tl.store(
            col_scale_ptr
            + feature[:, None].to(tl.int64) * col_scale_stride
            + col_sb[None, :],
            col_scales,
            mask=feature[:, None] < F,
        )
