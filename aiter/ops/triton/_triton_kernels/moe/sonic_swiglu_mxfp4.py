# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-read concat-layout SwiGLU and grouped MXFP4 producer."""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    remap_xcd_chunked,
)
from aiter.ops.triton._triton_kernels.moe.activations import silu
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_producer_repr = make_kernel_repr(
    "_sonic_swiglu_mxfp4_kernel",
    [
        "F",
        "PAD_MULTIPLE",
        "BLOCK_SIZE_T",
        "BLOCK_SIZE_F",
        "CONFIG_NUM_WARPS",
        "CONFIG_NUM_STAGES",
        "GRID_F",
        "GRID_SIZE",
        "NUM_XCDS",
        "XCD_CHUNK",
        "USE_ASM",
    ],
)


@triton.jit(
    repr=_producer_repr,
    do_not_specialize=[
        "a_stride",
        "row_payload_stride",
        "row_scale_stride",
        "col_payload_stride",
        "col_scale_stride",
    ],
)
def _sonic_swiglu_mxfp4_kernel(
    h1_ptr,
    a_ptr,
    row_payload_ptr,
    row_scale_ptr,
    col_payload_ptr,
    col_scale_ptr,
    compact_blocks,
    a_stride,
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
    GRID_F: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_T % 32 == 0)
    tl.static_assert(BLOCK_SIZE_F % 32 == 0)
    tl.static_assert(PAD_MULTIPLE % BLOCK_SIZE_T == 0)
    pid = remap_xcd_chunked(tl.program_id(0), GRID_SIZE, NUM_XCDS, XCD_CHUNK)
    block = pid // GRID_F
    feature_base = (pid % GRID_F) * BLOCK_SIZE_F
    feature = feature_base + tl.arange(0, BLOCK_SIZE_F)
    start = tl.load(compact_blocks + block * 3).to(tl.int64)
    end = tl.load(compact_blocks + block * 3 + 1).to(tl.int64)
    padded = tl.load(compact_blocks + block * 3 + 2).to(tl.int64)

    for part in tl.static_range(PAD_MULTIPLE // BLOCK_SIZE_T):
        token = start + part * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
        valid = (token[:, None] < end) & (feature[None, :] < F)
        gate = tl.load(
            h1_ptr + token[:, None] * (2 * F) + feature[None, :],
            mask=valid,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        up = tl.load(
            h1_ptr + token[:, None] * (2 * F) + F + feature[None, :],
            mask=valid,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        activation = (silu(gate) * up).to(a_ptr.dtype.element_ty)
        tl.store(
            a_ptr + token[:, None] * a_stride + feature[None, :],
            activation,
            mask=valid,
        )

        values = activation.to(tl.float32)
        row_payload, row_scales = _mxfp4_quant_op(
            values, BLOCK_SIZE_F, BLOCK_SIZE_T, 32, USE_ASM=USE_ASM
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
            tl.trans(values),
            BLOCK_SIZE_T,
            BLOCK_SIZE_F,
            32,
            USE_ASM=USE_ASM,
        )
        col_pb = (padded + part * BLOCK_SIZE_T) // 2 + tl.arange(0, BLOCK_SIZE_T // 2)
        col_sb = (padded + part * BLOCK_SIZE_T) // 32 + tl.arange(0, BLOCK_SIZE_T // 32)
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
