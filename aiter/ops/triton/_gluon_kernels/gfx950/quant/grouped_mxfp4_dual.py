# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""LDS-staged grouped row/column MXFP4 producer for gfx950."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.quant.quant import _mxfp4_quant_op
from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    remap_xcd_chunked,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_dual_v3_repr = make_kernel_repr(
    "_grouped_mxfp4_dual_v3_gluon_kernel",
    [
        "F",
        "PAD_MULTIPLE",
        "BLOCK_SIZE_T",
        "BLOCK_SIZE_F",
        "GRID_F",
        "GRID_T",
        "GRID_SIZE",
        "NUM_XCDS",
        "XCD_CHUNK",
        "CONFIG_NUM_WARPS",
    ],
)


@gluon.jit(repr=_dual_v3_repr)
def _grouped_mxfp4_dual_v3_gluon_kernel(
    input_ptr,
    row_payload_ptr,
    row_scale_ptr,
    col_payload_ptr,
    col_scale_ptr,
    compact_blocks,
    row_payload_stride,
    row_scale_stride,
    col_payload_stride,
    col_scale_stride,
    F: gl.constexpr,
    PAD_MULTIPLE: gl.constexpr,
    BLOCK_SIZE_T: gl.constexpr,
    BLOCK_SIZE_F: gl.constexpr,
    GRID_F: gl.constexpr,
    GRID_T: gl.constexpr,
    GRID_SIZE: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    XCD_CHUNK: gl.constexpr,
    CONFIG_NUM_WARPS: gl.constexpr,
):
    gl.static_assert(BLOCK_SIZE_T == 64)
    gl.static_assert(BLOCK_SIZE_F == 256)
    gl.static_assert(PAD_MULTIPLE == 256)
    gl.static_assert(CONFIG_NUM_WARPS == 8)

    pid = remap_xcd_chunked(gl.program_id(0), GRID_SIZE, NUM_XCDS, XCD_CHUNK)
    block = pid // (GRID_T * GRID_F)
    tile_id = pid % (GRID_T * GRID_F)
    part = tile_id // GRID_F
    feature_base = (tile_id % GRID_F) * BLOCK_SIZE_F
    start = gl.load(compact_blocks + block * 3).to(gl.int64) + part * BLOCK_SIZE_T
    end = gl.load(compact_blocks + block * 3 + 1).to(gl.int64)
    padded = gl.load(compact_blocks + block * 3 + 2).to(gl.int64) + part * BLOCK_SIZE_T

    # Eight row waves load and write 128-bit contiguous vectors.
    row_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 32],
        threads_per_warp=[8, 8],
        warps_per_cta=[8, 1],
        order=[1, 0],
    )
    # In phase two, each wave owns 32 feature rows and reads 32 tokens per lane.
    col_source_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[32, 1],
        threads_per_warp=[2, 32],
        warps_per_cta=[1, 8],
        order=[0, 1],
    )
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )

    local_t = gl.arange(0, BLOCK_SIZE_T, layout=gl.SliceLayout(1, row_layout))
    local_f = gl.arange(0, BLOCK_SIZE_F, layout=gl.SliceLayout(0, row_layout))
    token = start + local_t
    feature = feature_base + local_f
    input_offsets = (token[:, None] * F + feature[None, :]).to(gl.uint32)
    valid = (token[:, None] < end) & (feature[None, :] < F)
    values = gl.amd.cdna4.buffer_load(
        input_ptr, input_offsets, mask=valid, other=0.0, cache=".cg"
    )

    tile = gl.allocate_shared_memory(
        gl.bfloat16,
        [BLOCK_SIZE_T, BLOCK_SIZE_F],
        layout=shared_layout,
    )
    tile.store(values)

    row_payload, row_scales = _mxfp4_quant_op(values, BLOCK_SIZE_F, BLOCK_SIZE_T, 32)
    row_t = gl.arange(0, BLOCK_SIZE_T)
    row_p = gl.arange(0, BLOCK_SIZE_F // 2)
    row_s = gl.arange(0, BLOCK_SIZE_F // 32)
    row_payload_offsets = (
        (start + row_t[:, None]).to(gl.int64) * row_payload_stride
        + feature_base // 2
        + row_p[None, :]
    ).to(gl.uint32)
    row_scale_offsets = (
        (start + row_t[:, None]).to(gl.int64) * row_scale_stride
        + feature_base // 32
        + row_s[None, :]
    ).to(gl.uint32)
    row_mask = start + row_t < end
    gl.amd.cdna4.buffer_store(
        row_payload,
        row_payload_ptr,
        row_payload_offsets,
        mask=row_mask[:, None],
    )
    gl.amd.cdna4.buffer_store(
        row_scales,
        row_scale_ptr,
        row_scale_offsets,
        mask=row_mask[:, None],
    )

    # The LDS round trip ends the row live range before column quantization.
    col_values = gl.permute(tile.load(layout=col_source_layout), 1, 0)
    col_payload, col_scales = _mxfp4_quant_op(
        col_values, BLOCK_SIZE_T, BLOCK_SIZE_F, 32
    )
    col_f = gl.arange(0, BLOCK_SIZE_F)
    col_p = gl.arange(0, BLOCK_SIZE_T // 2)
    col_s = gl.arange(0, BLOCK_SIZE_T // 32)
    col_payload_offsets = (
        (feature_base + col_f[:, None]).to(gl.int64) * col_payload_stride
        + padded // 2
        + col_p[None, :]
    ).to(gl.uint32)
    col_scale_offsets = (
        (feature_base + col_f[:, None]).to(gl.int64) * col_scale_stride
        + padded // 32
        + col_s[None, :]
    ).to(gl.uint32)
    col_mask = feature_base + col_f < F
    gl.amd.cdna4.buffer_store(
        col_payload,
        col_payload_ptr,
        col_payload_offsets,
        mask=col_mask[:, None],
    )
    gl.amd.cdna4.buffer_store(
        col_scales,
        col_scale_ptr,
        col_scale_offsets,
        mask=col_mask[:, None],
    )
