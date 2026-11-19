# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Segmented transpose and quantization with independent token-axis scales."""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_colwise_repr = make_kernel_repr(
    "_grouped_mxfp4_colwise_kernel",
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
    repr=_colwise_repr,
    do_not_specialize=["payload_stride", "scale_stride"],
)
def _grouped_mxfp4_colwise_kernel(
    input_ptr,
    payload_ptr,
    scale_ptr,
    real_offsets,
    padded_offsets,
    prep_blocks,
    payload_stride,
    scale_stride,
    F: tl.constexpr,
    PAD_MULTIPLE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
    CONFIG_NUM_WARPS: tl.constexpr,
    CONFIG_NUM_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_T % 32 == 0)
    tl.static_assert(PAD_MULTIPLE % BLOCK_SIZE_T == 0)
    block = tl.program_id(0)
    feature = tl.program_id(1) * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
    expert = tl.load(prep_blocks + block * 2).to(tl.int32)
    local = tl.load(prep_blocks + block * 2 + 1).to(tl.int64) * PAD_MULTIPLE
    start = tl.load(real_offsets + expert).to(tl.int64)
    end = tl.load(real_offsets + expert + 1).to(tl.int64)
    padded = tl.load(padded_offsets + expert).to(tl.int64)
    for part in tl.static_range(PAD_MULTIPLE // BLOCK_SIZE_T):
        token = start + local + part * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
        values = tl.load(
            input_ptr + token[:, None] * F + feature[None, :],
            mask=(token[:, None] < end) & (feature[None, :] < F),
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        # Reuse the runtime's exact E8M0/E2M1 recipe on each columnwise 1x32 block.
        payload, scales = _mxfp4_quant_op(
            tl.trans(values), BLOCK_SIZE_T, BLOCK_SIZE_F, 32
        )
        pb = (padded + local + part * BLOCK_SIZE_T) // 2 + tl.arange(
            0, BLOCK_SIZE_T // 2
        )
        sb = (padded + local + part * BLOCK_SIZE_T) // 32 + tl.arange(
            0, BLOCK_SIZE_T // 32
        )
        tl.store(
            payload_ptr + feature[:, None].to(tl.int64) * payload_stride + pb[None, :],
            payload,
            mask=feature[:, None] < F,
        )
        tl.store(
            scale_ptr + feature[:, None].to(tl.int64) * scale_stride + sb[None, :],
            scales,
            mask=feature[:, None] < F,
        )
