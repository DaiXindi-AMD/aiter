# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Variable-K extension of AITER's gfx950 explicit-layout FP4 GEMM."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_repr = make_kernel_repr(
    "_grouped_mxfp4_wgrad_gluon_kernel",
    [
        "M",
        "N",
        "E",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "ACCUMULATE",
        "TRANSPOSE_OUTPUT",
        "ALIGNMENT_HINTS",
        "CONFIG_NUM_WARPS",
        "CONFIG_NUM_STAGES",
        "CONFIG_MATRIX_INSTR_NONKDIM",
        "CACHE_MODIFIER",
    ],
)


@gluon.jit(repr=_repr, do_not_specialize=["payload_stride", "scale_stride"])
def _grouped_mxfp4_wgrad_gluon_kernel(
    dy_payload,
    dy_scales,
    x_payload,
    x_scales,
    out_ptr,
    padded_offsets,
    payload_stride,
    scale_stride,
    M: gl.constexpr,
    N: gl.constexpr,
    E: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ACCUMULATE: gl.constexpr,
    TRANSPOSE_OUTPUT: gl.constexpr,
    ALIGNMENT_HINTS: gl.constexpr,
    CONFIG_NUM_WARPS: gl.constexpr,
    CONFIG_NUM_STAGES: gl.constexpr,
    CONFIG_MATRIX_INSTR_NONKDIM: gl.constexpr,
    CACHE_MODIFIER: gl.constexpr,
):
    gl.static_assert(
        BLOCK_SIZE_M == 256 and BLOCK_SIZE_N == 128 and BLOCK_SIZE_K == 256
    )
    gl.static_assert(CONFIG_NUM_WARPS == 8 and CONFIG_NUM_STAGES == 2)
    gl.static_assert(CONFIG_MATRIX_INSTR_NONKDIM == 32)
    gl.static_assert(
        not TRANSPOSE_OUTPUT, "public wrapper swaps operands for transposed output"
    )
    payload_stride = payload_stride.to(gl.int64)
    scale_stride = scale_stride.to(gl.int64)
    if ALIGNMENT_HINTS:
        payload_stride = gl.multiple_of(payload_stride, BLOCK_SIZE_K // 2)
        scale_stride = gl.multiple_of(scale_stride, BLOCK_SIZE_K // 32)
    tile = gl.program_id(0)
    expert = tile % E
    pm, pn = pid_grid(
        tile // E,
        gl.cdiv(M, BLOCK_SIZE_M),
        gl.cdiv(N, BLOCK_SIZE_N),
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    p0 = gl.load(padded_offsets + expert).to(gl.int64)
    p1 = gl.load(padded_offsets + expert + 1).to(gl.int64)
    if ALIGNMENT_HINTS:
        p0 = gl.multiple_of(p0, BLOCK_SIZE_K)
        p1 = gl.multiple_of(p1, BLOCK_SIZE_K)
    steps = (p1 - p0) // BLOCK_SIZE_K

    # A/B payload layouts reuse the image's 256-tile, eight-wave FP4 GEMM.
    amk: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [8, 1], [1, 0])
    bkn: gl.constexpr = gl.BlockedLayout([16, 1], [8, 8], [1, 8], [0, 1])
    scales_layout: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [8, 1], [1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[2, 4],
    )
    da_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mma, k_width=16
    )
    db_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mma, k_width=16
    )
    as_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        da_layout, [BLOCK_SIZE_M, BLOCK_SIZE_K // 32], scale_factor=32
    )
    bs_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        db_layout, [BLOCK_SIZE_N, BLOCK_SIZE_K // 32], scale_factor=32
    )
    store_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
        warp_bases=[[0, 32], [0, 64], [32, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
    shared_s: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0, 1]
    )
    r = pm * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, amk))
    c = pn * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, bkn))
    ka = gl.arange(0, 128, layout=gl.SliceLayout(0, amk))
    kb = gl.arange(0, 128, layout=gl.SliceLayout(1, bkn))
    sr = pm * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, scales_layout)
    )
    sc = pn * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(1, scales_layout)
    )
    ks = gl.arange(0, 8, layout=gl.SliceLayout(0, scales_layout))
    # Gluon 3.7 buffer ops require 32-bit offsets; the wrapper rejects spans >= 4 GiB.
    ao = (r[:, None].to(gl.int64) * payload_stride + p0 // 2 + ka[None, :]).to(
        gl.uint32
    )
    bo = (c[None, :].to(gl.int64) * payload_stride + p0 // 2 + kb[:, None]).to(
        gl.uint32
    )
    aso = (sr[:, None].to(gl.int64) * scale_stride + p0 // 32 + ks[None, :]).to(
        gl.uint32
    )
    bso = (sc[:, None].to(gl.int64) * scale_stride + p0 // 32 + ks[None, :]).to(
        gl.uint32
    )
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), gl.float32, layout=mma)
    if steps > 0:
        sm_a = gl.allocate_shared_memory(
            gl.uint8, [BLOCK_SIZE_M, BLOCK_SIZE_K // 2], layout=shared_a
        )
        sm_b = gl.allocate_shared_memory(
            gl.uint8, [BLOCK_SIZE_K // 2, BLOCK_SIZE_N], layout=shared_b
        )
        sm_as = gl.allocate_shared_memory(
            gl.uint8, [BLOCK_SIZE_M, BLOCK_SIZE_K // 32], layout=shared_s
        )
        sm_bs = gl.allocate_shared_memory(
            gl.uint8, [BLOCK_SIZE_N, BLOCK_SIZE_K // 32], layout=shared_s
        )
        a = gl.amd.cdna4.buffer_load(dy_payload, ao, mask=r[:, None] < M, other=0)
        sa = gl.amd.cdna4.buffer_load(dy_scales, aso, mask=sr[:, None] < M, other=127)
        b = gl.amd.cdna4.buffer_load(
            ptr=x_payload,
            offsets=bo,
            mask=c[None, :] < N,
            other=0,
            cache=CACHE_MODIFIER,
        )
        sb = gl.amd.cdna4.buffer_load(
            ptr=x_scales,
            offsets=bso,
            mask=sc[:, None] < N,
            other=127,
            cache=CACHE_MODIFIER,
        )
        sm_a.store(a)
        sm_as.store(sa)
        for step in range(steps - 1):
            advance = (step + 1).to(gl.uint32) * 128
            scale_advance = (step + 1).to(gl.uint32) * 8
            next_a = gl.amd.cdna4.buffer_load(
                dy_payload, ao + advance, mask=r[:, None] < M, other=0
            )
            sm_b.store(b)
            sm_bs.store(sb)
            ca, csa = sm_a.load(layout=da_layout), sm_as.load(layout=as_layout)
            next_sa = gl.amd.cdna4.buffer_load(
                dy_scales, aso + scale_advance, mask=sr[:, None] < M, other=127
            )
            csb = sm_bs.load(layout=bs_layout)
            next_b = gl.amd.cdna4.buffer_load(
                ptr=x_payload,
                offsets=bo + advance,
                mask=c[None, :] < N,
                other=0,
                cache=CACHE_MODIFIER,
            )
            next_sb = gl.amd.cdna4.buffer_load(
                ptr=x_scales,
                offsets=bso + scale_advance,
                mask=sc[:, None] < N,
                other=127,
                cache=CACHE_MODIFIER,
            )
            cb = sm_b.load(layout=db_layout)
            acc = gl.amd.cdna4.mfma_scaled(
                a=ca,
                a_scale=csa,
                a_format="e2m1",
                b=cb,
                b_scale=csb,
                b_format="e2m1",
                acc=acc,
            )
            sm_a.store(next_a)
            sm_as.store(next_sa)
            b, sb = next_b, next_sb
        sm_b.store(b)
        sm_bs.store(sb)
        acc = gl.amd.cdna4.mfma_scaled(
            a=sm_a.load(layout=da_layout),
            a_scale=sm_as.load(layout=as_layout),
            a_format="e2m1",
            b=sm_b.load(layout=db_layout),
            b_scale=sm_bs.load(layout=bs_layout),
            b_format="e2m1",
            acc=acc,
        )
    rr = pm * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, store_layout)
    )
    cc = pn * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, store_layout)
    )
    mask = (rr[:, None] < M) & (cc[None, :] < N)
    dst = expert.to(gl.int64) * (M * N) + rr[:, None].to(gl.int64) * N + cc[None, :]
    dst = dst.to(gl.uint32)
    if ACCUMULATE:
        if steps > 0:
            value = gl.convert_layout(acc, layout=store_layout, assert_trivial=False)
            prev = gl.amd.cdna4.buffer_load(out_ptr, dst, mask=mask, other=0.0)
            gl.amd.cdna4.buffer_store(value + prev, out_ptr, dst, mask)
    else:
        value = gl.convert_layout(acc, layout=store_layout, assert_trivial=False)
        gl.amd.cdna4.buffer_store(value, out_ptr, dst, mask)
