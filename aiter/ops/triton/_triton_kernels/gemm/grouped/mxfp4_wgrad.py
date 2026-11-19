# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Variable-K MXFP4 GEMM with one writer per output tile."""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_wgrad_repr = make_kernel_repr(
    "_grouped_mxfp4_wgrad_kernel",
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
        "PERSISTENT",
        "ALIGNMENT_HINTS",
        "CONFIG_NUM_WARPS",
        "CONFIG_NUM_STAGES",
        "CONFIG_WAVES_PER_EU",
        "CONFIG_MATRIX_INSTR_NONKDIM",
        "CACHE_MODIFIER",
    ],
)


@triton.jit(
    repr=_wgrad_repr,
    do_not_specialize=["payload_stride", "scale_stride", "num_programs"],
)
def _grouped_mxfp4_wgrad_kernel(
    dy_payload,
    dy_scales,
    x_payload,
    x_scales,
    out_ptr,
    padded_offsets,
    payload_stride,
    scale_stride,
    num_programs,
    M: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    TRANSPOSE_OUTPUT: tl.constexpr,
    PERSISTENT: tl.constexpr,
    ALIGNMENT_HINTS: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
    CONFIG_NUM_WARPS: tl.constexpr,
    CONFIG_NUM_STAGES: tl.constexpr,
    CONFIG_WAVES_PER_EU: tl.constexpr,
    CONFIG_MATRIX_INSTR_NONKDIM: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K % 32 == 0)
    if ALIGNMENT_HINTS:
        # The producer pads every expert to BLOCK_SIZE_K; token strides stay runtime.
        payload_stride = tl.multiple_of(payload_stride, BLOCK_SIZE_K // 2)
        scale_stride = tl.multiple_of(scale_stride, BLOCK_SIZE_K // 32)
    num_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M)
    num_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
    total: tl.constexpr = E * num_m * num_n
    tile = tl.program_id(0)
    # Interleave experts within each output tile to mix short and long K loops.
    while tile < total:
        expert = tile % E
        local_tile = tile // E
        pm, pn = pid_grid(local_tile, num_m, num_n, GROUP_SIZE_M=GROUP_SIZE_M)
        p0 = tl.load(padded_offsets + expert).to(tl.int64)
        p1 = tl.load(padded_offsets + expert + 1).to(tl.int64)
        if ALIGNMENT_HINTS:
            p0 = tl.multiple_of(p0, BLOCK_SIZE_K)
            p1 = tl.multiple_of(p1, BLOCK_SIZE_K)
        r = pm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c = pn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        kb = tl.arange(0, BLOCK_SIZE_K // 2)
        ks = tl.arange(0, BLOCK_SIZE_K // 32)
        a_ptrs = (
            dy_payload
            + r[:, None].to(tl.int64) * payload_stride
            + p0 // 2
            + kb[None, :]
        )
        b_ptrs = (
            x_payload + c[None, :].to(tl.int64) * payload_stride + p0 // 2 + kb[:, None]
        )
        sa_ptrs = (
            dy_scales + r[:, None].to(tl.int64) * scale_stride + p0 // 32 + ks[None, :]
        )
        sb_ptrs = (
            x_scales + c[:, None].to(tl.int64) * scale_stride + p0 // 32 + ks[None, :]
        )
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
        for _ in range((p1 - p0) // BLOCK_SIZE_K):
            a = tl.load(
                a_ptrs, mask=r[:, None] < M, other=0, cache_modifier=CACHE_MODIFIER
            )
            b = tl.load(
                b_ptrs, mask=c[None, :] < N, other=0, cache_modifier=CACHE_MODIFIER
            )
            sa = tl.load(sa_ptrs, mask=r[:, None] < M, other=127)
            sb = tl.load(sb_ptrs, mask=c[:, None] < N, other=127)
            acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
            a_ptrs += BLOCK_SIZE_K // 2
            b_ptrs += BLOCK_SIZE_K // 2
            sa_ptrs += BLOCK_SIZE_K // 32
            sb_ptrs += BLOCK_SIZE_K // 32
        mask = (r[:, None] < M) & (c[None, :] < N)
        if TRANSPOSE_OUTPUT:
            dst = (
                out_ptr
                + expert.to(tl.int64) * (M * N)
                + c[None, :].to(tl.int64) * M
                + r[:, None]
            )
        else:
            dst = (
                out_ptr
                + expert.to(tl.int64) * (M * N)
                + r[:, None].to(tl.int64) * N
                + c[None, :]
            )
        # Empty experts write zero in overwrite mode and leave accumulation unchanged.
        if ACCUMULATE:
            if p1 > p0:
                acc += tl.load(dst, mask=mask, other=0.0)
                tl.store(dst, acc, mask=mask)
        else:
            tl.store(dst, acc, mask=mask)
        if PERSISTENT:
            tile += num_programs
        else:
            tile = total
