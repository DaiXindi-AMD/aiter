# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Private fused MXFP4 training kernels.

The standalone global-memory Hadamard kernel from Lumen is intentionally not
migrated: every production caller either uses the PyTorch reference transform
or one of the fused kernels below.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

from .mxfp4_layout import (
    MXFP4_SCALE_KCHUNK_C,
    MXFP4_SCALE_STRIPE_C,
    MXFP4_SHUFFLE_TILE_ROWS_C,
    MXFP4_SHUFFLE_UNIT_BYTES_C,
    _shuffled_fp4_offsets,
    _swizzled_scale_offsets,
)
from .quant_mxfp4 import _calculate_fp4_scales, _pack_fp4


@triton.jit
def _hadamard16_butterfly(x, ROWS: tl.constexpr):
    """Normalized in-register Hadamard-16 for ``x`` shaped ``(ROWS, 16)``."""
    x_r = tl.reshape(x, (ROWS, 8, 2, 1))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(
        tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16)
    )

    x_r = tl.reshape(x, (ROWS, 4, 2, 2))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(
        tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16)
    )

    x_r = tl.reshape(x, (ROWS, 2, 2, 4))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(
        tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16)
    )

    x_r = tl.reshape(x, (ROWS, 1, 2, 8))
    x_p = tl.permute(x_r, (0, 1, 3, 2))
    top, bot = tl.split(x_p)
    x = tl.reshape(
        tl.permute(tl.join(top + bot, top - bot), (0, 1, 3, 2)), (ROWS, 16)
    )
    return x * 0.25


@triton.jit
def _hadamard16_mfma(x, hmat_ptr, ROWS: tl.constexpr):
    """Normalized signed Hadamard-16 via one BF16 matrix instruction."""
    G: tl.constexpr = 16
    hmat = tl.load(
        hmat_ptr + tl.arange(0, G)[:, None] * G + tl.arange(0, G)[None, :]
    )
    return tl.dot(x.reshape(ROWS, G), hmat, out_dtype=tl.float32)


_fused_hadamard_quant_mxfp4_kernel_repr = make_kernel_repr(
    "_fused_hadamard_quant_mxfp4_kernel",
    ["BLOCK_M", "BLOCK_N", "QUANT_BLOCK_SIZE", "USE_SR", "USE_ASM"],
)


@triton.jit(repr=_fused_hadamard_quant_mxfp4_kernel_repr)
def _fused_hadamard_quant_mxfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    sign_ptr,
    hmat_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Hadamard-16 rotate and MXFP4-quantize without an intermediate write."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    G: tl.constexpr = 16
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    NUM_GROUPS: tl.constexpr = BLOCK_N // G
    ROWS: tl.constexpr = BLOCK_M * NUM_GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_in = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    )

    if x_in.type.element_ty == tl.bfloat16:
        x = _hadamard16_mfma(x_in, hmat_ptr, ROWS=ROWS).reshape(
            BLOCK_M, BLOCK_N
        )
    else:
        sign = tl.load(sign_ptr + tl.arange(0, G)).to(tl.float32)
        x = x_in.to(tl.float32).reshape(ROWS, G) * sign[None, :]
        x = _hadamard16_butterfly(x, ROWS=ROWS).reshape(BLOCK_M, BLOCK_N)

    scales = _calculate_fp4_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )
    y = _pack_fp4(
        x,
        scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    offs_yn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn,
        y,
    )
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    tl.store(
        s_ptr + offs_m[:, None] * stride_sm + offs_sn[None, :] * stride_sn,
        scales,
    )


_dual_layout_quant_mxfp4_kernel_repr = make_kernel_repr(
    "_dual_layout_quant_mxfp4_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "QUANT_BLOCK_SIZE",
        "USE_SR_A",
        "USE_SR_B",
        "USE_ASM",
        "SWIZZLE_SCALE",
        "NUM_SCALE_COLS_A",
        "NUM_SCALE_COLS_B",
        "SHUFFLE_B",
        "NUM_PACKED_COLS_B",
    ],
)


@triton.jit(repr=_dual_layout_quant_mxfp4_kernel_repr)
def _dual_layout_quant_mxfp4_kernel(
    x_ptr,
    a_ptr,
    as_ptr,
    b_ptr,
    bs_ptr,
    sign_ptr,
    hmat_ptr,
    stride_xm,
    stride_xn,
    stride_am,
    stride_an,
    stride_asm,
    stride_asn,
    stride_bm,
    stride_bn,
    stride_bsm,
    stride_bsn,
    philox_seed,
    philox_offset_a,
    philox_offset_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_SR_A: tl.constexpr,
    USE_SR_B: tl.constexpr,
    USE_ASM: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    NUM_SCALE_COLS_A: tl.constexpr,
    NUM_SCALE_COLS_B: tl.constexpr,
    SHUFFLE_B: tl.constexpr = False,
    NUM_PACKED_COLS_B: tl.constexpr = 0,
):
    """Read one tile and emit row-major plus rotated/transposed MXFP4."""
    G: tl.constexpr = 16
    tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_M % G == 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_in = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    )
    x = x_in.to(tl.float32)

    a_scales = _calculate_fp4_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )
    a = _pack_fp4(
        x,
        a_scales,
        philox_seed,
        philox_offset_a,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR_A,
        USE_ASM=USE_ASM,
    )

    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_an = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    tl.store(
        a_ptr + offs_m[:, None] * stride_am + offs_an[None, :] * stride_an,
        a,
    )
    offs_asn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if SWIZZLE_SCALE:
        tl.store(
            as_ptr
            + _swizzled_scale_offsets(
                offs_m[:, None],
                offs_asn[None, :],
                NUM_SCALE_COLS_A,
                STRIPE=MXFP4_SCALE_STRIPE_C,
                KCHUNK=MXFP4_SCALE_KCHUNK_C,
            ),
            a_scales,
        )
    else:
        tl.store(
            as_ptr
            + offs_m[:, None] * stride_asm
            + offs_asn[None, :] * stride_asn,
            a_scales,
        )

    ROWS_B: tl.constexpr = BLOCK_N * (BLOCK_M // G)
    if x_in.type.element_ty == tl.bfloat16:
        xt = _hadamard16_mfma(
            tl.trans(x_in), hmat_ptr, ROWS=ROWS_B
        ).reshape(BLOCK_N, BLOCK_M)
    else:
        sign = tl.load(sign_ptr + tl.arange(0, G)).to(tl.float32)
        xt = tl.trans(x).reshape(ROWS_B, G) * sign[None, :]
        xt = _hadamard16_butterfly(xt, ROWS=ROWS_B).reshape(BLOCK_N, BLOCK_M)

    b_scales = _calculate_fp4_scales(
        xt,
        BLOCK_M=BLOCK_N,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )
    b = _pack_fp4(
        xt,
        b_scales,
        philox_seed,
        philox_offset_b,
        BLOCK_M=BLOCK_N,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR_B,
        USE_ASM=USE_ASM,
    )

    HALF_BLOCK_M: tl.constexpr = BLOCK_M // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    offs_bn = pid_m * HALF_BLOCK_M + tl.arange(0, HALF_BLOCK_M)
    if SHUFFLE_B:
        tl.store(
            b_ptr
            + _shuffled_fp4_offsets(
                offs_n[:, None],
                offs_bn[None, :],
                NUM_PACKED_COLS_B,
                TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS_C,
                UNIT=MXFP4_SHUFFLE_UNIT_BYTES_C,
            ),
            b,
        )
    else:
        tl.store(
            b_ptr + offs_n[:, None] * stride_bm + offs_bn[None, :] * stride_bn,
            b,
        )

    offs_bsn = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    if SWIZZLE_SCALE:
        tl.store(
            bs_ptr
            + _swizzled_scale_offsets(
                offs_n[:, None],
                offs_bsn[None, :],
                NUM_SCALE_COLS_B,
                STRIPE=MXFP4_SCALE_STRIPE_C,
                KCHUNK=MXFP4_SCALE_KCHUNK_C,
            ),
            b_scales,
        )
    else:
        tl.store(
            bs_ptr
            + offs_n[:, None] * stride_bsm
            + offs_bsn[None, :] * stride_bsn,
            b_scales,
        )


@triton.jit
def _fp4_e2m1_decode(code):
    """Decode one unpacked FP4 E2M1 code to fp32."""
    magnitude = code & 0x07
    sign = (code >> 3).to(tl.float32)
    val = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(
                            magnitude == 5,
                            3.0,
                            tl.where(magnitude == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where(sign > 0.5, -val, val)


@triton.jit
def _e8m0_decode(scale_raw):
    """Decode E8M0 with AITER's endpoint semantics."""
    scale_bits = scale_raw.to(tl.uint32) << 23
    scale_bits = tl.where(scale_raw == 0, 0x00400000, scale_bits)
    scale_bits = tl.where(scale_raw == 0xFF, 0x7F800001, scale_bits)
    return scale_bits.to(tl.float32, bitcast=True)


_dequant_hadamard_quant_mxfp4_kernel_repr = make_kernel_repr(
    "_dequant_hadamard_quant_mxfp4_kernel",
    [
        "BLOCK_M",
        "BLOCK_K",
        "QUANT_BLOCK_SIZE",
        "USE_SR",
        "USE_ASM",
        "SWIZZLE_SCALE",
        "NUM_SCALE_COLS",
        "SHUFFLE_DATA",
        "NUM_PACKED_COLS",
        "IN_SCALE_SWIZZLED",
        "NUM_IN_SCALE_COLS",
    ],
)


@triton.jit(repr=_dequant_hadamard_quant_mxfp4_kernel_repr)
def _dequant_hadamard_quant_mxfp4_kernel(
    fp4_ptr,
    in_scale_ptr,
    out_ptr,
    out_scale_ptr,
    hmat_ptr,
    stride_fm,
    stride_fk,
    stride_ism,
    stride_isk,
    stride_om,
    stride_on,
    stride_osm,
    stride_osn,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    NUM_SCALE_COLS: tl.constexpr,
    SHUFFLE_DATA: tl.constexpr,
    NUM_PACKED_COLS: tl.constexpr,
    IN_SCALE_SWIZZLED: tl.constexpr = False,
    NUM_IN_SCALE_COLS: tl.constexpr = 0,
):
    """Dequantize, transpose, rotate, and requantize entirely in registers."""
    G: tl.constexpr = 16
    tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_M % G == 0)
    tl.static_assert(BLOCK_K % QUANT_BLOCK_SIZE == 0)

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    HALF_BLOCK_K: tl.constexpr = BLOCK_K // 2
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // QUANT_BLOCK_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_fk = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)
    packed = tl.load(
        fp4_ptr + offs_m[:, None] * stride_fm + offs_fk[None, :] * stride_fk
    ).to(tl.uint8)
    vals = tl.reshape(
        tl.join(
            _fp4_e2m1_decode(packed & 0x0F),
            _fp4_e2m1_decode((packed >> 4) & 0x0F),
        ),
        (BLOCK_M, BLOCK_K),
    )

    offs_sk = pid_k * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)
    if IN_SCALE_SWIZZLED:
        in_scale_offs = _swizzled_scale_offsets(
            offs_m[:, None],
            offs_sk[None, :],
            NUM_IN_SCALE_COLS,
            STRIPE=MXFP4_SCALE_STRIPE_C,
            KCHUNK=MXFP4_SCALE_KCHUNK_C,
        )
    else:
        in_scale_offs = (
            offs_m[:, None] * stride_ism + offs_sk[None, :] * stride_isk
        )
    scale_raw = tl.load(in_scale_ptr + in_scale_offs).to(tl.int32)
    scale_f32 = _e8m0_decode(scale_raw)
    x = vals * (
        scale_f32.reshape(BLOCK_M, SCALE_BLOCK_K, 1)
        .broadcast_to(BLOCK_M, SCALE_BLOCK_K, QUANT_BLOCK_SIZE)
        .reshape(BLOCK_M, BLOCK_K)
    )

    ROWS: tl.constexpr = BLOCK_K * (BLOCK_M // G)
    xt = _hadamard16_mfma(
        tl.trans(x.to(tl.bfloat16)), hmat_ptr, ROWS=ROWS
    ).reshape(BLOCK_K, BLOCK_M)

    out_scales = _calculate_fp4_scales(
        xt,
        BLOCK_M=BLOCK_K,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
    )
    y = _pack_fp4(
        xt,
        out_scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_K,
        BLOCK_N=BLOCK_M,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    HALF_BLOCK_M: tl.constexpr = BLOCK_M // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_on = pid_m * HALF_BLOCK_M + tl.arange(0, HALF_BLOCK_M)
    if SHUFFLE_DATA:
        tl.store(
            out_ptr
            + _shuffled_fp4_offsets(
                offs_k[:, None],
                offs_on[None, :],
                NUM_PACKED_COLS,
                TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS_C,
                UNIT=MXFP4_SHUFFLE_UNIT_BYTES_C,
            ),
            y,
        )
    else:
        tl.store(
            out_ptr
            + offs_k[:, None] * stride_om
            + offs_on[None, :] * stride_on,
            y,
        )

    offs_osn = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    if SWIZZLE_SCALE:
        tl.store(
            out_scale_ptr
            + _swizzled_scale_offsets(
                offs_k[:, None],
                offs_osn[None, :],
                NUM_SCALE_COLS,
                STRIPE=MXFP4_SCALE_STRIPE_C,
                KCHUNK=MXFP4_SCALE_KCHUNK_C,
            ),
            out_scales,
        )
    else:
        tl.store(
            out_scale_ptr
            + offs_k[:, None] * stride_osm
            + offs_osn[None, :] * stride_osn,
            out_scales,
        )


_dequant_transpose_mxfp4_kernel_repr = make_kernel_repr(
    "_dequant_transpose_mxfp4_kernel",
    ["K", "BLOCK_M", "BLOCK_K", "QUANT_BLOCK_SIZE"],
)


@triton.jit(repr=_dequant_transpose_mxfp4_kernel_repr)
def _dequant_transpose_mxfp4_kernel(
    fp4_ptr,
    scale_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    stride_fm,
    stride_fk,
    stride_sm,
    stride_sk,
    stride_ok,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Fused packed-MXFP4 dequantize plus transpose to BF16."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    HALF_BLOCK_K: tl.constexpr = BLOCK_K // 2
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // QUANT_BLOCK_SIZE

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_packed = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)
    mask_fp4 = (rm[:, None] < M) & (rk_packed[None, :] < (K // 2))
    packed = tl.load(
        fp4_ptr + rm[:, None] * stride_fm + rk_packed[None, :] * stride_fk,
        mask=mask_fp4,
        other=0,
    ).to(tl.uint8)
    vals = tl.reshape(
        tl.join(
            _fp4_e2m1_decode(packed & 0x0F),
            _fp4_e2m1_decode((packed >> 4) & 0x0F),
        ),
        (BLOCK_M, BLOCK_K),
    )

    rk_scale = pid_k * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)
    mask_scale = (rm[:, None] < M) & (
        rk_scale[None, :] < (K // QUANT_BLOCK_SIZE)
    )
    scale_raw = tl.load(
        scale_ptr + rm[:, None] * stride_sm + rk_scale[None, :] * stride_sk,
        mask=mask_scale,
        other=127,
    ).to(tl.int32)
    scale_f32 = _e8m0_decode(scale_raw)
    scale_expanded = (
        scale_f32.reshape(BLOCK_M, SCALE_BLOCK_K, 1)
        .broadcast_to(BLOCK_M, SCALE_BLOCK_K, QUANT_BLOCK_SIZE)
        .reshape(BLOCK_M, BLOCK_K)
    )
    result = (vals * scale_expanded).to(tl.bfloat16)

    rk_full = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_out = (rk_full[:, None] < K) & (rm[None, :] < M)
    tl.store(
        out_ptr + rk_full[:, None] * stride_ok + rm[None, :] * stride_om,
        tl.trans(result),
        mask=mask_out,
    )
