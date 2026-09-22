# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Private MXFP4 quantization kernels missing from AITER's existing paths.

The public wrapper uses AITER's HIP/Triton quantizers for ordinary 1x32 RTN.
This module supplies stochastic payload rounding, 2-D scales, and fused layout
stores. Scale selection remains deterministic round-even for both RTN and SR.
"""

import os

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.mxfp4_layout import (
    MXFP4_SCALE_KCHUNK_C,
    MXFP4_SCALE_STRIPE_C,
    MXFP4_SHUFFLE_TILE_ROWS_C,
    MXFP4_SHUFFLE_UNIT_BYTES_C,
    _shuffled_fp4_offsets,
    _swizzled_scale_offsets,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

SR_PHILOX_ROUNDS_DEFAULT = 7
SR_PHILOX_ROUNDS = int(
    os.environ.get("AITER_MXFP4_SR_PHILOX_ROUNDS", SR_PHILOX_ROUNDS_DEFAULT)
)
SR_PHILOX_ROUNDS_C = tl.constexpr(SR_PHILOX_ROUNDS)

if (
    SR_PHILOX_ROUNDS != SR_PHILOX_ROUNDS_DEFAULT
    and os.environ.get("RANK", "0") == "0"
):
    print(
        f"[aiter] AITER_MXFP4_SR_PHILOX_ROUNDS={SR_PHILOX_ROUNDS} "
        f"(default {SR_PHILOX_ROUNDS_DEFAULT}); this changes MXFP4 "
        "stochastic-rounding numerics.",
        flush=True,
    )


@triton.jit
def _calculate_fp4_scales(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
):
    """Compute deterministic round-even E8M0 scales for E2M1 payloads."""
    E8M0_EXPONENT_BIAS: tl.constexpr = 127
    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)
    if IS_2D_BLOCK:
        tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)

    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits: tl.constexpr = 23
        hp_ebits: tl.constexpr = 8
        hp_exp_bias: tl.constexpr = 127
    else:
        hp_int_dtype = tl.int16
        hp_mbits: tl.constexpr = 7
        hp_ebits: tl.constexpr = 8
        hp_exp_bias: tl.constexpr = 127

    sbits: tl.constexpr = 1
    mbits: tl.constexpr = 1
    target_max_pow2: tl.constexpr = 2
    NEW_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    if IS_2D_BLOCK:
        NEW_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        x_r = x.reshape(
            NEW_BLOCK_M,
            QUANT_BLOCK_SIZE,
            NEW_BLOCK_N,
            QUANT_BLOCK_SIZE,
        )
        x_r = tl.permute(x_r, (0, 2, 1, 3))
        max_abs = tl.max(tl.abs(x_r), axis=-1)
        max_abs = tl.max(max_abs, axis=-1)
    else:
        x_r = x.reshape(BLOCK_M, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x_r), axis=-1)
    max_abs = max_abs.to(x.type.element_ty)

    # This is AITER/torchao's EVEN scale rule. It is deliberately independent
    # of USE_SR: stochastic rounding applies only to the E2M1 payload.
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + sbits)) - 1) << hp_mbits
    max_abs = (max_abs + val_to_add) & mask

    extracted_pow2 = (
        ((max_abs >> hp_mbits) & 0b11111111).to(tl.int32) - hp_exp_bias
    )
    scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
    scale_e8m0_unbiased = tl.minimum(
        tl.maximum(scale_e8m0_unbiased, -E8M0_EXPONENT_BIAS),
        E8M0_EXPONENT_BIAS + 1,
    )
    return (scale_e8m0_unbiased + E8M0_EXPONENT_BIAS).to(tl.uint8)


@triton.jit
def _generate_randval(
    m: tl.constexpr,
    n: tl.constexpr,
    philox_seed,
    philox_offset,
):
    """Return an ``(m, n)`` tile of independent Philox words."""
    tile_id = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    if n % 4 == 0:
        QN: tl.constexpr = n // 4
        ms = tl.arange(0, m)
        ns = tl.arange(0, QN)
        tile_offset = philox_offset + tile_id * m * QN
        rng_offsets = tile_offset + ms[:, None] * QN + ns[None, :]
        r0, r1, r2, r3 = tl.randint4x(
            philox_seed, rng_offsets, SR_PHILOX_ROUNDS_C
        )
        return tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(m, n)

    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    tile_offset = philox_offset + tile_id * m * n
    rng_offsets = tile_offset + ms[:, None] * n + ns[None, :]
    r1, _, _, _ = tl.randint4x(
        philox_seed, rng_offsets, SR_PHILOX_ROUNDS_C
    )
    return r1


@triton.jit
def _pack_fp4(
    x,
    scales,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_SR: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
):
    """Pack E2M1 pairs, using gfx950 RTN/SR conversion when requested."""
    FP4_E2M1_MAX: tl.constexpr = 6.0
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

    # The gfx950 conversion consumes an fp32 power-of-two scale. Raw E8M0 zero
    # denotes 2^-127, which is exactly representable as the fp32 subnormal
    # 0x00400000. Using raw one here would quantize with 2^-126 while storing
    # raw zero, making the payload and its scale disagree by a factor of two.
    scale_bits = scales.to(tl.uint32) << 23
    scale_bits = tl.where(scales == 0, 0x00400000, scale_bits)
    scales_fp32 = scale_bits.to(tl.float32, bitcast=True)

    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        scales_fp32 = (
            scales_fp32.expand_dims(axis=(1, 3))
            .broadcast_to(
                SCALE_BLOCK_M,
                QUANT_BLOCK_SIZE,
                SCALE_BLOCK_N,
                HALF_QUANT_BLOCK_SIZE,
            )
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )
    else:
        scales_fp32 = (
            scales_fp32.expand_dims(axis=2)
            .broadcast_to(BLOCK_M, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE)
            .reshape(BLOCK_M, HALF_BLOCK_N)
        )

    # Keep the ASM and portable random tiles separate. Triton still type-checks
    # code after the constexpr ASM return, so reusing a half-width ASM tile in
    # the full-width fallback creates a shape mismatch during SR compilation.
    randval_asm = 0
    randval_fallback = 0
    if USE_SR:
        if USE_ASM:
            randval_asm = _generate_randval(
                BLOCK_M,
                HALF_BLOCK_N,
                philox_seed,
                philox_offset,
            )
        else:
            randval_fallback = _generate_randval(
                BLOCK_M,
                BLOCK_N,
                philox_seed,
                philox_offset,
            )

    if USE_ASM:
        if x0.type.element_ty == tl.float32:
            if USE_SR:
                x_packed = (
                    x1.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
                ) | x0.to(tl.uint32, bitcast=True)
                y = tl.inline_asm_elementwise(
                    asm=(
                        "v_cvt_scalef32_sr_pk_fp4_f32 $0, $1, $2, $3 "
                        "op_sel:[0,0,0,0];"
                    ),
                    constraints="=&v,v,v,v",
                    args=[x_packed, randval_asm, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
            else:
                y = tl.inline_asm_elementwise(
                    asm=(
                        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3 "
                        "op_sel:[0,0,0,0];"
                    ),
                    constraints="=&v,v,v,v",
                    args=[x0, x1, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
        else:
            x_packed_bf16 = (
                x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
            ) | x0.to(tl.uint16, bitcast=True)
            if USE_SR:
                y = tl.inline_asm_elementwise(
                    asm=(
                        "v_cvt_scalef32_sr_pk_fp4_bf16 $0, $1, $2, $3 "
                        "op_sel:[0,0,0,0];"
                    ),
                    constraints="=&v,v,v,v",
                    args=[x_packed_bf16, randval_asm, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
            else:
                y = tl.inline_asm_elementwise(
                    asm=(
                        "v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2 "
                        "op_sel:[0,0,0,0];"
                    ),
                    constraints="=&v,v,v",
                    args=[x_packed_bf16, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
        return (y & 0xFF).to(tl.uint8).reshape(BLOCK_M, HALF_BLOCK_N)

    # Development fallback. Public SR/training wrappers gate this path to
    # gfx950, where USE_ASM=True; retain a portable RTN implementation for
    # kernel development and 2-D layout validation on future architectures.
    x_scaled = x / scales_fp32.expand_dims(axis=2).broadcast_to(
        BLOCK_M, HALF_BLOCK_N, 2
    ).reshape(BLOCK_M, BLOCK_N)
    abs_val = tl.abs(x_scaled)
    sign = (x_scaled < 0.0).to(tl.uint8)

    if USE_SR:
        # Stochastic selection between adjacent E2M1 values. The scale remains
        # deterministic; only the payload consumes Philox randomness.
        lower = tl.where(
            abs_val < 0.5,
            0.0,
            tl.where(
                abs_val < 1.0,
                0.5,
                tl.where(
                    abs_val < 1.5,
                    1.0,
                    tl.where(
                        abs_val < 2.0,
                        1.5,
                        tl.where(
                            abs_val < 3.0,
                            2.0,
                            tl.where(abs_val < 4.0, 3.0, 4.0),
                        ),
                    ),
                ),
            ),
        )
        upper = tl.where(
            abs_val < 0.5,
            0.5,
            tl.where(
                abs_val < 1.0,
                1.0,
                tl.where(
                    abs_val < 1.5,
                    1.5,
                    tl.where(
                        abs_val < 2.0,
                        2.0,
                        tl.where(
                            abs_val < 3.0,
                            3.0,
                            tl.where(abs_val < 4.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        )
        lower_code = tl.where(
            abs_val < 0.5,
            0,
            tl.where(
                abs_val < 1.0,
                1,
                tl.where(
                    abs_val < 1.5,
                    2,
                    tl.where(
                        abs_val < 2.0,
                        3,
                        tl.where(
                            abs_val < 3.0,
                            4,
                            tl.where(abs_val < 4.0, 5, 6),
                        ),
                    ),
                ),
            ),
        ).to(tl.uint8)
        upper_code = tl.minimum(lower_code + 1, 7)
        probability_up = (abs_val - lower) / (upper - lower)
        random_unit = (randval_fallback.to(tl.uint32) >> 8).to(tl.float32) * (
            1.0 / 16777216.0
        )
        code = tl.where(random_unit < probability_up, upper_code, lower_code)
        code = tl.where(abs_val >= FP4_E2M1_MAX, 7, code)
    else:
        # RNE boundaries: strict comparisons select the even lower encoding at
        # 0.25, 1.25, 2.5, and 5.0; the other ties select the even upper code.
        code = tl.zeros_like(abs_val).to(tl.uint8)
        code = tl.where(abs_val > 0.25, 1, code)
        code = tl.where(abs_val >= 0.75, 2, code)
        code = tl.where(abs_val > 1.25, 3, code)
        code = tl.where(abs_val >= 1.75, 4, code)
        code = tl.where(abs_val > 2.50, 5, code)
        code = tl.where(abs_val >= 3.50, 6, code)
        code = tl.where(abs_val > 5.00, 7, code)

    fp4_code = (sign << 3) | code
    codes_reshaped = fp4_code.reshape(BLOCK_M, HALF_BLOCK_N, 2)
    even, odd = tl.split(codes_reshaped)
    return (even | (odd << 4)).to(tl.uint8)


_convert_to_mxfp4_kernel_repr = make_kernel_repr(
    "_convert_to_mxfp4_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "QUANT_BLOCK_SIZE",
        "IS_2D_BLOCK",
        "USE_SR",
        "USE_ASM",
        "SWIZZLE_SCALE",
        "NUM_SCALE_COLS",
        "SHUFFLE_DATA",
        "NUM_PACKED_COLS",
    ],
)


@triton.jit(repr=_convert_to_mxfp4_kernel_repr)
def _convert_to_mxfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
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
    IS_2D_BLOCK: tl.constexpr,
    USE_SR: tl.constexpr,
    USE_ASM: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    NUM_SCALE_COLS: tl.constexpr,
    SHUFFLE_DATA: tl.constexpr = False,
    NUM_PACKED_COLS: tl.constexpr = 0,
):
    """BF16/FP32 to packed MXFP4 and deterministic E8M0 block scales."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    )
    scales = _calculate_fp4_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )
    y = _pack_fp4(
        x,
        scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_SR=USE_SR,
        USE_ASM=USE_ASM,
    )

    offs_yn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    if SHUFFLE_DATA:
        tl.store(
            y_ptr
            + _shuffled_fp4_offsets(
                offs_m[:, None],
                offs_yn[None, :],
                NUM_PACKED_COLS,
                TILE_ROWS=MXFP4_SHUFFLE_TILE_ROWS_C,
                UNIT=MXFP4_SHUFFLE_UNIT_BYTES_C,
            ),
            y,
        )
    else:
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn,
            y,
        )

    if SWIZZLE_SCALE:
        tl.store(
            s_ptr
            + _swizzled_scale_offsets(
                offs_sm[:, None],
                offs_sn[None, :],
                NUM_SCALE_COLS,
                STRIPE=MXFP4_SCALE_STRIPE_C,
                KCHUNK=MXFP4_SCALE_KCHUNK_C,
            ),
            scales,
        )
    else:
        tl.store(
            s_ptr
            + offs_sm[:, None] * stride_sm
            + offs_sn[None, :] * stride_sn,
            scales,
        )
