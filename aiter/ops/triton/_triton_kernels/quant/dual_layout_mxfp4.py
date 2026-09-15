# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _hadamard16(x, rows: tl.constexpr):
    """Apply a normalized Hadamard-16 transform to the last dimension."""
    # Normalize before the butterfly so finite inputs whose normalized result
    # is representable cannot overflow in an intermediate unnormalized sum.
    x = x * 0.25
    x_reshaped = tl.reshape(x, (rows, 8, 2, 1))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )

    x_reshaped = tl.reshape(x, (rows, 4, 2, 2))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )

    x_reshaped = tl.reshape(x, (rows, 2, 2, 4))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )

    x_reshaped = tl.reshape(x, (rows, 1, 2, 8))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )
    return x


@triton.jit
def _mxfp4_scale_op(x):
    """Return deterministic EVEN E8M0 scales for grouped FP32 values."""
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    return scale_e8m0_unbiased.to(tl.uint8) + 127


@triton.jit
def _mxfp4_e8m0_to_fp32(scales):
    """Decode E8M0, including raw zero's exact value of 2^-127."""
    scale_bits = scales.to(tl.uint32) << 23
    scale_bits = tl.where(scales == 0, 0x00400000, scale_bits)
    return scale_bits.to(tl.float32, bitcast=True)


@triton.jit
def _mxfp4_sr_random_words(
    rows,
    pid_n,
    N,
    philox_seed,
    philox_offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Generate one random word per packed pair from global Philox counters."""
    COUNTERS_PER_TILE_ROW: tl.constexpr = BLOCK_SIZE_N // 8
    n_u64 = tl.cast(N, tl.uint64)
    counters_per_row = n_u64 // 8
    counter_cols = tl.cast(pid_n, tl.uint64) * COUNTERS_PER_TILE_ROW + tl.arange(
        0, COUNTERS_PER_TILE_ROW
    ).to(tl.uint64)
    offsets = (
        tl.cast(philox_offset, tl.uint64)
        + rows.to(tl.uint64)[:, None] * counters_per_row
        + counter_cols[None, :]
    )
    random_0, random_1, random_2, random_3 = tl.randint4x(philox_seed, offsets)
    return tl.join(tl.join(random_0, random_1), tl.join(random_2, random_3)).reshape(
        BLOCK_SIZE_M, BLOCK_SIZE_N // 2
    )


@triton.jit
def _mxfp4_sr_pack(
    x,
    scales,
    random_words,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Pack adjacent values with gfx950 stochastic scaled E2M1 conversion."""
    HALF_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = MXFP4_QUANT_BLOCK_SIZE // 2
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_low, x_high = tl.split(x.reshape(BLOCK_SIZE_M, HALF_BLOCK_SIZE_N, 2))
    scale_fp32 = _mxfp4_e8m0_to_fp32(scales)
    scale_fp32 = (
        scale_fp32.expand_dims(axis=2)
        .broadcast_to(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, HALF_QUANT_BLOCK_SIZE)
        .reshape(BLOCK_SIZE_M, HALF_BLOCK_SIZE_N)
    )

    if x_low.type.element_ty == tl.float32:
        packed_input = (
            x_high.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
        ) | x_low.to(tl.uint32, bitcast=True)
        packed = tl.inline_asm_elementwise(
            asm="v_cvt_scalef32_sr_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];",
            constraints="=&v,v,v,v",
            args=[packed_input, random_words, scale_fp32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    else:
        tl.static_assert(x_low.type.element_ty == tl.bfloat16)
        packed_input = (
            x_high.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        ) | x_low.to(tl.uint16, bitcast=True)
        packed = tl.inline_asm_elementwise(
            asm="v_cvt_scalef32_sr_pk_fp4_bf16 $0, $1, $2, $3 op_sel:[0,0,0,0];",
            constraints="=&v,v,v,v",
            args=[packed_input, random_words, scale_fp32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )

    return (packed & 0xFF).to(tl.uint8).reshape(BLOCK_SIZE_M, HALF_BLOCK_SIZE_N)


@triton.jit
def _mxfp4_rtn_pack(
    x,
    scales,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Pack adjacent values with gfx950 round-to-nearest E2M1 conversion."""
    HALF_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = MXFP4_QUANT_BLOCK_SIZE // 2
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_low, x_high = tl.split(x.reshape(BLOCK_SIZE_M, HALF_BLOCK_SIZE_N, 2))
    scale_fp32 = _mxfp4_e8m0_to_fp32(scales)
    scale_fp32 = (
        scale_fp32.expand_dims(axis=2)
        .broadcast_to(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, HALF_QUANT_BLOCK_SIZE)
        .reshape(BLOCK_SIZE_M, HALF_BLOCK_SIZE_N)
    )

    if x_low.type.element_ty == tl.float32:
        packed = tl.inline_asm_elementwise(
            asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];",
            constraints="=&v,v,v,v",
            args=[x_low, x_high, scale_fp32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
    else:
        tl.static_assert(x_low.type.element_ty == tl.bfloat16)
        packed_input = (
            x_high.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
        ) | x_low.to(tl.uint16, bitcast=True)
        packed = tl.inline_asm_elementwise(
            asm="v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2 op_sel:[0,0,0,0];",
            constraints="=&v,v,v",
            args=[packed_input, scale_fp32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )

    return (packed & 0xFF).to(tl.uint8).reshape(BLOCK_SIZE_M, HALF_BLOCK_SIZE_N)


_dual_layout_quant_mxfp4_repr = make_kernel_repr(
    "_dual_layout_quant_mxfp4_kernel",
    ["BLOCK_SIZE", "HADAMARD_SIZE", "USE_SR_ROW", "USE_SR_TRANSPOSED"],
)


@triton.jit(repr=_dual_layout_quant_mxfp4_repr)
def _dual_layout_quant_mxfp4_kernel(
    x_ptr,
    row_ptr,
    row_scale_ptr,
    transposed_ptr,
    transposed_scale_ptr,
    sign_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_row_m_in,
    stride_row_n_in,
    stride_row_scale_m_in,
    stride_row_scale_n_in,
    stride_transposed_m_in,
    stride_transposed_n_in,
    stride_transposed_scale_m_in,
    stride_transposed_scale_n_in,
    M,
    N,
    philox_seed,
    philox_offset_row,
    philox_offset_transposed,
    BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    USE_SR_ROW: tl.constexpr,
    USE_SR_TRANSPOSED: tl.constexpr,
):
    """Read one tile and emit MXFP4(x) plus MXFP4(H16(x.T))."""
    tl.static_assert(BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)

    pid_m = tl.cast(tl.program_id(0), tl.int64)
    pid_n = tl.cast(tl.program_id(1), tl.int64)

    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_row_m = tl.cast(stride_row_m_in, tl.int64)
    stride_row_n = tl.cast(stride_row_n_in, tl.int64)
    stride_row_scale_m = tl.cast(stride_row_scale_m_in, tl.int64)
    stride_row_scale_n = tl.cast(stride_row_scale_n_in, tl.int64)
    stride_transposed_m = tl.cast(stride_transposed_m_in, tl.int64)
    stride_transposed_n = tl.cast(stride_transposed_n_in, tl.int64)
    stride_transposed_scale_m = tl.cast(stride_transposed_scale_m_in, tl.int64)
    stride_transposed_scale_n = tl.cast(stride_transposed_scale_n_in, tl.int64)

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x_m + cols[None, :] * stride_x_n,
        cache_modifier=".cg",
    )
    x_fp32 = x.to(tl.float32)

    row_grouped = x_fp32.reshape(BLOCK_SIZE, 1, BLOCK_SIZE)
    row_scales = _mxfp4_scale_op(row_grouped).reshape(BLOCK_SIZE, 1)
    if USE_SR_ROW:
        row_random_words = _mxfp4_sr_random_words(
            rows,
            pid_n,
            N,
            philox_seed,
            philox_offset_row,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
        row_packed = _mxfp4_sr_pack(
            x,
            row_scales,
            row_random_words,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        row_packed = _mxfp4_rtn_pack(
            x,
            row_scales,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=BLOCK_SIZE,
        )

    packed_cols = pid_n * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(tl.int64)
    tl.store(
        row_ptr + rows[:, None] * stride_row_m + packed_cols[None, :] * stride_row_n,
        row_packed,
    )
    tl.store(
        row_scale_ptr + rows * stride_row_scale_m + pid_n * stride_row_scale_n,
        row_scales.reshape(BLOCK_SIZE),
    )

    sign = tl.load(sign_ptr + tl.arange(0, HADAMARD_SIZE).to(tl.int64)).to(tl.float32)
    HADAMARD_ROWS: tl.constexpr = BLOCK_SIZE * (BLOCK_SIZE // HADAMARD_SIZE)
    transposed = tl.trans(x_fp32).reshape(HADAMARD_ROWS, HADAMARD_SIZE)
    transposed = transposed * sign[None, :]
    transposed = _hadamard16(transposed, rows=HADAMARD_ROWS).reshape(
        BLOCK_SIZE, BLOCK_SIZE
    )

    transposed_grouped = transposed.reshape(BLOCK_SIZE, 1, BLOCK_SIZE)
    transposed_scales = _mxfp4_scale_op(transposed_grouped).reshape(BLOCK_SIZE, 1)
    if USE_SR_TRANSPOSED:
        transposed_random_words = _mxfp4_sr_random_words(
            cols,
            pid_m,
            M,
            philox_seed,
            philox_offset_transposed,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
        transposed_packed = _mxfp4_sr_pack(
            transposed,
            transposed_scales,
            transposed_random_words,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        transposed_packed = _mxfp4_rtn_pack(
            transposed,
            transposed_scales,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=BLOCK_SIZE,
        )

    packed_rows = pid_m * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(tl.int64)
    tl.store(
        transposed_ptr
        + cols[:, None] * stride_transposed_m
        + packed_rows[None, :] * stride_transposed_n,
        transposed_packed,
    )
    tl.store(
        transposed_scale_ptr
        + cols * stride_transposed_scale_m
        + pid_m * stride_transposed_scale_n,
        transposed_scales.reshape(BLOCK_SIZE),
    )
