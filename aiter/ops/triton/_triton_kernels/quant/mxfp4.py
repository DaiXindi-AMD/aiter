# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import (
    _mxfp4_pack_op,
    _mxfp4_scale_from_amax,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _decode_e2m1(code):
    magnitude = code & 0x07
    value = tl.where(
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
    return tl.where((code & 0x08) != 0, -value, value)


@triton.jit
def _decode_e8m0(raw_scale):
    scale_bits = raw_scale.to(tl.uint32) << 23
    scale_bits = tl.where(raw_scale == 0, 0x00400000, scale_bits)
    return scale_bits.to(tl.float32, bitcast=True)


@triton.jit
def _normalized_hadamard16(x, ROWS: tl.constexpr):
    # Normalize before the butterfly so large finite inputs cannot overflow in
    # an otherwise finite normalized H16 result.
    x = x * 0.25
    x_reshaped = tl.reshape(x, (ROWS, 8, 2, 1))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    top, bottom = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(top + bottom, top - bottom), (0, 1, 3, 2)),
        (ROWS, 16),
    )

    x_reshaped = tl.reshape(x, (ROWS, 4, 2, 2))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    top, bottom = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(top + bottom, top - bottom), (0, 1, 3, 2)),
        (ROWS, 16),
    )

    x_reshaped = tl.reshape(x, (ROWS, 2, 2, 4))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    top, bottom = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(top + bottom, top - bottom), (0, 1, 3, 2)),
        (ROWS, 16),
    )

    x_reshaped = tl.reshape(x, (ROWS, 1, 2, 8))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    top, bottom = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(top + bottom, top - bottom), (0, 1, 3, 2)),
        (ROWS, 16),
    )
    return x


_dequant_hadamard_quant_mxfp4_repr = make_kernel_repr(
    "_dequant_hadamard_quant_mxfp4_kernel",
    ["BLOCK_SIZE", "HADAMARD_SIZE"],
)


@triton.jit(repr=_dequant_hadamard_quant_mxfp4_repr)
def _dequant_hadamard_quant_mxfp4_kernel(
    packed_ptr,
    input_scale_ptr,
    sign_ptr,
    output_ptr,
    output_scale_ptr,
    stride_packed_m_in,
    stride_packed_k_in,
    stride_input_scale_m_in,
    stride_input_scale_k_in,
    stride_output_k_in,
    stride_output_m_in,
    stride_output_scale_k_in,
    stride_output_scale_m_in,
    BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
):
    """Dequantize, transpose, H16-rotate, and requantize one 32x32 tile."""
    tl.static_assert(BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)

    pid_m = tl.cast(tl.program_id(0), tl.int64)
    pid_k = tl.cast(tl.program_id(1), tl.int64)

    stride_packed_m = tl.cast(stride_packed_m_in, tl.int64)
    stride_packed_k = tl.cast(stride_packed_k_in, tl.int64)
    stride_input_scale_m = tl.cast(stride_input_scale_m_in, tl.int64)
    stride_input_scale_k = tl.cast(stride_input_scale_k_in, tl.int64)
    stride_output_k = tl.cast(stride_output_k_in, tl.int64)
    stride_output_m = tl.cast(stride_output_m_in, tl.int64)
    stride_output_scale_k = tl.cast(stride_output_scale_k_in, tl.int64)
    stride_output_scale_m = tl.cast(stride_output_scale_m_in, tl.int64)

    offsets_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offsets_k_packed = pid_k * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(
        tl.int64
    )
    packed_offsets = (
        offsets_m[:, None] * stride_packed_m
        + offsets_k_packed[None, :] * stride_packed_k
    )
    packed = tl.load(packed_ptr + packed_offsets, cache_modifier=".cg").to(tl.uint8)

    low = _decode_e2m1(packed & 0x0F)
    high = _decode_e2m1((packed >> 4) & 0x0F)
    values = tl.reshape(tl.join(low, high), (BLOCK_SIZE, BLOCK_SIZE))

    input_scale_offsets = (
        offsets_m * stride_input_scale_m + pid_k * stride_input_scale_k
    )
    input_scale_raw = tl.load(
        input_scale_ptr + input_scale_offsets, cache_modifier=".cg"
    ).to(tl.uint8)
    invalid_input_scale = tl.max((input_scale_raw == 255).to(tl.int32), axis=0) != 0
    input_scale = _decode_e8m0(input_scale_raw)
    input_scale = tl.where(input_scale_raw == 255, 0.0, input_scale)
    dequantized = values * input_scale[:, None]

    sign = tl.load(sign_ptr + tl.arange(0, HADAMARD_SIZE)).to(tl.float32)
    rows: tl.constexpr = BLOCK_SIZE * (BLOCK_SIZE // HADAMARD_SIZE)
    # The Lumen composition materializes this stage as BF16 before H16. E2M1
    # times a power-of-two scale is exact in BF16 whenever it is finite.
    transposed = tl.trans(dequantized.to(tl.bfloat16)).to(tl.float32)
    signed = tl.reshape(transposed, (rows, HADAMARD_SIZE)) * sign[None, :]
    rotated = tl.reshape(
        _normalized_hadamard16(signed, ROWS=rows),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    # Lumen's fused production path quantizes the FP32 H16 accumulator directly.
    # Its matrix multiply canonicalizes exact cancellation and signed zero to
    # +0, so preserve that byte-level behavior before E2M1 packing.
    rotated = tl.where(rotated == 0.0, 0.0, rotated)

    rotated_bits = rotated.to(tl.uint32, bitcast=True)
    invalid_value = (rotated_bits & 0x7F800000) == 0x7F800000
    invalid_row = tl.max(invalid_value.to(tl.int32), axis=1) != 0
    finite_rotated = tl.where(invalid_value, 0.0, rotated)
    row_amax = tl.max(tl.abs(finite_rotated), axis=1, keep_dims=True)
    output_scale_raw, quant_scale = _mxfp4_scale_from_amax(row_amax)

    # exp2(-127) may flush to zero on gfx950. Use the smallest normal
    # reciprocal for raw 254, then apply the remaining factor to normalized
    # values instead of materializing a subnormal reciprocal.
    safe_quant_scale = tl.where(output_scale_raw == 254, tl.exp2(-126.0), quant_scale)
    normalized = finite_rotated * safe_quant_scale
    normalized = tl.where(output_scale_raw == 254, normalized * 0.5, normalized)
    output = _mxfp4_pack_op(
        tl.reshape(normalized, (BLOCK_SIZE, 1, BLOCK_SIZE)),
        BLOCK_SIZE,
        BLOCK_SIZE,
        BLOCK_SIZE,
    )
    output_scale_raw = tl.reshape(output_scale_raw, (BLOCK_SIZE,))
    invalid_row = invalid_row | invalid_input_scale
    output_scale_raw = tl.where(invalid_row, 255, output_scale_raw).to(tl.uint8)
    output = tl.where(invalid_row[:, None], 0, output)

    offsets_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offsets_m_packed = pid_m * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(
        tl.int64
    )
    output_offsets = (
        offsets_k[:, None] * stride_output_k
        + offsets_m_packed[None, :] * stride_output_m
    )
    tl.store(output_ptr + output_offsets, output)

    output_scale_offsets = (
        offsets_k * stride_output_scale_k + pid_m * stride_output_scale_m
    )
    tl.store(output_scale_ptr + output_scale_offsets, output_scale_raw)
