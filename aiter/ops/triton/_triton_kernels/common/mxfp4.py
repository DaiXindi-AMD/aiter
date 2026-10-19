# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strict MXFP4 v1 device helpers; legacy native-pair quantization is separate."""

import triton
import triton.language as tl


@triton.jit
def finite_v1(x):
    return (x.to(tl.uint32, bitcast=True) & 0x7F800000) != 0x7F800000


@triton.jit
def selector_v1(amax):
    bits = amax.to(tl.uint32, bitcast=True)
    exponent = (((bits + 0x00200000) & 0xFF800000) >> 23).to(tl.int32)
    return tl.where(exponent == 255, 254, tl.maximum(exponent - 2, 0)).to(tl.uint8)


@triton.jit
def normalize_v1(x, raw):
    """Divide finite FP32 by E8M0 with one exact gradual-underflow rounding."""
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits & 0x80000000
    exponent = ((bits >> 23) & 255).to(tl.int32)
    mantissa = bits & 0x7FFFFF
    n = tl.where(exponent == 0, mantissa, mantissa | 0x800000)
    p = tl.where(exponent == 0, -149, exponent - 150) - (raw.to(tl.int32) - 127)
    work = n
    highest = tl.full(x.shape, 0, tl.int32)
    for step in tl.static_range(4, -1, -1):
        shift: tl.constexpr = 1 << step
        advance = work >= (1 << shift)
        highest += tl.where(advance, shift, 0)
        work = tl.where(advance, work >> shift, work)
    unbiased = p + highest
    normal = ((unbiased + 127).to(tl.uint32) << 23) | ((n << (23 - highest)) & 0x7FFFFF)
    shift = -(p + 149)
    safe_shift = tl.minimum(tl.maximum(shift, 1), 63)
    wide = n.to(tl.uint64)
    kept = wide >> safe_shift
    remainder = wide & ((tl.full((), 1, tl.uint64) << safe_shift) - 1)
    half = tl.full((), 1, tl.uint64) << (safe_shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((kept & 1) != 0))
    subnormal = tl.where(shift > 63, 0, kept + increment.to(tl.uint64))
    subnormal = tl.where(
        shift <= 0, wide << tl.minimum(tl.maximum(-shift, 0), 63), subnormal
    )
    magnitude = tl.where(unbiased >= -126, normal, subnormal.to(tl.uint32))
    magnitude = tl.where(n == 0, 0, magnitude)
    return (sign | magnitude).to(tl.float32, bitcast=True)


@triton.jit
def philox_element_v1(seed, draw):
    """Return direct r0/r1/r2/r3 words for absolute uint64 element draws."""
    seed = seed.to(tl.uint64)
    draw = draw.to(tl.uint64)
    counter = draw >> 2
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    c2 = tl.full(draw.shape, 0, tl.uint32)
    c3 = tl.full(draw.shape, 0, tl.uint32)
    k0 = seed.to(tl.uint32)
    k1 = (seed >> 32).to(tl.uint32)
    for _ in tl.static_range(10):
        p0 = c0.to(tl.uint64) * 0xD2511F53
        p2 = c2.to(tl.uint64) * 0xCD9E8D57
        c0, c1, c2, c3 = (
            (p2 >> 32).to(tl.uint32) ^ c1 ^ k0,
            p2.to(tl.uint32),
            (p0 >> 32).to(tl.uint32) ^ c3 ^ k1,
            p0.to(tl.uint32),
        )
        k0 = (k0 + 0x9E3779B9).to(tl.uint32)
        k1 = (k1 + 0xBB67AE85).to(tl.uint32)
    lane = (draw & 3).to(tl.uint32)
    return tl.where(lane == 0, c0, tl.where(lane == 1, c1, tl.where(lane == 2, c2, c3)))


@triton.jit
def e2m1_v1(x, word, SR: tl.constexpr):
    bits = x.to(tl.uint32, bitcast=True)
    sign = ((bits >> 28) & 8).to(tl.uint8)
    a = tl.minimum(tl.abs(x), 6.0)
    if SR:
        lower = tl.where(
            a >= 4,
            6,
            tl.where(
                a >= 3,
                5,
                tl.where(
                    a >= 2,
                    4,
                    tl.where(
                        a >= 1.5, 3, tl.where(a >= 1, 2, tl.where(a >= 0.5, 1, 0))
                    ),
                ),
            ),
        )
        lo = tl.where(
            lower == 6,
            4.0,
            tl.where(
                lower == 5, 3.0, tl.where(lower == 4, 2.0, lower.to(tl.float32) * 0.5)
            ),
        )
        q = tl.where(lower == 6, 1, tl.where(lower >= 4, 0, -1))
        delta = a - lo
        delta_bits = delta.to(tl.uint32, bitcast=True)
        exponent = ((delta_bits >> 23) & 255).to(tl.int32)
        n = (delta_bits & 0x7FFFFF) | tl.where(exponent == 0, 0, 0x800000)
        shift = tl.where(exponent == 0, -149, exponent - 150) - q + 32
        rshift = tl.minimum(tl.maximum(-shift, 1), 63)
        wide = n.to(tl.uint64)
        threshold = (wide >> rshift) + (
            (wide & ((tl.full((), 1, tl.uint64) << rshift) - 1)) != 0
        )
        threshold = tl.where(-shift > 63, (n != 0).to(tl.uint64), threshold)
        threshold = tl.where(
            shift >= 0, wide << tl.minimum(tl.maximum(shift, 0), 63), threshold
        )
        code = lower + (word.to(tl.uint64) < threshold).to(tl.int32)
        code = tl.where(a == 6.0, 7, code)
    else:
        code = (a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32)
        code += (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
        code += (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
        code += (a > 5.0).to(tl.int32)
    return code.to(tl.uint8) | sign


@triton.jit
def pack_codes_v1(codes, ROWS: tl.constexpr, COLS: tl.constexpr):
    low, high = tl.split(codes.reshape(ROWS, COLS // 2, 2))
    return low | (high << 4)


@triton.jit
def quantize_groups_v1(
    x, draw, seed, SR: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr
):
    """Quantize independent row groups and report their input poison flags."""
    groups = x.reshape(ROWS, COLS // 32, 32)
    poisoned = tl.max((~finite_v1(groups)).to(tl.int32), 2) != 0
    safe = tl.where(finite_v1(groups), groups, 0.0)
    raw = selector_v1(tl.max(tl.abs(safe), 2))
    normalized = normalize_v1(safe, raw[:, :, None]).reshape(ROWS, COLS)
    if SR:
        words = philox_element_v1(seed, draw)
    else:
        words = tl.full((ROWS, COLS), 0, tl.uint32)
    codes = e2m1_v1(normalized, words, SR).reshape(ROWS, COLS // 32, 32)
    codes = tl.where(poisoned[:, :, None], 0, codes).reshape(ROWS, COLS)
    return (
        pack_codes_v1(codes, ROWS, COLS),
        tl.where(poisoned, 255, raw).to(tl.uint8),
        poisoned,
    )


@triton.jit
def normalized_hadamard16_v1(x, ROWS: tl.constexpr):
    """Extend #6458's ordered butterfly with sticky intermediate overflow."""
    x = x * 0.25
    overflow = ~finite_v1(x)
    for stage in tl.static_range(4):
        h: tl.constexpr = 1 << stage
        pairs = x.reshape(ROWS, 16 // (2 * h), 2, h).permute(0, 1, 3, 2)
        upper, lower = tl.split(pairs)
        x = tl.join(upper + lower, upper - lower).permute(0, 1, 3, 2).reshape(ROWS, 16)
        overflow = overflow | (~finite_v1(x))
    return x, tl.max(overflow.to(tl.int32), 1) != 0
