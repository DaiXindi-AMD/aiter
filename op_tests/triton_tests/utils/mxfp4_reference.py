# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Independent exact-rational test oracle, not an implementation dependency."""

from fractions import Fraction

LEVELS = tuple(Fraction(v, 2) for v in (0, 1, 2, 3, 4, 6, 8, 12))


def fp32_fraction(bits):
    exponent, fraction = (bits >> 23) & 255, bits & 0x7FFFFF
    if exponent == 255:
        raise ValueError("nonfinite")
    value = Fraction(fraction if exponent == 0 else fraction + (1 << 23))
    power = -149 if exponent == 0 else exponent - 150
    value *= Fraction(2) ** power
    return -value if bits >> 31 else value


def rn32_bits(value, negative_zero=False):
    sign = value < 0 or (value == 0 and negative_zero)
    a = abs(value)
    if not a:
        return int(sign) << 31
    exponent = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** exponent:
        exponent -= 1
    quantum = Fraction(2) ** max(-149, exponent - 23)
    scaled = a / quantum
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and integer & 1):
        integer += 1
    if integer == 1 << 24:
        integer >>= 1
        exponent += 1
    if exponent > 127:
        return (int(sign) << 31) | 0x7F800000
    if exponent < -126:
        magnitude = integer
    else:
        magnitude = ((exponent + 127) << 23) | (integer & 0x7FFFFF)
    return (int(sign) << 31) | magnitude


def selector_reference(bits):
    a = abs(fp32_fraction(bits))
    if a == 0:
        return 0
    # Enumerate normal binade thresholds independently of the production add/mask.
    if a >= Fraction(7, 4) * Fraction(2) ** 127:
        return 254
    exponent = -127
    for e in range(-126, 128):
        threshold = Fraction(7, 4) * Fraction(2) ** (e + 1)
        if a >= threshold:
            exponent = e
        else:
            break
    return max(0, min(254, exponent + 127))


def e2m1_reference(bits, word=None):
    x = fp32_fraction(bits)
    a = min(abs(x), LEVELS[-1])
    sign = 8 if bits >> 31 else 0
    if word is None:
        code = min(range(8), key=lambda i: (abs(a - LEVELS[i]), i & 1))
    elif a in LEVELS:
        code = LEVELS.index(a)
    else:
        lower = max(i for i, value in enumerate(LEVELS) if value < a)
        probability = (a - LEVELS[lower]) / (LEVELS[lower + 1] - LEVELS[lower])
        threshold = -(-(probability.numerator << 32) // probability.denominator)
        code = lower + (word < threshold)
    return code | sign


def philox_reference(seed, draw):
    mask = (1 << 32) - 1
    counter = draw // 4
    state = [counter & mask, counter >> 32, 0, 0]
    key = [seed & mask, seed >> 32]
    for _ in range(10):
        products = [0xD2511F53 * state[0], 0xCD9E8D57 * state[2]]
        state = [
            (products[1] >> 32) ^ state[1] ^ key[0],
            products[1] & mask,
            (products[0] >> 32) ^ state[3] ^ key[1],
            products[0] & mask,
        ]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return state[draw % 4]


def row_reference(bits, seed=None, base=0):
    result, scales = [], []
    for start in range(0, len(bits), 32):
        group = bits[start : start + 32]
        if any(((value >> 23) & 255) == 255 for value in group):
            scales.append(255)
            codes = [0] * 32
        else:
            amax = max(group, key=lambda b: abs(fp32_fraction(b))) & 0x7FFFFFFF
            raw = selector_reference(amax)
            scales.append(raw)
            codes = []
            for i, value in enumerate(group):
                normalized = rn32_bits(
                    fp32_fraction(value) / Fraction(2) ** (raw - 127), value >> 31
                )
                word = (
                    None if seed is None else philox_reference(seed, base + start + i)
                )
                codes.append(e2m1_reference(normalized, word))
        result.extend(codes[i] | (codes[i + 1] << 4) for i in range(0, 32, 2))
    return result, scales


def h16_reference(bits):
    stage = [rn32_bits(fp32_fraction(b) / 4, b >> 31) for b in bits]
    for h in (1, 2, 4, 8):
        following = stage[:]
        for start in range(0, 16, 2 * h):
            for j in range(h):
                ai, bi = start + j, start + j + h
                a, b = fp32_fraction(stage[ai]), fp32_fraction(stage[bi])
                following[ai] = rn32_bits(
                    a + b, (stage[ai] >> 31) and (stage[bi] >> 31)
                )
                following[bi] = rn32_bits(
                    a - b, (stage[ai] >> 31) and not (stage[bi] >> 31)
                )
        stage = following
    return stage
