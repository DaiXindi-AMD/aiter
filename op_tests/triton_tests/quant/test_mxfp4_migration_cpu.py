# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import os

import pytest
import torch

import aiter.ops.mxfp4 as compatibility_mxfp4
import aiter.ops.triton._triton_kernels.quant.quant_mxfp4 as mxfp4_kernels
import aiter.ops.triton.quant.mxfp4 as mxfp4
from aiter.ops.triton.utils.shuffle import (
    shuffle_scale_gemm,
    shuffle_scale_gemm_expanded,
)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    assert codes.shape[-1] % 2 == 0
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def test_top_level_module_is_a_compatibility_export():
    assert compatibility_mxfp4.convert_to_mxfp4 is mxfp4.convert_to_mxfp4
    assert compatibility_mxfp4.swizzle_mxfp4_scale is mxfp4.swizzle_mxfp4_scale


@pytest.fixture
def rebuild_sr_rounds():
    saved = os.environ.get("AITER_MXFP4_SR_PHILOX_ROUNDS")

    def rebuild(rounds):
        if rounds is None:
            os.environ.pop("AITER_MXFP4_SR_PHILOX_ROUNDS", None)
        else:
            os.environ["AITER_MXFP4_SR_PHILOX_ROUNDS"] = str(rounds)
        return importlib.reload(mxfp4_kernels)

    yield rebuild

    if saved is None:
        os.environ.pop("AITER_MXFP4_SR_PHILOX_ROUNDS", None)
    else:
        os.environ["AITER_MXFP4_SR_PHILOX_ROUNDS"] = saved
    importlib.reload(mxfp4_kernels)


def test_convert_from_mxfp4_cpu_nibble_order_and_scale():
    codes = torch.tensor(
        [[1, 2, 8, 15], [4, 5, 6, 7]], dtype=torch.uint8, device="cpu"
    ).repeat(1, 8)
    packed = _pack_codes(codes)
    scales = torch.tensor([[127], [128]], dtype=torch.uint8, device="cpu")

    out = mxfp4.convert_from_mxfp4(packed, scales, output_dtype=torch.float32)

    expected = torch.tensor(
        [[0.5, 1.0, -0.0, -6.0], [4.0, 6.0, 8.0, 12.0]],
        dtype=torch.float32,
        device="cpu",
    ).repeat(1, 8)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_convert_from_mxfp4_cpu_e8m0_endpoints():
    codes = torch.full((1, 32), 2, dtype=torch.uint8, device="cpu")
    packed = _pack_codes(codes)

    smallest = mxfp4.convert_from_mxfp4(
        packed,
        torch.tensor([[0]], dtype=torch.uint8, device="cpu"),
        output_dtype=torch.float32,
    )
    invalid = mxfp4.convert_from_mxfp4(
        packed,
        torch.tensor([[0xFF]], dtype=torch.uint8, device="cpu"),
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(
        smallest,
        torch.full_like(smallest, 2.0**-127),
        rtol=0,
        atol=0,
    )
    assert torch.isnan(invalid).all()


def test_convert_from_mxfp4_2d_cpu_expands_both_scale_axes():
    codes = torch.full((64, 64), 2, dtype=torch.uint8, device="cpu")
    packed = _pack_codes(codes)
    scales = torch.tensor(
        [[127, 128], [129, 130]], dtype=torch.uint8, device="cpu"
    )

    out = mxfp4.convert_from_mxfp4_2d(
        packed, scales, output_dtype=torch.float32
    )

    expected = scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    expected = torch.pow(2.0, expected - 127)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_public_mxfp4_apis_reject_nonstandard_block_size():
    x = torch.zeros((32, 32), dtype=torch.bfloat16, device="cpu")
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cpu")
    scales = torch.full((32, 1), 127, dtype=torch.uint8, device="cpu")
    scales_2d = torch.full((1, 1), 127, dtype=torch.uint8, device="cpu")
    sign = torch.ones(16, dtype=torch.float32, device="cpu")
    calls = {
        "convert_to_mxfp4": lambda: mxfp4.convert_to_mxfp4(x, block_size=16),
        "convert_to_mxfp4_2d": lambda: mxfp4.convert_to_mxfp4_2d(
            x, block_size=16
        ),
        "convert_from_mxfp4": lambda: mxfp4.convert_from_mxfp4(
            packed, scales, block_size=16
        ),
        "convert_from_mxfp4_2d": lambda: mxfp4.convert_from_mxfp4_2d(
            packed, scales_2d, block_size=16
        ),
        "hadamard_quant_mxfp4": lambda: mxfp4.hadamard_quant_mxfp4(
            x, sign, block_size=16
        ),
        "dual_layout_quant_mxfp4": lambda: mxfp4.dual_layout_quant_mxfp4(
            x, sign, block_size=16
        ),
        "dequant_hadamard_quant_mxfp4": lambda: (
            mxfp4.dequant_hadamard_quant_mxfp4(
                packed, scales, sign, block_size=16
            )
        ),
        "dequant_transpose_mxfp4": lambda: mxfp4.dequant_transpose_mxfp4(
            packed, scales, block_size=16
        ),
        "swizzle_expanded_mxfp4_scale": lambda: (
            mxfp4.swizzle_expanded_mxfp4_scale(scales_2d, block_size=16)
        ),
    }

    for call in calls.values():
        with pytest.raises(ValueError, match="block_size must be 32"):
            call()


def test_dequant_hadamard_quant_validates_scale_metadata():
    packed = torch.zeros((32, 32), dtype=torch.uint8, device="cpu")
    sign = torch.ones(16, dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="row-major scale shape"):
        mxfp4.dequant_hadamard_quant_mxfp4(
            packed, torch.zeros((1, 1), dtype=torch.uint8), sign
        )
    with pytest.raises(ValueError, match="swizzled scale shape"):
        mxfp4.dequant_hadamard_quant_mxfp4(
            packed,
            torch.zeros((32, 2), dtype=torch.uint8),
            sign,
            in_scale_swizzled=True,
        )
    with pytest.raises(TypeError, match="raw E8M0 bytes"):
        mxfp4.dequant_hadamard_quant_mxfp4(
            packed, torch.zeros((32, 2), dtype=torch.float32), sign
        )
    with pytest.raises(ValueError, match="physically contiguous"):
        mxfp4.dequant_hadamard_quant_mxfp4(
            packed, torch.zeros((2, 32), dtype=torch.uint8).t(), sign
        )


def test_hadamard_transform_validates_sign_dtype():
    x = torch.zeros((1, 16), dtype=torch.float32, device="cpu")
    sign = torch.ones(16, dtype=torch.int8, device="cpu")

    with pytest.raises(TypeError, match="sign_vector must have dtype"):
        mxfp4.hadamard_transform(x, sign, g=16)


def test_packed_transpose_cpu_round_trip():
    logical = torch.arange(24, dtype=torch.uint8, device="cpu").reshape(4, 6) & 0x0F
    packed = _pack_codes(logical)

    transposed = mxfp4.transpose_packed_fp4(packed)
    restored = mxfp4.transpose_packed_fp4(transposed)

    assert transposed.shape == (6, 2)
    assert torch.equal(restored, packed)


def test_packed_transpose_cpu_round_trip_through_shuffled_layout():
    generator = torch.Generator(device="cpu").manual_seed(7)
    packed = torch.randint(
        0, 256, (64, 32), dtype=torch.uint8, device="cpu", generator=generator
    )

    transposed_shuffled = mxfp4.transpose_packed_fp4(packed, shuffle_data=True)
    restored = mxfp4.transpose_packed_fp4(
        transposed_shuffled, in_shuffled=True
    )

    assert torch.equal(restored, packed)


def test_scale_swizzle_cpu_matches_existing_aiter_inverse():
    scales = (
        torch.arange(64 * 16, dtype=torch.int32, device="cpu")
        .to(torch.uint8)
        .reshape(64, 16)
    )

    shuffled = mxfp4.swizzle_mxfp4_scale(scales)
    restored = shuffled.view(64, 16)
    restored = restored.view(2, 2, 4, 16, 2, 2, 1)
    restored = restored.permute(0, 5, 3, 1, 4, 2, 6).contiguous().view(64, 16)

    assert shuffled.shape == (2, 512)
    assert torch.equal(restored, scales)


def test_canonical_scale_shuffle_matches_gfx950_reference_bytes():
    scales = (
        torch.arange(64 * 16, dtype=torch.int32, device="cpu")
        .to(torch.uint8)
        .reshape(64, 16)
    )

    canonical = shuffle_scale_gemm(
        scales,
        arch="gfx950",
        preshuffle_factor=32,
        scale_kwidth=8,
    )
    reference = scales.view(1, 2, 2, 16, 2, 2, 4, 1)
    reference = reference.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    reference = reference.view(2, 512)

    assert torch.equal(canonical.view(torch.uint8), reference.view(torch.uint8))


def test_expanded_2d_scale_swizzle_cpu_matches_explicit_expansion():
    scales = torch.arange(16, dtype=torch.uint8, device="cpu").reshape(2, 8)

    fused = mxfp4.swizzle_expanded_mxfp4_scale(scales, block_size=32)
    reference = mxfp4.swizzle_mxfp4_scale(
        scales.repeat_interleave(32, dim=0).contiguous()
    )
    fused_t = mxfp4.swizzle_expanded_mxfp4_scale(
        scales.transpose(0, 1).contiguous(), block_size=32, transpose=True
    )

    assert torch.equal(fused, reference)
    assert torch.equal(fused_t, reference)


def test_expanded_scale_swizzle_compatibility_matches_canonical_bytes():
    scales = torch.arange(16, dtype=torch.uint8, device="cpu").reshape(2, 8)

    compatibility = mxfp4.swizzle_expanded_mxfp4_scale(scales, block_size=32)
    canonical = shuffle_scale_gemm_expanded(
        scales,
        block_size=32,
        arch="gfx950",
        preshuffle_factor=32,
        scale_kwidth=8,
    )
    reference = shuffle_scale_gemm(
        scales.repeat_interleave(32, dim=0),
        arch="gfx950",
        preshuffle_factor=32,
        scale_kwidth=8,
    )

    assert torch.equal(canonical.view(torch.uint8), reference.view(torch.uint8))
    assert torch.equal(compatibility.view(torch.uint8), canonical.view(torch.uint8))


def test_hadamard_transform_cpu_is_normalized_and_involutory():
    x = torch.arange(16, dtype=torch.float32, device="cpu").reshape(2, 8)
    signs = torch.ones(4, device="cpu")

    rotated = mxfp4.hadamard_transform(x, signs, g=4)
    restored = mxfp4.hadamard_transform(rotated, signs, g=4)

    torch.testing.assert_close(rotated.norm(), x.norm(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(restored, x, rtol=1e-6, atol=1e-6)


def test_hadamard_transform_cpu_applies_sign_before_rotation():
    x = torch.arange(1, 9, dtype=torch.float32, device="cpu").reshape(2, 4)
    signs = torch.tensor([1.0, -1.0, -1.0, 1.0], device="cpu")
    hadamard = 0.5 * torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        device="cpu",
    )

    rotated = mxfp4.hadamard_transform(x, signs, g=4)
    expected = (x * signs) @ hadamard
    unsigned = x @ hadamard

    torch.testing.assert_close(rotated, expected, rtol=0, atol=0)
    assert not torch.equal(rotated, unsigned)


def test_standard_rtn_wrapper_delegates_to_existing_aiter_quant(monkeypatch):
    calls = []

    def fake_quant(x, *, swizzle_scale=False):
        calls.append((x, swizzle_scale))
        return (
            torch.zeros(
                (x.shape[0], x.shape[1] // 2), dtype=torch.uint8, device="cpu"
            ),
            torch.full(
                (x.shape[0], x.shape[1] // 32),
                127,
                dtype=torch.uint8,
                device="cpu",
            ),
        )

    monkeypatch.setattr(mxfp4, "_aiter_rtn_quant", fake_quant)
    x = torch.zeros((3, 64), dtype=torch.bfloat16, device="cpu")
    packed, scales = mxfp4.convert_to_mxfp4(x)

    assert len(calls) == 1 and torch.equal(calls[0][0], x)
    assert calls[0][1] is False
    assert packed.shape == (3, 32)
    assert scales.shape == (3, 2)


def test_standard_rtn_scale_swizzle_delegates_to_existing_aiter(monkeypatch):
    calls = []

    def fake_quant(x, *, swizzle_scale=False):
        calls.append((x, swizzle_scale))
        return (
            torch.zeros((32, 128), dtype=torch.uint8, device="cpu"),
            torch.zeros((1, 256), dtype=torch.uint8, device="cpu"),
        )

    monkeypatch.setattr(mxfp4, "_require_gfx950", lambda *_args: None)
    monkeypatch.setattr(mxfp4, "_aiter_rtn_quant", fake_quant)
    x = torch.zeros((32, 256), dtype=torch.bfloat16, device="cpu")

    packed, scales = mxfp4.convert_to_mxfp4(x, swizzle_scale=True)

    assert len(calls) == 1 and torch.equal(calls[0][0], x)
    assert calls[0][1] is True
    assert packed.shape == (32, 128)
    assert scales.shape == (1, 256)


def test_fp32_rtn_is_not_silently_delegated_through_bf16(monkeypatch):
    def fail_if_called(_):
        raise AssertionError("FP32 must not use the BF16 AITER delegation path")

    monkeypatch.setattr(mxfp4, "_aiter_rtn_quant", fail_if_called)
    x = torch.zeros((2, 32), dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError, match="requires gfx950"):
        mxfp4.convert_to_mxfp4(x)


def test_layout_support_predicates():
    assert mxfp4.mxfp4_scale_swizzle_supported(32, 8)
    assert not mxfp4.mxfp4_scale_swizzle_supported(16, 8)
    assert mxfp4.mxfp4_data_shuffle_supported(16, 32)
    assert not mxfp4.mxfp4_data_shuffle_supported(16, 16)


def test_sr_philox_round_default_is_documented(rebuild_sr_rounds):
    module = rebuild_sr_rounds(None)
    assert module.SR_PHILOX_ROUNDS == module.SR_PHILOX_ROUNDS_DEFAULT


def test_sr_philox_round_override_reaches_triton_constant(rebuild_sr_rounds):
    module = rebuild_sr_rounds(4)
    assert module.SR_PHILOX_ROUNDS == 4
    assert module.SR_PHILOX_ROUNDS_C.value == 4
