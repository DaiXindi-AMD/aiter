# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import os

import pytest
import torch

import aiter.ops.triton._triton_kernels.quant.quant_mxfp4 as mxfp4_kernels
import aiter.ops.triton.quant.mxfp4 as mxfp4


BLOCK = 32
SR_DRAWS = 96


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return "gfx950" in getattr(properties, "gcnArchName", "")


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="migrated MXFP4 training kernels require gfx950"
)


def _randn(shape, *, dtype=torch.bfloat16):
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    return torch.randn(shape, device="cuda", dtype=dtype)


def _signs(g=16):
    return torch.where(
        torch.arange(g, device="cuda") % 2 == 0,
        torch.ones(g, device="cuda"),
        -torch.ones(g, device="cuda"),
    ).to(torch.bfloat16)


def _explicit_rht_quant(x, sign):
    rotated = mxfp4.hadamard_transform(x.float(), sign, g=sign.numel())
    return mxfp4.convert_to_mxfp4(rotated, use_sr=False)


@pytest.fixture
def rebuild_sr_rounds():
    saved = os.environ.get("AITER_MXFP4_SR_PHILOX_ROUNDS")

    def rebuild(rounds):
        os.environ["AITER_MXFP4_SR_PHILOX_ROUNDS"] = str(rounds)
        importlib.reload(mxfp4_kernels)
        return importlib.reload(mxfp4)

    yield rebuild

    if saved is None:
        os.environ.pop("AITER_MXFP4_SR_PHILOX_ROUNDS", None)
    else:
        os.environ["AITER_MXFP4_SR_PHILOX_ROUNDS"] = saved
    importlib.reload(mxfp4_kernels)
    importlib.reload(mxfp4)


def _sr_residual_std(draws=SR_DRAWS):
    x = _randn((256, BLOCK * 8))
    accumulator = torch.zeros_like(x, dtype=torch.float32)
    for draw in range(draws):
        packed, scales = mxfp4.convert_to_mxfp4(
            x,
            block_size=BLOCK,
            use_sr=True,
            philox_seed=1234 + draw,
            philox_offset=0,
        )
        accumulator += mxfp4.convert_from_mxfp4(
            packed, scales, output_dtype=torch.float32, block_size=BLOCK
        )
    residual = accumulator / draws - x.float()
    denominator = x.float().abs().mean().clamp_min(1e-6)
    return (residual.std() / denominator).item()


def test_convert_1d_fused_scale_swizzle_matches_aiter_rtn():
    x = _randn((64, 256))

    ref_q, ref_s = mxfp4.convert_to_mxfp4(x, use_sr=False)
    got_q, got_s = mxfp4.convert_to_mxfp4(
        x, use_sr=False, swizzle_scale=True
    )

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(
        got_s, mxfp4.swizzle_mxfp4_scale(ref_s), atol=0, rtol=0
    )


def test_convert_1d_sr_fixed_philox_only_changes_payload_rounding():
    x = _randn((64, 256))
    kwargs = dict(use_sr=True, philox_seed=1234, philox_offset=5678)

    q1, s1 = mxfp4.convert_to_mxfp4(x, **kwargs)
    q2, s2 = mxfp4.convert_to_mxfp4(x, **kwargs)
    q3, s3 = mxfp4.convert_to_mxfp4(
        x, use_sr=True, philox_seed=1234, philox_offset=5679
    )
    _, rtn_scales = mxfp4.convert_to_mxfp4(x, use_sr=False)

    torch.testing.assert_close(q1, q2, atol=0, rtol=0)
    torch.testing.assert_close(s1, s2, atol=0, rtol=0)
    torch.testing.assert_close(s1, s3, atol=0, rtol=0)
    torch.testing.assert_close(s1, rtn_scales, atol=0, rtol=0)
    assert not torch.equal(q1, q3), "Philox offset did not affect SR payloads"


@pytest.mark.parametrize("use_sr", [False, True])
def test_convert_1d_raw_zero_scale_matches_quantized_payload(use_sr):
    x = torch.full(
        (32, 32), 2.0**-125, dtype=torch.float32, device="cuda"
    )

    packed, scales = mxfp4.convert_to_mxfp4(
        x, use_sr=use_sr, philox_seed=1234, philox_offset=0
    )
    restored = mxfp4.convert_from_mxfp4(
        packed, scales, output_dtype=torch.float32
    )

    assert torch.count_nonzero(scales).item() == 0
    assert torch.all(packed == 0x66)
    torch.testing.assert_close(restored, x, atol=0, rtol=0)


def test_sr_dither_does_not_repeat_between_program_tiles():
    tile = _randn((64, 64))
    x = tile.repeat(2, 1)

    packed, scales = mxfp4.convert_to_mxfp4(
        x,
        block_size=BLOCK,
        use_sr=True,
        philox_seed=1234,
        philox_offset=0,
    )

    torch.testing.assert_close(scales[:64], scales[64:], atol=0, rtol=0)
    assert not torch.equal(packed[:64], packed[64:])


def test_sr_philox_round_floor_preserves_dither_quality(rebuild_sr_rounds):
    rebuild_sr_rounds(mxfp4_kernels.SR_PHILOX_ROUNDS_DEFAULT)
    default_std = _sr_residual_std()
    rebuild_sr_rounds(4)
    floor_std = _sr_residual_std()
    rebuild_sr_rounds(2)
    starved_std = _sr_residual_std()

    assert floor_std == pytest.approx(default_std, rel=0.15)
    assert starved_std > 2 * default_std


def test_convert_2d_is_transpose_invariant():
    x = _randn((64, 256), dtype=torch.float32)

    q, scales = mxfp4.convert_to_mxfp4_2d(x, use_sr=False)
    q_t, scales_t = mxfp4.convert_to_mxfp4_2d(
        x.t().contiguous(), use_sr=False
    )

    torch.testing.assert_close(
        q_t, mxfp4.transpose_packed_fp4(q), atol=0, rtol=0
    )
    torch.testing.assert_close(scales_t, scales.t().contiguous(), atol=0, rtol=0)
    dequant = mxfp4.convert_from_mxfp4_2d(q, scales, output_dtype=torch.float32)
    error = (x - dequant).float().pow(2).sum()
    snr = 10 * torch.log10(x.float().pow(2).sum() / error)
    assert snr.item() >= 4.0


@pytest.mark.parametrize("shape", [(64, 32), (6, 32), (4, 6)])
def test_packed_transpose_gpu_matches_cpu_reference_and_shuffle(shape):
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    ref = mxfp4.transpose_packed_fp4(packed.cpu())
    got = mxfp4.transpose_packed_fp4(packed)
    torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)

    if not mxfp4.mxfp4_data_shuffle_supported(
        packed.shape[1] * 2, packed.shape[0] // 2
    ):
        return

    ref_shuffled = mxfp4.transpose_packed_fp4(
        packed.cpu(), shuffle_data=True
    )
    got_shuffled = mxfp4.transpose_packed_fp4(packed, shuffle_data=True)

    torch.testing.assert_close(got_shuffled.cpu(), ref_shuffled, atol=0, rtol=0)
    torch.testing.assert_close(
        mxfp4.transpose_packed_fp4(got_shuffled, in_shuffled=True),
        packed,
        atol=0,
        rtol=0,
    )


def test_dual_layout_rtn_matches_decomposed_kernels():
    x = _randn((64, 256))
    sign = _signs()

    got = mxfp4.dual_layout_quant_mxfp4(
        x,
        sign,
        use_sr_row=False,
        use_sr_transposed=False,
    )
    row_q, row_s = mxfp4.convert_to_mxfp4(x, use_sr=False)
    col_q, col_s = _explicit_rht_quant(x.t().contiguous(), sign)

    for actual, expected in zip(got, (row_q, row_s, col_q, col_s)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_hadamard_quant_rtn_matches_explicit_rht():
    x = _randn((64, 256))
    sign = _signs()

    got_q, got_s = mxfp4.hadamard_quant_mxfp4(x, sign, use_sr=False)
    ref_q, ref_s = _explicit_rht_quant(x, sign)

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, ref_s, atol=0, rtol=0)


def test_fp32_hadamard_quant_accepts_noncontiguous_sign_vector():
    x = _randn((64, 256), dtype=torch.float32)
    sign_storage = torch.empty(32, dtype=torch.float32, device="cuda")
    sign_storage[::2] = _signs().float()
    sign = sign_storage[::2]
    assert not sign.is_contiguous()

    got_q, got_s = mxfp4.hadamard_quant_mxfp4(x, sign, use_sr=False)
    ref_q, ref_s = mxfp4.hadamard_quant_mxfp4(
        x, sign.contiguous(), use_sr=False
    )

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, ref_s, atol=0, rtol=0)


def test_bf16_rht_cache_invalidates_after_inplace_sign_update():
    x = _randn((64, 256))
    sign = torch.ones(16, dtype=torch.bfloat16, device="cuda")

    before_q, _ = mxfp4.hadamard_quant_mxfp4(x, sign, use_sr=False)
    sign[0].mul_(-1)
    got_q, got_s = mxfp4.hadamard_quant_mxfp4(x, sign, use_sr=False)
    ref_q, ref_s = _explicit_rht_quant(x, sign)

    assert not torch.equal(before_q, got_q)
    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, ref_s, atol=0, rtol=0)


def test_rht_and_dequant_inputs_must_share_a_device():
    x = _randn((32, 64), dtype=torch.float32)
    cpu_sign = torch.ones(16, dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="sign_vector must be on the same device"):
        mxfp4.hadamard_quant_mxfp4(x, cpu_sign, use_sr=False)

    packed = torch.zeros((32, 32), dtype=torch.uint8, device="cuda")
    scales = torch.full((32, 2), 127, dtype=torch.uint8, device="cpu")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="scales must be on the same device"):
        mxfp4.dequant_hadamard_quant_mxfp4(
            packed, scales, sign, use_sr=False
        )


@pytest.mark.parametrize("shape", [(64, 256), (6, 256)])
def test_dequant_transpose_matches_decomposed_reference(shape):
    x = _randn(shape)
    packed, scales = mxfp4.convert_to_mxfp4(x, use_sr=False)

    got = mxfp4.dequant_transpose_mxfp4(packed, scales)
    ref = mxfp4.convert_from_mxfp4(
        packed, scales, output_dtype=torch.bfloat16
    ).t().contiguous()

    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_dequant_transpose_preserves_e8m0_nan_semantics():
    codes = torch.full((2, 32), 2, dtype=torch.uint8, device="cuda")
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.tensor([[127], [0xFF]], dtype=torch.uint8, device="cuda")

    got = mxfp4.dequant_transpose_mxfp4(packed, scales, block_size=32)
    ref = mxfp4.convert_from_mxfp4(
        packed, scales, output_dtype=torch.bfloat16, block_size=32
    ).t().contiguous()

    torch.testing.assert_close(got, ref, atol=0, rtol=0, equal_nan=True)


def test_dequant_hadamard_quant_matches_two_pass_reference():
    x = _randn((64, 256))
    sign = _signs()
    packed, scales = mxfp4.convert_to_mxfp4(x, use_sr=False)

    intermediate = mxfp4.dequant_transpose_mxfp4(packed, scales)
    ref_q, ref_s = _explicit_rht_quant(intermediate, sign)
    got_q, got_s = mxfp4.dequant_hadamard_quant_mxfp4(
        packed, scales, sign, use_sr=False
    )

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, ref_s, atol=0, rtol=0)


def test_dequant_hadamard_quant_accepts_exact_swizzled_input_scale_layout():
    x = _randn((64, 256))
    sign = _signs()
    packed, scales = mxfp4.convert_to_mxfp4(x, use_sr=False)
    swizzled_scales = mxfp4.swizzle_mxfp4_scale(scales)

    ref_q, ref_s = mxfp4.dequant_hadamard_quant_mxfp4(
        packed, scales, sign, use_sr=False
    )
    got_q, got_s = mxfp4.dequant_hadamard_quant_mxfp4(
        packed,
        swizzled_scales,
        sign,
        use_sr=False,
        in_scale_swizzled=True,
    )

    torch.testing.assert_close(got_q, ref_q, atol=0, rtol=0)
    torch.testing.assert_close(got_s, ref_s, atol=0, rtol=0)


def test_scale_swizzles_match_cpu_reference():
    scales = torch.randint(0, 255, (64, 16), dtype=torch.uint8, device="cuda")
    got = mxfp4.swizzle_mxfp4_scale(scales)
    ref = mxfp4.swizzle_mxfp4_scale(scales.cpu())
    torch.testing.assert_close(got.cpu(), ref, atol=0, rtol=0)

    scales_2d = torch.randint(0, 255, (2, 8), dtype=torch.uint8, device="cuda")
    for transpose in (False, True):
        got_expanded = mxfp4.swizzle_expanded_mxfp4_scale(
            scales_2d, transpose=transpose
        )
        ref_expanded = mxfp4.swizzle_expanded_mxfp4_scale(
            scales_2d.cpu(), transpose=transpose
        )
        torch.testing.assert_close(
            got_expanded.cpu(), ref_expanded, atol=0, rtol=0
        )
