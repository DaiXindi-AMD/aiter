# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for dual-layout MXFP4 quantization and fused SwiGLU backward."""

import math
import struct

import pytest
import torch

from aiter.ops.triton import quant as quant_ops
from aiter.ops.triton.activation import swiglu_bwd_split
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.quant.fused_swiglu_dual_layout_mxfp4 import (
    _MAX_PHILOX_COUNTER,
    _PHILOX_COLUMN_OFFSET,
    _philox_streams,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm, shuffle_weight

_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_DUAL_API_NAME = "dual_layout_quant_mxfp4"
_BWD_API_NAME = "fused_swiglu_bwd_dual_layout_mxfp4"
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_GFX950 = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx950",
    reason="dual-layout MXFP4 requires gfx950",
)


def _dual_api():
    return getattr(quant_ops, _DUAL_API_NAME)


def _bwd_api():
    return getattr(quant_ops, _BWD_API_NAME)


def _normalized_hadamard16(device: torch.device) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < _HADAMARD_SIZE:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * 0.25


def _h16_transposed(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    blocks = x.T.reshape(cols, rows // _HADAMARD_SIZE, _HADAMARD_SIZE)
    return (blocks.float() @ _normalized_hadamard16(x.device)).reshape(cols, rows)


def _h16_transposed_cpu(x: torch.Tensor) -> torch.Tensor:
    """Apply normalized H16 with scalar CPU arithmetic for an independent oracle."""
    source = x.detach().cpu().float()
    rows, cols = source.shape
    result = torch.empty((cols, rows), dtype=torch.float32)
    for col in range(cols):
        for group_start in range(0, rows, _HADAMARD_SIZE):
            for output_lane in range(_HADAMARD_SIZE):
                total = 0.0
                for input_lane in range(_HADAMARD_SIZE):
                    parity = (output_lane & input_lane).bit_count() & 1
                    sign = -1.0 if parity else 1.0
                    total += float(source[group_start + input_lane, col]) * sign
                result[col, group_start + output_lane] = total * 0.25
    return result


def _e8m0_even_byte(amax: float) -> int:
    """Encode a non-negative FP32 maximum with the kernel's EVEN scale rule."""
    if amax < 0.0 or not math.isfinite(amax):
        raise ValueError("amax must be finite and non-negative")
    bits = struct.unpack("<I", struct.pack("<f", float(amax)))[0]
    rounded_bits = (bits + 0x200000) & 0xFF800000
    rounded = struct.unpack("<f", struct.pack("<I", rounded_bits))[0]
    exponent = -127 if rounded == 0.0 else math.floor(math.log2(rounded)) - 2
    return max(-127, min(127, exponent)) + 127


def _fp4_rtn_code(value: float) -> int:
    """Encode one normalized scalar as E2M1 with round-to-nearest-even."""
    magnitude = abs(value)
    code = min(
        range(len(_FP4_MAGNITUDES)),
        key=lambda index: (abs(magnitude - _FP4_MAGNITUDES[index]), index & 1),
    )
    return code | (0x8 if value < 0.0 else 0)


def _mxfp4_rtn_cpu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a CPU matrix without calling AITER, Triton, or torch matmul."""
    source = x.detach().cpu().float()
    rows, cols = source.shape
    assert cols % _BLOCK_SIZE == 0
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    scales = torch.empty((rows, cols // _BLOCK_SIZE), dtype=torch.uint8)
    for row in range(rows):
        for block_start in range(0, cols, _BLOCK_SIZE):
            block = source[row, block_start : block_start + _BLOCK_SIZE]
            scale_byte = _e8m0_even_byte(float(block.abs().max()))
            scales[row, block_start // _BLOCK_SIZE] = scale_byte
            scale = math.ldexp(1.0, scale_byte - 127)
            for pair in range(_BLOCK_SIZE // 2):
                low = _fp4_rtn_code(float(block[2 * pair]) / scale)
                high = _fp4_rtn_code(float(block[2 * pair + 1]) / scale)
                packed[row, block_start // 2 + pair] = low | (high << 4)
    return packed, scales


def _canonicalize_signed_zero(packed: torch.Tensor) -> torch.Tensor:
    """Treat positive and negative FP4 zero as the same inactive value."""
    low = packed & 0x0F
    high = packed >> 4
    low = torch.where((low & 0x07) == 0, 0, low)
    high = torch.where((high & 0x07) == 0, 0, high)
    return low | (high << 4)


def _layout_reference(
    x: torch.Tensor,
    *,
    swizzle_scale: bool,
    shuffle_col: bool,
):
    row, row_scale = dynamic_mxfp4_quant(x)
    col, col_scale = dynamic_mxfp4_quant(_h16_transposed(x))
    if swizzle_scale:
        row_scale = shuffle_scale_gemm(
            row_scale, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
        col_scale = shuffle_scale_gemm(
            col_scale, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
    if shuffle_col:
        col = shuffle_weight(col, layout=(16, 16), arch="gfx950")
    return row, row_scale, col, col_scale


def _split_reference(
    grad: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    use_sr: bool,
    gate_seed: int | None,
    gate_offset: int | None,
    up_seed: int | None,
    up_offset: int | None,
    swizzle_scale: bool,
):
    dgate, dup = swiglu_bwd_split(grad, gate, up)
    dgate_layout = _dual_api()(
        dgate,
        use_sr=use_sr,
        philox_seed=gate_seed,
        philox_offset=gate_offset,
        swizzle_scale=swizzle_scale,
    )
    dup_layout = _dual_api()(
        dup,
        use_sr=use_sr,
        philox_seed=up_seed,
        philox_offset=up_offset,
        swizzle_scale=swizzle_scale,
    )
    return dgate, dup, *dgate_layout, *dup_layout


def _assert_exact(actual, expected) -> None:
    names = (
        "dgate",
        "dup",
        "dgate_row",
        "dgate_row_scale",
        "dgate_col",
        "dgate_col_scale",
        "dup_row",
        "dup_row_scale",
        "dup_col",
        "dup_col_scale",
    )
    assert len(actual) == len(expected) == len(names)
    for name, result, reference in zip(names, actual, expected):
        assert result.shape == reference.shape, f"{name} shape mismatch"
        assert result.dtype == reference.dtype, f"{name} dtype mismatch"
        assert result.is_contiguous(), f"{name} must be contiguous"
        torch.testing.assert_close(
            result,
            reference,
            atol=0,
            rtol=0,
            msg=lambda message: f"{name} mismatch: {message}",
        )


def _dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed_u8 = packed.view(torch.uint8)
    codes = torch.empty(
        (*packed_u8.shape[:-1], packed_u8.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[..., 0::2] = packed_u8 & 0x0F
    codes[..., 1::2] = packed_u8 >> 4
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )[codes.long()]
    scale_values = torch.ldexp(
        torch.ones(scales.shape, dtype=torch.float32, device=scales.device),
        scales.to(torch.int32) - 127,
    )
    return values * scale_values.repeat_interleave(_BLOCK_SIZE, dim=-1)


def _snr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    signal = torch.linalg.vector_norm(reference.float())
    noise = torch.linalg.vector_norm(reference.float() - actual.float())
    return float(
        20 * torch.log10(signal / noise.clamp_min(torch.finfo(torch.float32).tiny))
    )


def test_dual_layout_and_backward_apis_are_publicly_exported():
    assert callable(_dual_api())
    assert callable(_bwd_api())


def test_philox_stream_contract_uses_production_column_offset():
    M, N = 64, 256
    seed, row_offset, col_offset = _philox_streams(M, N, True, 1234, 77, name="test")
    assert seed == 1234
    assert row_offset == 77
    assert col_offset == 77 + _PHILOX_COLUMN_OFFSET

    assert _philox_streams(M, N, False, None, None, name="test") == (0, 0, 0)
    with pytest.raises(ValueError, match="required"):
        _philox_streams(M, N, True, None, None, name="test")
    with pytest.raises(TypeError, match="integers"):
        _philox_streams(M, N, True, 1.5, 0, name="test")
    with pytest.raises(ValueError, match="only valid"):
        _philox_streams(M, N, False, 1, 0, name="test")

    counters = M * N // 8
    last_valid = _MAX_PHILOX_COUNTER - counters + 1 - _PHILOX_COLUMN_OFFSET
    _philox_streams(M, N, True, 1, last_valid, name="test")
    with pytest.raises(ValueError, match="leave room"):
        _philox_streams(M, N, True, 1, last_valid + 1, name="test")


@_GFX950
@pytest.mark.parametrize(
    "swizzle_scale,shuffle_col",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=("canonical", "col-shuffle", "scale-swizzle", "both"),
)
def test_dual_layout_quant_mxfp4_rtn_matches_reference(
    swizzle_scale: bool,
    shuffle_col: bool,
):
    torch.manual_seed(20260928)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda") * 2
    actual = _dual_api()(
        x,
        use_sr=False,
        swizzle_scale=swizzle_scale,
        shuffle_col=shuffle_col,
    )
    expected = _layout_reference(
        x, swizzle_scale=swizzle_scale, shuffle_col=shuffle_col
    )

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
    if swizzle_scale:
        assert getattr(actual[1], "_mxfp4_scale_swizzled", False)
        assert getattr(actual[3], "_mxfp4_scale_swizzled", False)
    if shuffle_col:
        assert getattr(actual[2], "_mxfp4_data_shuffled", False)
        assert getattr(actual[2], "is_shuffled", False)


@_GFX950
@pytest.mark.parametrize(
    "rows,swizzle_scale",
    [(32, False), (64, False), (96, False), (128, False), (256, False), (256, True)],
)
def test_fused_backward_rtn_matches_split_plus_dual_layout(
    rows: int, swizzle_scale: bool
):
    shape = (rows, 256)
    torch.manual_seed(73)
    grad = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(grad)
    up = torch.randn_like(grad)

    actual = _bwd_api()(grad, gate, up, use_sr=False, swizzle_scale=swizzle_scale)
    expected = _split_reference(
        grad,
        gate,
        up,
        use_sr=False,
        gate_seed=None,
        gate_offset=None,
        up_seed=None,
        up_offset=None,
        swizzle_scale=swizzle_scale,
    )
    _assert_exact(actual, expected)


@_GFX950
@pytest.mark.parametrize("rows", [32, 64, 256, 512])
def test_fused_backward_sr_matches_two_production_quantizers_across_tiles(rows: int):
    torch.manual_seed(11)
    grad = torch.randn((rows, 256), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(grad)
    up = torch.randn_like(grad)
    kwargs = dict(
        use_sr=True,
        gate_philox_seed=101,
        gate_philox_offset=1009,
        up_philox_seed=211,
        up_philox_offset=2003,
    )

    actual = _bwd_api()(grad, gate, up, **kwargs)
    expected = _split_reference(
        grad,
        gate,
        up,
        use_sr=True,
        gate_seed=kwargs["gate_philox_seed"],
        gate_offset=kwargs["gate_philox_offset"],
        up_seed=kwargs["up_philox_seed"],
        up_offset=kwargs["up_philox_offset"],
        swizzle_scale=False,
    )
    _assert_exact(actual, expected)


def test_cpu_mxfp4_oracle_known_encodings():
    values = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert [_e8m0_even_byte(value) for value in values] == [
        0x00,
        0x7B,
        0x7C,
        0x7D,
        0x7E,
        0x7F,
        0x80,
    ]
    assert [_e8m0_even_byte(value) for value in (1.7421875, 1.75, 1.7578125)] == [
        0x7D,
        0x7E,
        0x7E,
    ]

    ties = (
        0.25,
        0.75,
        1.25,
        1.75,
        2.5,
        3.5,
        5.0,
        0.0,
        -0.25,
        -0.75,
        -1.25,
        -1.75,
        -2.5,
        -3.5,
        -5.0,
        0.0,
    )
    packed = [
        _fp4_rtn_code(ties[index]) | (_fp4_rtn_code(ties[index + 1]) << 4)
        for index in range(0, len(ties), 2)
    ]
    assert packed == [0x20, 0x42, 0x64, 0x06, 0xA8, 0xCA, 0xEC, 0x0E]


@_GFX950
def test_fused_backward_rtn_matches_sparse_cpu_h16_oracle():
    desired_dgate = torch.zeros((32, 32), dtype=torch.bfloat16, device="cuda")
    desired_dup = torch.zeros_like(desired_dgate)
    for lane in range(_HADAMARD_SIZE):
        desired_dgate[lane, lane] = 4.0
        desired_dgate[_HADAMARD_SIZE + lane, lane] = 2.0
        col = _HADAMARD_SIZE + lane
        desired_dup[lane, col] = 4.0
        desired_dup[_HADAMARD_SIZE + lane, col] = 2.0

    grad = torch.zeros_like(desired_dgate)
    gate = torch.zeros_like(desired_dgate)
    up = torch.zeros_like(desired_dgate)
    dgate_mask = desired_dgate != 0
    dup_mask = desired_dup != 0
    up[dgate_mask] = 1.0
    grad[dgate_mask] = 2.0 * desired_dgate[dgate_mask]
    gate[dup_mask] = 16.0
    grad[dup_mask] = desired_dup[dup_mask] / 16.0

    actual = _bwd_api()(grad, gate, up, use_sr=False)
    torch.testing.assert_close(actual[0], desired_dgate, atol=0, rtol=0)
    torch.testing.assert_close(actual[1], desired_dup, atol=0, rtol=0)

    for source, packed, scales in (
        (desired_dgate, actual[2], actual[3]),
        (_h16_transposed_cpu(desired_dgate), actual[4], actual[5]),
        (desired_dup, actual[6], actual[7]),
        (_h16_transposed_cpu(desired_dup), actual[8], actual[9]),
    ):
        expected_packed, expected_scales = _mxfp4_rtn_cpu(source)
        torch.testing.assert_close(scales.cpu(), expected_scales, atol=0, rtol=0)
        torch.testing.assert_close(
            _canonicalize_signed_zero(packed.cpu()),
            _canonicalize_signed_zero(expected_packed),
            atol=0,
            rtol=0,
        )


@_GFX950
def test_fused_backward_preserves_split_bf16_rounding_cuts():
    gate_value = torch.tensor(0.0023040771484375, dtype=torch.bfloat16, device="cuda")
    up_value = torch.tensor(1.9375, dtype=torch.bfloat16, device="cuda")
    grad_value = torch.tensor(-0.4375, dtype=torch.bfloat16, device="cuda")
    gate = gate_value.expand(32, 32).contiguous()
    up = up_value.expand_as(gate).contiguous()
    grad = grad_value.expand_as(gate).contiguous()

    expected_dgate, expected_dup = swiglu_bwd_split(grad, gate, up)
    actual = _bwd_api()(grad, gate, up, use_sr=False)
    torch.testing.assert_close(actual[0], expected_dgate, atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected_dup, atol=0, rtol=0)


@_GFX950
def test_fused_backward_sr_is_reproducible_and_branch_isolated():
    torch.manual_seed(19)
    grad = torch.randn((64, 256), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(grad)
    up = torch.randn_like(grad)
    kwargs = dict(
        use_sr=True,
        gate_philox_seed=101,
        gate_philox_offset=1009,
        up_philox_seed=211,
        up_philox_offset=2003,
    )
    first = _bwd_api()(grad, gate, up, **kwargs)
    repeated = _bwd_api()(grad, gate, up, **kwargs)
    _assert_exact(first, repeated)

    changed_gate = _bwd_api()(grad, gate, up, **{**kwargs, "gate_philox_seed": 103})
    torch.testing.assert_close(first[0], changed_gate[0], atol=0, rtol=0)
    torch.testing.assert_close(first[1], changed_gate[1], atol=0, rtol=0)
    assert not torch.equal(first[2], changed_gate[2])
    assert not torch.equal(first[4], changed_gate[4])
    for index in (3, 5, 6, 7, 8, 9):
        torch.testing.assert_close(first[index], changed_gate[index], atol=0, rtol=0)

    changed_up = _bwd_api()(grad, gate, up, **{**kwargs, "up_philox_offset": 2017})
    for index in (0, 1, 2, 3, 4, 5, 7, 9):
        torch.testing.assert_close(first[index], changed_up[index], atol=0, rtol=0)
    assert not torch.equal(first[6], changed_up[6])
    assert not torch.equal(first[8], changed_up[8])


@_GFX950
def test_fused_backward_sr_dequant_snr_and_statistical_bias():
    torch.manual_seed(29)
    grad = torch.randn((32, 256), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(grad)
    up = torch.randn_like(grad)
    dgate, dup = swiglu_bwd_split(grad, gate, up)
    references = (dgate, _h16_transposed(dgate), dup, _h16_transposed(dup))
    accumulated_error = torch.zeros((), dtype=torch.float64, device="cuda")
    accumulated_count = 0

    for sample in range(16):
        output = _bwd_api()(
            grad,
            gate,
            up,
            use_sr=True,
            gate_philox_seed=1000 + sample,
            gate_philox_offset=2000 + sample * 17,
            up_philox_seed=3000 + sample,
            up_philox_offset=4000 + sample * 19,
        )
        dequantized = (
            _dequantize(output[2], output[3]),
            _dequantize(output[4], output[5]),
            _dequantize(output[6], output[7]),
            _dequantize(output[8], output[9]),
        )
        for reference, actual in zip(references, dequantized):
            assert _snr(reference, actual) >= 8.0
            accumulated_error += (actual - reference.float()).double().sum()
            accumulated_count += reference.numel()

    mean_bias = accumulated_error.abs() / accumulated_count
    reference_scale = torch.cat([value.flatten() for value in references]).abs().mean()
    assert float(mean_bias) <= 0.03 * float(reference_scale) + 1e-4


def test_fused_backward_validates_input_and_rng_contract():
    good = torch.empty((32, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="2-D"):
        _bwd_api()(
            good.unsqueeze(0), good.unsqueeze(0), good.unsqueeze(0), use_sr=False
        )
    with pytest.raises(TypeError, match="torch.bfloat16"):
        _bwd_api()(good.float(), good.float(), good.float(), use_sr=False)
    with pytest.raises(ValueError, match="matching shapes"):
        _bwd_api()(
            good, good, torch.empty((64, 256), dtype=torch.bfloat16), use_sr=False
        )
    with pytest.raises(ValueError, match="contiguous"):
        _bwd_api()(good.T, good.T, good.T, use_sr=False)
    with pytest.raises(ValueError, match="non-zero"):
        _bwd_api()(good[:0], good[:0], good[:0], use_sr=False)
    with pytest.raises(ValueError, match="divisible by 32"):
        bad = torch.empty((32, 240), dtype=torch.bfloat16)
        _bwd_api()(bad, bad, bad, use_sr=False)
    with pytest.raises(TypeError, match="swizzle_scale must be bool"):
        _bwd_api()(good, good, good, use_sr=False, swizzle_scale=1)
    with pytest.raises(ValueError, match="required"):
        _bwd_api()(good, good, good, use_sr=True)
    with pytest.raises(ValueError, match="CUDA"):
        _bwd_api()(good, good, good, use_sr=False)


@_GFX950
def test_fused_backward_validates_swizzle_constraints():
    grad = torch.empty((256, 224), dtype=torch.bfloat16, device="cuda")
    gate = torch.empty_like(grad)
    up = torch.empty_like(grad)
    with pytest.raises(ValueError, match="row scales.*tile evenly"):
        _bwd_api()(grad, gate, up, use_sr=False, swizzle_scale=True)


@_CUDA
def test_fused_backward_requires_gfx950():
    if arch_info.get_arch() == "gfx950":
        pytest.skip("gfx950 exercises the supported path")
    grad = torch.empty((32, 256), dtype=torch.bfloat16, device="cuda")
    gate = torch.empty_like(grad)
    up = torch.empty_like(grad)
    with pytest.raises(RuntimeError, match="gfx950"):
        _bwd_api()(grad, gate, up, use_sr=False)
