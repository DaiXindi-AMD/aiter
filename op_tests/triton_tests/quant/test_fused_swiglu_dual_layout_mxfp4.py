# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for fused BF16 SwiGLU and dual-layout MXFP4 output."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton import quant as quant_ops
from aiter.ops.triton.activation import swiglu_fwd_split
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm, shuffle_weight

_BLOCK_SIZE = 32
_HADAMARD_SIZE = 16
_FUSED_API_NAME = "fused_swiglu_dual_layout_mxfp4"

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_GFX950 = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx950",
    reason="fused SwiGLU dual-layout MXFP4 requires gfx950",
)


def _fused_api():
    return getattr(quant_ops, _FUSED_API_NAME)


def _run_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swizzle_scale: bool = False,
    shuffle_col: bool = False,
):
    return _fused_api()(
        gate,
        up,
        swizzle_scale=swizzle_scale,
        shuffle_col=shuffle_col,
    )


def _eager_double_bf16_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Apply the two BF16 rounding cuts with the PyTorch SiLU reference."""
    silu_bf16 = F.silu(gate.float()).to(torch.bfloat16)
    return (silu_bf16.float() * up.float()).to(torch.bfloat16)


def _single_round_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


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
    return matrix / (_HADAMARD_SIZE**0.5)


def _h16_transposed(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    blocks = x.T.float().reshape(cols, rows // _HADAMARD_SIZE, _HADAMARD_SIZE)
    return (blocks @ _normalized_hadamard16(x.device)).reshape(cols, rows)


def _canonical_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # The current production split path uses AITER's exp2 SiLU approximation.
    # Reuse its public wrapper so this test isolates the new fusion and layouts.
    activation = swiglu_fwd_split(gate, up)
    row_packed, row_scale = dynamic_mxfp4_quant(activation)
    col_packed, col_scale = dynamic_mxfp4_quant(_h16_transposed(activation))
    return activation, row_packed, row_scale, col_packed, col_scale


def _layout_reference(
    canonical: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    swizzle_scale: bool,
    shuffle_col: bool,
):
    activation, row_packed, row_scale, col_packed, col_scale = canonical
    if swizzle_scale:
        row_scale = shuffle_scale_gemm(
            row_scale,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
        )
        col_scale = shuffle_scale_gemm(
            col_scale,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
        )
    if shuffle_col:
        col_packed = shuffle_weight(
            col_packed,
            layout=(16, 16),
            arch="gfx950",
        )
    return activation, row_packed, row_scale, col_packed, col_scale


def _assert_exact_outputs(actual, expected) -> None:
    names = ("activation", "row_packed", "row_scale", "col_packed", "col_scale")
    assert len(actual) == len(expected) == len(names)
    for name, result, reference in zip(names, actual, expected):
        assert result.shape == reference.shape, (
            f"{name} shape mismatch: {tuple(result.shape)} != "
            f"{tuple(reference.shape)}"
        )
        assert (
            result.dtype == reference.dtype
        ), f"{name} dtype mismatch: {result.dtype} != {reference.dtype}"
        assert result.is_contiguous(), f"{name} must be contiguous"
        torch.testing.assert_close(
            result,
            reference,
            atol=0,
            rtol=0,
            msg=lambda message: f"{name} differs from production reference: {message}",
        )


def test_fused_swiglu_dual_layout_mxfp4_is_publicly_exported():
    assert callable(_fused_api())


@_GFX950
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 4096), id="qwen3_small_rows_d4096"),
        pytest.param((64, 256), id="m64_config_bucket"),
        pytest.param((96, 256), id="m96_masked_m128_bucket"),
        pytest.param((160, 256), id="m160_masked_default_bucket"),
        pytest.param((256, 12288), id="qwen3_medium_rows_d12288"),
    ],
)
def test_fused_swiglu_dual_layout_mxfp4_matches_canonical_reference(shape):
    torch.manual_seed(20260921)
    gate = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(gate)

    actual = _run_fused(gate, up)
    expected = _canonical_reference(gate, up)

    _assert_exact_outputs(actual, expected)
    rows, cols = shape
    assert actual[0].shape == (rows, cols)
    assert actual[1].shape == (rows, cols // 2)
    assert actual[2].shape == (rows, cols // _BLOCK_SIZE)
    assert actual[3].shape == (cols, rows // 2)
    assert actual[4].shape == (cols, rows // _BLOCK_SIZE)


@_GFX950
@pytest.mark.parametrize(
    "swizzle_scale,shuffle_col",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=("canonical", "col-shuffle", "scale-swizzle", "swizzle-and-shuffle"),
)
def test_fused_swiglu_dual_layout_mxfp4_layout_combinations(
    swizzle_scale: bool,
    shuffle_col: bool,
):
    torch.manual_seed(73)
    gate = torch.randn((256, 4096), dtype=torch.bfloat16, device="cuda") * 3
    up = torch.randn_like(gate) * 2
    canonical = _canonical_reference(gate, up)
    expected = _layout_reference(
        canonical,
        swizzle_scale=swizzle_scale,
        shuffle_col=shuffle_col,
    )

    actual = _run_fused(
        gate,
        up,
        swizzle_scale=swizzle_scale,
        shuffle_col=shuffle_col,
    )

    _assert_exact_outputs(actual, expected)
    if swizzle_scale:
        assert actual[2].shape == (gate.shape[0] // 32, gate.shape[1])
        assert actual[4].shape == (gate.shape[1] // 32, gate.shape[0])
    if shuffle_col:
        assert getattr(actual[3], "is_shuffled", False)


@_GFX950
def test_fused_swiglu_dual_layout_mxfp4_preserves_bf16_rounding_cuts():
    gate_values = torch.tensor(
        [-1.15625, -0.43359375, 0.30859375, 1.1171875],
        dtype=torch.bfloat16,
        device="cuda",
    )
    up_values = torch.tensor(
        [-5.3125, 0.78125, -1.15625, -1.1171875],
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = gate_values.repeat(32, 64)
    up = up_values.repeat(32, 64)

    eager_double_rounded = _eager_double_bf16_swiglu(gate, up)
    single_rounded = _single_round_swiglu(gate, up)
    assert not torch.equal(eager_double_rounded, single_rounded)

    actual = _run_fused(gate, up)
    expected = _canonical_reference(gate, up)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[0], eager_double_rounded, atol=1e-3, rtol=2e-3)
    _assert_exact_outputs(actual, expected)


@_GFX950
def test_fused_swiglu_dual_layout_mxfp4_rtn_edge_values():
    """Cover zero, E2M1 tie points, saturation, and raw-zero E8M0 scales."""
    values = torch.tensor(
        [
            0.0,
            0.25,
            -0.25,
            0.5,
            -0.5,
            0.75,
            -0.75,
            1.0,
            -1.0,
            1.25,
            -1.25,
            1.5,
            -1.5,
            1.75,
            -1.75,
            2.0,
            -2.0,
            2.5,
            -2.5,
            3.0,
            -3.0,
            3.5,
            -3.5,
            4.0,
            -4.0,
            5.0,
            -5.0,
            6.0,
            -6.0,
            8.0,
            -8.0,
            0.0,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    tiny = torch.full(
        (32,),
        torch.finfo(torch.bfloat16).tiny,
        dtype=torch.bfloat16,
        device="cuda",
    )
    row = torch.cat(
        (values, torch.zeros(32, device="cuda", dtype=torch.bfloat16), tiny)
    )
    row = row.repeat(3)[:256]
    gate = torch.full((32, 256), 1.28125, dtype=torch.bfloat16, device="cuda")
    up = row.repeat(32, 1)

    actual = _run_fused(gate, up)
    expected = _canonical_reference(gate, up)

    _assert_exact_outputs(actual, expected)


def test_fused_swiglu_dual_layout_mxfp4_validates_input_contract():
    good = torch.empty((32, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="2-D"):
        _run_fused(good.unsqueeze(0), good)
    with pytest.raises(TypeError, match="torch.bfloat16"):
        _run_fused(good.float(), good.float())
    with pytest.raises(ValueError, match="matching shapes"):
        _run_fused(good, torch.empty((64, 256), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        _run_fused(good.T, good.T)
    with pytest.raises(ValueError, match="non-zero"):
        _run_fused(good[:0], good[:0])
    with pytest.raises(ValueError, match="divisible by 32"):
        bad_width = torch.empty((32, 240), dtype=torch.bfloat16)
        _run_fused(bad_width, bad_width)
    with pytest.raises(TypeError, match="swizzle_scale must be bool"):
        _run_fused(good, good, swizzle_scale=1)
    with pytest.raises(TypeError, match="shuffle_col must be bool"):
        _run_fused(good, good, shuffle_col=1)
    with pytest.raises(ValueError, match="CUDA"):
        _run_fused(good, good)


@_GFX950
def test_fused_swiglu_dual_layout_mxfp4_validates_layout_constraints():
    gate = torch.empty((256, 224), dtype=torch.bfloat16, device="cuda")
    up = torch.empty_like(gate)
    with pytest.raises(ValueError, match="row scales.*tile evenly"):
        _run_fused(gate, up, swizzle_scale=True)

    gate = torch.empty((32, 256), dtype=torch.bfloat16, device="cuda")
    up = torch.empty_like(gate)
    with pytest.raises(ValueError, match="B shuffle"):
        _run_fused(gate, up, shuffle_col=True)


@_CUDA
def test_fused_swiglu_dual_layout_mxfp4_requires_gfx950():
    if arch_info.get_arch() == "gfx950":
        pytest.skip("gfx950 exercises the supported path")
    gate = torch.empty((256, 256), dtype=torch.bfloat16, device="cuda")
    up = torch.empty_like(gate)
    with pytest.raises(RuntimeError, match="gfx950"):
        _run_fused(gate, up)
