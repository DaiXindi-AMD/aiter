# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant import (
    dynamic_mxfp4_quant_2way,
    dynamic_mxfp4_quant_blockscale,
)
from aiter.ops.triton.utils.shuffle import shuffle_weight
from aiter.utility.fp4_utils import (
    e8m0_to_f32,
    f32_to_mx_e8m0_scale,
    f32_to_mxfp4,
)
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

_BLOCK_SIZE = 32
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _torch_mxfp4_quant_blockscale(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for deterministic 32x32 MXFP4 quantization."""
    x = x.float().cpu()
    rows, cols = x.shape
    tiles = (
        x.reshape(
            rows // _BLOCK_SIZE,
            _BLOCK_SIZE,
            cols // _BLOCK_SIZE,
            _BLOCK_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    amax = tiles.abs().amax(dim=(-2, -1))
    scales = f32_to_mx_e8m0_scale(
        amax,
        mode=MxScaleRoundModeInt.Even,
        dtype=MxDtypeInt.FP4_E2M1,
    ).view(torch.uint8)
    scales = scales.clamp_max(254)
    scale_f32 = e8m0_to_f32(scales).float()
    scaled = (tiles / scale_f32[:, :, None, None]).permute(0, 2, 1, 3)
    packed = f32_to_mxfp4(scaled.reshape(rows, cols)).view(torch.uint8)
    return packed, scales


def _valid_cpu_pair(cols: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((32, cols), dtype=torch.bfloat16),
        torch.empty((64, cols), dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (32,),
        (1, 32, 32),
    ],
)
def test_dynamic_mxfp4_quant_2way_validation_rejects_non_2d(shape):
    x0, _ = _valid_cpu_pair()
    x1 = torch.empty(shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="x1 must be 2-D"):
        dynamic_mxfp4_quant_2way(x0, x1)


def test_dynamic_mxfp4_quant_2way_validation_rejects_dtype():
    x0, x1 = _valid_cpu_pair()
    with pytest.raises(TypeError, match="x1 must have dtype torch.bfloat16"):
        dynamic_mxfp4_quant_2way(x0, x1.float())


def test_dynamic_mxfp4_quant_2way_validation_rejects_noncontiguous():
    x0, _ = _valid_cpu_pair()
    x1 = torch.empty((64, 64), dtype=torch.bfloat16).t()
    assert not x1.is_contiguous()
    with pytest.raises(ValueError, match="x1 must be contiguous"):
        dynamic_mxfp4_quant_2way(x0, x1)


@pytest.mark.parametrize(
    "shape,message",
    [
        ((0, 64), "dimensions must be non-zero"),
        ((48, 64), "rows=48 must be divisible by 32"),
        ((32, 48), "columns=48 must be divisible by 32"),
    ],
)
def test_dynamic_mxfp4_quant_2way_validation_rejects_shape(shape, message):
    _, x1 = _valid_cpu_pair()
    x0 = torch.empty(shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        dynamic_mxfp4_quant_2way(x0, x1)


def test_dynamic_mxfp4_quant_2way_validation_rejects_k_mismatch():
    x0, _ = _valid_cpu_pair(cols=32)
    _, x1 = _valid_cpu_pair(cols=64)
    with pytest.raises(ValueError, match="must have the same K"):
        dynamic_mxfp4_quant_2way(x0, x1)


def test_dynamic_mxfp4_quant_2way_validation_rejects_device_mismatch():
    x0, _ = _valid_cpu_pair()
    x1 = torch.empty((64, 64), dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="must be on the same device"):
        dynamic_mxfp4_quant_2way(x0, x1)


def test_dynamic_mxfp4_quant_2way_validation_rejects_cpu():
    x0, x1 = _valid_cpu_pair()
    with pytest.raises(ValueError, match="must be on a CUDA device"):
        dynamic_mxfp4_quant_2way(x0, x1)


def test_dynamic_mxfp4_quant_2way_validation_rejects_unshufflable_k():
    x0, x1 = _valid_cpu_pair(cols=32)
    with pytest.raises(ValueError, match="shuffle_data requires"):
        dynamic_mxfp4_quant_2way(x0, x1, shuffle_data=True)


def test_mxfp4_blockscale_two_source_boundary_is_exact_on_cpu():
    torch.manual_seed(11)
    x0 = torch.randn((32, 64), dtype=torch.bfloat16)
    x1 = torch.randn((64, 64), dtype=torch.bfloat16)

    packed, scales = _torch_mxfp4_quant_blockscale(torch.cat((x0, x1), dim=0))
    packed0, scales0 = _torch_mxfp4_quant_blockscale(x0)
    packed1, scales1 = _torch_mxfp4_quant_blockscale(x1)

    torch.testing.assert_close(packed, torch.cat((packed0, packed1)), atol=0, rtol=0)
    torch.testing.assert_close(scales, torch.cat((scales0, scales1)), atol=0, rtol=0)
    shuffled = shuffle_weight(packed, layout=(16, 16), arch="gfx950")
    shuffled_parts = torch.cat(
        (
            shuffle_weight(packed0, layout=(16, 16), arch="gfx950"),
            shuffle_weight(packed1, layout=(16, 16), arch="gfx950"),
        )
    )
    torch.testing.assert_close(shuffled, shuffled_parts, atol=0, rtol=0)


@_CUDA
@pytest.mark.parametrize("rows0,rows1,cols", [(32, 64, 64), (64, 96, 128)])
@pytest.mark.parametrize("shuffle_data", [False, True])
def test_dynamic_mxfp4_quant_2way_matches_cat_reference(
    rows0: int,
    rows1: int,
    cols: int,
    shuffle_data: bool,
):
    torch.manual_seed(20)
    x0 = torch.randn((rows0, cols), dtype=torch.bfloat16, device="cuda") * 4
    x1 = torch.randn((rows1, cols), dtype=torch.bfloat16, device="cuda") * 2
    concatenated = torch.cat((x0, x1), dim=0)

    packed, scales = dynamic_mxfp4_quant_2way(x0, x1, shuffle_data=shuffle_data)
    expected_packed, expected_scales = _torch_mxfp4_quant_blockscale(concatenated)
    if shuffle_data:
        expected_packed = shuffle_weight(
            expected_packed, layout=(16, 16), arch="gfx950"
        )

    assert packed.shape == (rows0 + rows1, cols // 2)
    assert scales.shape == (
        (rows0 + rows1) // _BLOCK_SIZE,
        cols // _BLOCK_SIZE,
    )
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.uint8
    assert packed.is_contiguous()
    assert scales.is_contiguous()
    assert getattr(packed, "is_shuffled", False) is shuffle_data
    torch.testing.assert_close(packed.cpu(), expected_packed, atol=0, rtol=0)
    torch.testing.assert_close(scales.cpu(), expected_scales, atol=0, rtol=0)

    single_packed, single_scales = dynamic_mxfp4_quant_blockscale(
        concatenated, shuffle_data=shuffle_data
    )
    torch.testing.assert_close(packed, single_packed, atol=0, rtol=0)
    torch.testing.assert_close(scales, single_scales, atol=0, rtol=0)


@_CUDA
def test_dynamic_mxfp4_quant_2way_edge_values_match_reference():
    x0 = torch.zeros((64, 128), dtype=torch.bfloat16, device="cuda")
    x1 = torch.zeros((32, 128), dtype=torch.bfloat16, device="cuda")

    scale_tie_values = (1.7421875, 1.75, 1.7578125)
    for tile_col, value in enumerate(scale_tie_values, start=1):
        x0[:32, tile_col * 32 : (tile_col + 1) * 32] = value

    payload_ties = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    x0[32, : payload_ties.numel()] = payload_ties
    x0[32, 8 : 8 + payload_ties.numel()] = -payload_ties
    x0[32, 32] = torch.finfo(torch.bfloat16).tiny
    x1[:32, :32] = torch.finfo(torch.bfloat16).max

    packed, scales = dynamic_mxfp4_quant_2way(x0, x1)
    expected_packed, expected_scales = _torch_mxfp4_quant_blockscale(
        torch.cat((x0, x1), dim=0)
    )

    packed_cpu = packed.cpu()
    scales_cpu = scales.cpu()
    torch.testing.assert_close(packed_cpu, expected_packed, atol=0, rtol=0)
    torch.testing.assert_close(scales_cpu, expected_scales, atol=0, rtol=0)

    assert scales_cpu[0, 0].item() == 0
    assert scales_cpu[2, 0].item() == 254
    assert torch.count_nonzero(scales_cpu == 0xFF).item() == 0
    assert scales_cpu[0, 1:].tolist() == [0x7D, 0x7E, 0x7E]
    assert scales_cpu[1, 0].item() == 0x7F
    torch.testing.assert_close(
        packed_cpu[32, :8],
        torch.tensor(
            [0x20, 0x42, 0x64, 0x06, 0xA8, 0xCA, 0xEC, 0x0E],
            dtype=torch.uint8,
        ),
        atol=0,
        rtol=0,
    )
