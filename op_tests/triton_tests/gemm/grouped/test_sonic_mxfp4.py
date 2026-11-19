# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical tests use the public packed-weight wrapper and independent FP4 decoding."""

import pytest
import torch

from aiter.ops.triton._triton_kernels.moe.sonicmoe.row_scale import row_scale_fp32
from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
    block_shape,
    family_for,
    grouped_mxfp4_weight_gemm,
    pack_grouped_mxfp4_weights,
    prepare_grouped_gemm_metadata,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def decode(payload, scales):
    low = (payload & 15).long()
    high = (payload >> 4).long()
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=payload.device
    )
    values = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


def decode_mxfp8(payload, scales):
    values = payload.to(torch.float32)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


def reference_activation(x, activation_format):
    """Decode the activation exactly the way the kernel will see it."""
    if activation_format == "bf16":
        return x.float()
    if activation_format == "mxfp4":
        return decode(*dynamic_mxfp4_quant(x))
    return decode_mxfp8(*dynamic_mxfp8_quant(x))


@pytest.mark.parametrize(
    "n,k", [(1536, 2048), (2048, 768), (2048, 1536), (768, 2048), (65, 96)]
)
@pytest.mark.parametrize("activation_format", ["bf16", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("multi_tile", [False, True])
def test_ragged_fwd_and_dgrad_shapes(n, k, activation_format, multi_tile):
    torch.manual_seed(1234)
    e = 32
    counts = (
        ((0, 65, 129, 1025) + tuple(i % 7 for i in range(e - 4)))
        if multi_tile
        else tuple(i % 7 for i in range(e))
    )
    x = torch.randn((sum(counts), k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((e, n, k), device="cuda", dtype=torch.bfloat16) * 0.02
    payload, scales = pack_grouped_mxfp4_weights(weight)
    decoded = decode(payload, scales)
    out = torch.full(
        (sum(counts), n), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    result = grouped_mxfp4_weight_gemm(
        x, payload, scales, counts, out, activation_format=activation_format
    )
    reference_x = reference_activation(x, activation_format)
    assert result is out
    refs = []
    offset = 0
    for expert, count in enumerate(counts):
        refs.append(reference_x[offset : offset + count] @ decoded[expert].T)
        offset += count
    reference = torch.cat(refs).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.002)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("activation_format", ["bf16", "mxfp4", "mxfp8"])
def test_production_shapes_share_one_metadata(activation_format):
    """The four F/D GEMMs of a Qwen3 MoE layer must resolve the same block shape.

    Lumen's module builds one offsets/block-map per layer and reuses it for F1/F2/
    D1/D2; that is only valid when every shape selects the same BLOCK_M/BLOCK_N.
    """
    family = family_for(activation_format)
    e, h, i = 32, 2048, 768
    shapes = ((2 * i, h), (h, i), (i, h), (h, 2 * i))
    blocks = {block_shape(family, 1024, n, k, e) for n, k in shapes}
    assert len(blocks) == 1, (family, blocks)


@pytest.mark.parametrize("activation_format", ["bf16", "mxfp4", "mxfp8"])
def test_prepared_metadata_is_bit_identical(activation_format):
    """Reusing one offsets/block-map build across GEMMs must not change the result."""
    torch.manual_seed(7)
    e, n, k = 32, 1536, 2048
    family = family_for(activation_format)
    counts = tuple([321, 0, 129, 1025] + [i * 7 % 11 for i in range(e - 4)])
    x = torch.randn((sum(counts), k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((e, n, k), device="cuda", dtype=torch.bfloat16) * 0.02
    payload, scales = pack_grouped_mxfp4_weights(weight)
    per_call = grouped_mxfp4_weight_gemm(
        x, payload, scales, counts, activation_format=activation_format
    )
    block_m, block_n = block_shape(family, max(counts), n, k, e)
    metadata = prepare_grouped_gemm_metadata(counts, x.device, block_m, block_n)
    reused = grouped_mxfp4_weight_gemm(
        x,
        payload,
        scales,
        counts,
        activation_format=activation_format,
        metadata=metadata,
    )
    assert torch.equal(per_call, reused)
    # A second GEMM of a different shape through the same offsets must also match.
    n2, k2 = 2048, 768
    x2 = torch.randn((sum(counts), k2), device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn((e, n2, k2), device="cuda", dtype=torch.bfloat16) * 0.02
    p2, s2 = pack_grouped_mxfp4_weights(w2)
    block_m2, block_n2 = block_shape(family, max(counts), n2, k2, e)
    assert (block_m2, block_n2) == (block_m, block_n), (
        family,
        (block_m2, block_n2),
        (block_m, block_n),
    )
    assert torch.equal(
        grouped_mxfp4_weight_gemm(
            x2, p2, s2, counts, activation_format=activation_format
        ),
        grouped_mxfp4_weight_gemm(
            x2, p2, s2, counts, activation_format=activation_format, metadata=metadata
        ),
    )


@pytest.mark.parametrize("rows,n", [(4097, 2048), (1, 768), (33, 96), (0, 128)])
def test_fused_row_scale_is_bit_identical(rows, n):
    """The fused router weighting must equal the unfused PyTorch chain exactly."""
    torch.manual_seed(11)
    x = torch.randn((rows, n), device="cuda", dtype=torch.bfloat16)
    score = torch.rand((rows,), device="cuda", dtype=torch.float32) * 4 - 2
    fused = row_scale_fp32(x, score)
    expected = (x.float() * score[:, None]).to(x.dtype)
    assert torch.equal(fused, expected)
    out = torch.empty_like(x)
    assert row_scale_fp32(x, score, out) is out
    assert torch.equal(out, expected)


def test_all_empty_and_contract_rejection():
    weight = torch.ones((2, 64, 64), dtype=torch.bfloat16, device="cuda")
    payload, scales = pack_grouped_mxfp4_weights(weight)
    empty = torch.empty((0, 64), dtype=torch.bfloat16, device="cuda")
    assert grouped_mxfp4_weight_gemm(empty, payload, scales, (0, 0)).shape == (0, 64)
    x = torch.ones((4, 64), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="sum"):
        grouped_mxfp4_weight_gemm(x, payload, scales, (1, 1))
    with pytest.raises(ValueError, match="overlap"):
        grouped_mxfp4_weight_gemm(x, payload, scales, (2, 2), out=x)
    with pytest.raises(ValueError, match="scales"):
        grouped_mxfp4_weight_gemm(x, payload, scales[:, :, :1].contiguous(), (2, 2))
    with pytest.raises(ValueError, match="activation_format"):
        grouped_mxfp4_weight_gemm(x, payload, scales, (2, 2), activation_format="fp3")
