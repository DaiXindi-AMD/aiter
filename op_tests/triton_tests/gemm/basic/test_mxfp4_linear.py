# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton._triton_kernels.kimi_delta_attn import fast_launch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.gemm.basic.mxfp4_linear import (
    mxfp4_linear_gemm,
    pack_mxfp4_linear_weight,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx950", reason="dense MXFP4 linear targets gfx950"
)

# Fprop and Dgrad of the Qwen3-30B-A3B attention projections, plus a ragged M.
SHAPES = [
    (4096, 5120, 2048),
    (4096, 2048, 5120),
    (4096, 2048, 4096),
    (4096, 4096, 2048),
    (1000, 512, 256),
]

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant(payload, scales):
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), device=payload.device)
    vals = torch.stack(
        (lut[(payload & 0xF).long()], lut[(payload >> 4).long()]), dim=-1
    ).reshape(payload.shape[0], -1)
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
    return vals * scale[:, : vals.shape[1]]


def _inputs(m, n, k):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda").bfloat16()
    w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
    return a, w


@pytest.mark.parametrize("m, n, k", SHAPES)
def test_matches_public_quant_and_gemm(m, n, k):
    a, w = _inputs(m, n, k)
    (w_k, s_k), _ = pack_mxfp4_linear_weight(w)
    q, s = dynamic_mxfp4_quant(a)
    expected = gemm_afp4wfp4(q, w_k, s, s_k, torch.bfloat16)
    for _ in range(2):
        assert torch.equal(mxfp4_linear_gemm(a, w_k, s_k), expected)


@pytest.mark.parametrize("n, k", [(5120, 2048), (2048, 4096)])
def test_weight_packs_quantize_each_axis(n, k):
    _, w = _inputs(1, n, k)
    (w_k, s_k), (w_n, s_n) = pack_mxfp4_linear_weight(w)
    for (payload, scales), (ref_p, ref_s) in (
        ((w_k, s_k), dynamic_mxfp4_quant(w)),
        ((w_n, s_n), dynamic_mxfp4_quant(w.t().contiguous())),
    ):
        assert torch.equal(payload, ref_p)
        assert torch.equal(scales, ref_s)


@pytest.mark.parametrize("m, n, k", SHAPES[:2])
def test_close_to_dequantized_reference(m, n, k):
    a, w = _inputs(m, n, k)
    (w_k, s_k), (w_n, s_n) = pack_mxfp4_linear_weight(w)
    q, s = dynamic_mxfp4_quant(a)
    ref = _dequant(q, s) @ _dequant(w_k, s_k).t()
    y = mxfp4_linear_gemm(a, w_k, s_k).float()
    assert ((y - ref).abs().max() / ref.abs().max()).item() < 8e-3

    dy = torch.randn(m, n, device="cuda").bfloat16()
    q, s = dynamic_mxfp4_quant(dy)
    ref = _dequant(q, s) @ _dequant(w_n, s_n).t()
    dx = mxfp4_linear_gemm(dy, w_n, s_n).float()
    assert ((dx - ref).abs().max() / ref.abs().max()).item() < 8e-3


def test_non_contiguous_input_uses_public_path():
    a, w = _inputs(256, 512, 256)
    (w_k, s_k), _ = pack_mxfp4_linear_weight(w)
    strided = torch.empty(256, 512, device="cuda").bfloat16()[:, :256]
    strided.copy_(a)
    assert torch.equal(
        mxfp4_linear_gemm(strided, w_k, s_k), mxfp4_linear_gemm(a, w_k, s_k)
    )


def test_rejects_mismatched_pack():
    a, w = _inputs(64, 512, 256)
    (w_k, s_k), _ = pack_mxfp4_linear_weight(w)
    with pytest.raises(ValueError):
        mxfp4_linear_gemm(a[:, :128].contiguous(), w_k, s_k)


def test_repeat_calls_reuse_the_compiled_kernels():
    a, w = _inputs(4096, 5120, 2048)
    (w_k, s_k), _ = pack_mxfp4_linear_weight(w)
    mxfp4_linear_gemm(a, w_k, s_k)
    with fast_launch.recording() as misses:
        mxfp4_linear_gemm(torch.randn_like(a), w_k, s_k)
    assert misses == []
