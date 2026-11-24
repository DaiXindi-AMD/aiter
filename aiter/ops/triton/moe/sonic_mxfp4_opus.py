# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A4W4 pre-routed SwiGLU experts with native MXFP4 Wgrad on the fused OPUS chain.

Same recipe and numerics as ``_PreRoutedMXFP4`` with ``wgrad_mode="mxfp4"`` and
row/column dual quant; it differs only in how the work is issued:

* HIP producers fuse x, SwiGLU, dout and SwiGLU-backward with their rowwise and
  expert-padded colwise MXFP4 quantization;
* the routed output is the epilogue of the second forward GEMM;
* an OPUS kernel computes both Wgrads from the colwise packs;
* GEMM block maps and Wgrad padding reach the GPU in one pinned copy, and the
  grouped GEMMs launch through ``fast_launch``.

Selected with ``wgrad_backend="opus"`` on ``moe_pre_routed_a4w4``.
"""

import functools

import numpy as np
import torch

import aiter.ops.triton.gemm.grouped.sonic_mxfp4 as grouped
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.mxfp4_moe import (
    QUANT_DOUT,
    QUANT_SWIGLU,
    QUANT_SWIGLU_BWD,
    QUANT_X,
    ROUTE_BLOCK,
    WGRAD_TILE,
    mxfp4_moe_fused_quant,
    opus_mxfp4_wgrad,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
    _grouped_gemm_kernel,
)
from aiter.ops.triton._triton_kernels.moe.sonicmoe.row_scale import row_dot_fp32
from aiter.ops.triton.utils.mxfp4_wgrad import MXFP4WgradMetadata, tensor_version

_GEMM = fast_launch(_grouped_gemm_kernel)


@functools.lru_cache(maxsize=256)
def _gemm_config(m_bucket, n, k, e):
    return grouped._resolve_cfg(grouped._FAMILY["mxfp4"], m_bucket, n, k, e)


@functools.lru_cache(maxsize=256)
def _block_shape(m_bucket, e, h, i):
    """The single (BLOCK_M, BLOCK_N) shared by the four expert GEMMs, or None."""
    shapes = ((2 * i, h), (h, i), (i, h), (h, 2 * i))
    cfgs = [_gemm_config(m_bucket, n, k, e) for n, k in shapes]
    if any(c["NUM_KSPLIT"] != 1 or c["BLOCK_SIZE_K"] % 32 for c in cfgs):
        return None
    blocks = {(c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"]) for c in cfgs}
    return next(iter(blocks)) if len(blocks) == 1 else None


def opus_wgrad_unsupported_reason(device, e, h, i, max_count):
    """None when the fused chain supports this layer, otherwise why not."""
    if get_gfx() != "gfx950":
        return "the OPUS MXFP4 chain requires gfx950"
    if h % WGRAD_TILE or i % WGRAD_TILE:
        return f"hidden and intermediate sizes must be multiples of {WGRAD_TILE}"
    if _block_shape(grouped._m_bucket(max_count), e, h, i) is None:
        return "the four expert GEMMs do not share one block shape"
    return None


def _build_metadata(counts, device, ns, block_m, block_n):
    """GEMM block maps and Wgrad padding metadata from one pinned H2D copy."""
    c = np.asarray(counts, dtype=np.int64)
    e = c.size
    off = np.zeros(e + 1, np.int64)
    np.cumsum(c, out=off[1:])
    gemm_parts = [off, c]
    m_blocks = -(-c // block_m)
    for n in ns:
        base = np.zeros(e + 1, np.int64)
        np.cumsum(m_blocks * -(-n // block_n), out=base[1:])
        gemm_parts.append(base)
    n_blocks = -(-c // ROUTE_BLOCK)
    padded = np.zeros(e + 1, np.int64)
    np.cumsum(n_blocks * ROUTE_BLOCK, out=padded[1:])
    expert = np.repeat(np.arange(e), n_blocks)
    first = np.zeros(e + 1, np.int64)
    np.cumsum(n_blocks, out=first[1:])
    index = np.arange(expert.size) - first[expert]
    prep = np.stack((expert, index), 1)
    compact = np.stack(
        (
            off[expert] + index * ROUTE_BLOCK,
            off[expert + 1],
            padded[expert] + index * ROUTE_BLOCK,
        ),
        1,
    )
    gemm_flat = np.concatenate(gemm_parts)
    wgrad_flat = np.concatenate((off, padded, prep.ravel(), compact.ravel()))
    if max(gemm_flat.max(initial=0), wgrad_flat.max(initial=0)) >= 2**31:
        raise ValueError("grouped MXFP4 metadata exceeds int32")
    # The Wgrad tensors start on a 256-byte boundary of the shared buffer.
    wgrad_start = -(-gemm_flat.size // 64) * 64
    host = torch.empty(
        wgrad_start + wgrad_flat.size, dtype=torch.int32, pin_memory=True
    )
    flat = host.numpy()
    flat[: gemm_flat.size] = gemm_flat
    flat[gemm_flat.size : wgrad_start] = 0
    flat[wgrad_start:] = wgrad_flat
    with torch.inference_mode(False):
        dev = host.to(device, non_blocking=True)
    meta = {
        "counts_dev": dev[e + 1 : 2 * e + 1],
        "offsets": dev[: e + 1],
        "block_m": block_m,
        "block_n": block_n,
        "e": e,
    }
    start = 2 * e + 1
    for n in ns:
        meta[("block_base", n)] = dev[start : start + e + 1]
        start += e + 1
    w = wgrad_start
    tensors = (
        dev[w : w + e + 1],
        dev[w + e + 1 : w + 2 * e + 2],
        dev[w + 2 * e + 2 : w + 2 * e + 2 + prep.size].view(-1, 2),
        dev[w + 2 * e + 2 + prep.size :].view(-1, 3),
    )
    wgrad_meta = MXFP4WgradMetadata(
        tuple(counts),
        ROUTE_BLOCK,
        int(off[-1]),
        int(padded[-1]),
        *tensors,
        dev.untyped_storage().data_ptr(),
        tuple(tensor_version(t) for t in tensors),
    )
    return meta, wgrad_meta


def _gemm(a_quant, weight, counts, meta, row_scale=None):
    """BF16 [R,N] A4W4 grouped GEMM; with row_scale also bf16(out * row_scale)."""
    ap, asc = a_quant
    payload, scales = weight
    e, n, half_k = payload.shape
    k = 2 * half_k
    cfg = _gemm_config(grouped._m_bucket(max(counts)), n, k, e)
    bm, bn = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]
    out = torch.empty((ap.shape[0], n), dtype=torch.bfloat16, device=ap.device)
    scaled = None if row_scale is None else torch.empty_like(out)
    grid = sum(-(-c // bm) for c in counts) * -(-n // bn)
    if grid == 0:
        return out, scaled
    offsets = meta["offsets"]
    extra = {}
    if row_scale is not None:
        extra = {"row_scale_ptr": row_scale, "scaled_C_ptr": scaled, "ROW_SCALE": True}
    _GEMM[(grid,)](
        A_ptr=ap,
        B_ptr=payload,
        C_ptr=out,
        cu_seqlens_ptr=offsets,
        bias_ptr=out,
        A_idx_ptr=offsets,
        scatter_idx_ptr=offsets,
        stride_ak=ap.stride(0),
        stride_am=ap.stride(1),
        stride_be=payload.stride(0),
        stride_bk=payload.stride(2),
        stride_bn=payload.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        stride_bias_e=0,
        stride_bias_n=0,
        N=n,
        K=k,
        E=e,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        HAS_BIAS=False,
        HAS_GATHER_IDX=False,
        HAS_SCATTER_IDX=False,
        B_scales_ptr=scales,
        W4_WEIGHT=True,
        CACHE_MODIFIER=cfg["cache_modifier"],
        A_scales_ptr=asc,
        A4_ACTIVATION=True,
        A8_ACTIVATION=False,
        expert_block_base_ptr=meta[("block_base", n)],
        BLOCK_MAP_ITERS=max(1, (e - 1).bit_length()),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
        **extra,
    )
    return out, scaled


class _PreRoutedMXFP4Opus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scores, w1, w2, counts, packed, main_grads):
        from aiter.ops.triton.moe.sonic_mxfp4 import pack_pre_routed_weights

        e, h, j = w1.shape
        i = j // 2
        block_m, block_n = _block_shape(grouped._m_bucket(max(counts)), e, h, i)
        meta, wgrad_meta = _build_metadata(
            counts, x.device, (j, h, i), block_m, block_n
        )
        if packed is None:
            packed = pack_pre_routed_weights(w1, w2)
        w1_t, w2_t, _w1_p, _w2_p = packed
        x_row, x_col = mxfp4_moe_fused_quant(QUANT_X, x, wgrad_meta, h)
        h1, _ = _gemm(x_row, w1_t, counts, meta)
        a_row, a_col = mxfp4_moe_fused_quant(QUANT_SWIGLU, h1, wgrad_meta, i, half=i)
        y, routed = _gemm(a_row, w2_t, counts, meta, row_scale=scores)
        ctx.counts, ctx.packed, ctx.meta, ctx.wgrad_meta = (
            counts,
            packed,
            meta,
            wgrad_meta,
        )
        ctx.main_grads, ctx.cols = main_grads, (x_col, a_col)
        ctx.save_for_backward(scores, w1, w2, h1, y)
        return routed

    @staticmethod
    def backward(ctx, dout):
        from aiter.ops.triton.moe.sonic_mxfp4 import (
            _main_grad_accumulate,
            _validate_main_grads,
        )

        scores, w1, w2, h1, y = ctx.saved_tensors
        if ctx.main_grads is not None:
            _validate_main_grads(ctx.main_grads, (w1, w2))
        counts, meta, wgrad_meta = ctx.counts, ctx.meta, ctx.wgrad_meta
        h = w1.shape[1]
        i = w2.shape[1]
        dout = dout.contiguous()
        ds = row_dot_fp32(dout, y)
        _w1_t, _w2_t, w1_p, w2_p = ctx.packed
        dout_row, dy_col = mxfp4_moe_fused_quant(
            QUANT_DOUT, dout, wgrad_meta, h, score=scores
        )
        uda, _ = _gemm(dout_row, w2_p, counts, meta)
        dh_row, dh_col = mxfp4_moe_fused_quant(
            QUANT_SWIGLU_BWD, h1, wgrad_meta, 2 * i, s1=uda, score=scores, half=i
        )
        dx, _ = _gemm(dh_row, w1_p, counts, meta)
        accumulate = _main_grad_accumulate((w1, w2), ctx.main_grads)
        targets = ctx.main_grads or (None, None)
        x_col, a_col = ctx.cols
        # No fallback after a write: a failed native call must fail this backward.
        grads = []
        for target, weight, left, right in zip(
            targets, (w1, w2), (x_col, a_col), (dh_col, dy_col)
        ):
            if target is None:
                target = torch.empty(
                    weight.shape, dtype=torch.float32, device=weight.device
                )
            grads.append(
                opus_mxfp4_wgrad(
                    left, right, wgrad_meta.padded_offsets, target, accumulate
                )
            )
        if ctx.main_grads is not None:
            for weight in (w1, w2):
                if hasattr(weight, "grad_added_to_main_grad"):
                    weight.grad_added_to_main_grad = True
            dw1 = dw2 = None
        else:
            dw1, dw2 = grads[0].to(w1.dtype), grads[1].to(w2.dtype)
        return dx, ds, dw1, dw2, None, None, None
