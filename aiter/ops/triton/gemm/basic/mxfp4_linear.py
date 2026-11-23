# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dense MXFP4 x MXFP4 linear GEMMs with independently quantized operands.

``mxfp4_linear_gemm(a, w_fp4, w_scales)`` quantizes a BF16 [M, K] operand to
row-wise MXFP4 (1x32 blocks, "even" E8M0 scale, round-to-nearest-even; the same
bytes as ``dynamic_mxfp4_quant``) and returns ``a_q @ w_q.T`` in BF16.
``pack_mxfp4_linear_weight(w)`` returns both packs of an [N, K] weight: along K
for Fprop (Y = X W^T) and along N for Dgrad (dX = dY W).

Both launches reuse the compiled kernels through ``fast_launch``; the public
quant and GEMM wrappers cost ~60 us of host time per pair, more than the GEMM
saves when the caller is host-bound.
"""

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant.dual_mxfp4_weight_pack import quantize_mxfp4_weight_dual
from aiter.ops.triton.quant.quant import _has_scaled_downcast, dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.quant_config_utils import get_quant_config

__all__ = ["mxfp4_linear_gemm", "pack_mxfp4_linear_weight"]

_MX_BLOCK = 32
_GEMM = fast_launch(_gemm_afp4wfp4_kernel)


def pack_mxfp4_linear_weight(weight):
    """Return ``((w_k, s_k), (w_n, s_n))`` for a BF16 [N, K] weight.

    ``w_k``/``s_k`` ([N, K/2], [N, K/32]) quantize along K and ``w_n``/``s_n``
    ([K, N/2], [K, N/32]) along N, each byte-identical to ``dynamic_mxfp4_quant``
    of ``weight`` and of ``weight.t()``.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be [N, K]")
    try:
        (row, row_s), (col, col_s) = quantize_mxfp4_weight_dual(
            weight.contiguous()[None]
        )
        return (row[0], row_s[0]), (col[0], col_s[0])
    except (ValueError, RuntimeError):
        return dynamic_mxfp4_quant(weight), dynamic_mxfp4_quant(weight.t().contiguous())


@functools.cache
def _fast_path_available(device_index):
    del device_index
    return arch_info.get_arch() == "gfx950" and _has_scaled_downcast()


@functools.cache
def _quant_kernel():
    from aiter.ops.triton._gluon_kernels.gfx950.quant.quant import (
        gluon_dynamic_mxfp4_quant_kernel_gfx950,
    )

    return fast_launch(gluon_dynamic_mxfp4_quant_kernel_gfx950)


@functools.cache
def _quant_meta(m, k):
    cfg = get_quant_config("MXFP4", M=m, N=k)
    bm, bn, num_iter = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["NUM_ITER"]
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bn * num_iter))
    kwargs = {
        "M": m,
        "N": k,
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "NUM_ITER": num_iter,
        "NUM_STAGES": cfg["NUM_STAGES"],
        "num_warps": cfg["NUM_WARPS"],
        "MXFP4_QUANT_BLOCK_SIZE": _MX_BLOCK,
        "EVEN_M_N": m % bm == 0 and k % (bn * num_iter) == 0,
    }
    return grid, kwargs


@functools.cache
def _gemm_meta(m, n, k_packed):
    """Single-split config for the Triton kernel, or None to use the wrapper."""
    cfg, _ = _get_config(m, n, k_packed)
    if cfg.get("NUM_KSPLIT", 1) != 1 or cfg["BLOCK_SIZE_K"] >= 2 * k_packed:
        return None
    cfg["BLOCK_SIZE_K"] = max(cfg["BLOCK_SIZE_K"], 128)
    cfg["SPLITK_BLOCK_SIZE"] = 2 * k_packed
    grid = (triton.cdiv(m, cfg["BLOCK_SIZE_M"]) * triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)
    return grid, cfg


def _quantize(a):
    m, k = a.shape
    payload = torch.empty((m, k // 2), dtype=torch.uint8, device=a.device)
    scales = torch.empty((k // _MX_BLOCK, m), dtype=torch.uint8, device=a.device).T
    grid, kwargs = _quant_meta(m, k)
    _quant_kernel()[grid](
        x_ptr=a,
        x_fp4_ptr=payload,
        bs_ptr=scales,
        stride_x_m_in=a.stride(0),
        stride_x_n_in=a.stride(1),
        stride_x_fp4_m_in=payload.stride(0),
        stride_x_fp4_n_in=payload.stride(1),
        stride_bs_m_in=scales.stride(0),
        stride_bs_n_in=scales.stride(1),
        **kwargs,
    )
    return payload, scales


def mxfp4_linear_gemm(a, w_fp4, w_scales):
    """BF16 ``quant(a) @ w.T`` for a BF16 [M, K] ``a`` and an [N, K/2] MXFP4 weight pack."""
    m, k = a.shape
    n = w_fp4.shape[0]
    if w_fp4.shape != (n, k // 2) or w_scales.shape != (n, k // _MX_BLOCK):
        raise ValueError(
            f"weight pack {tuple(w_fp4.shape)}/{tuple(w_scales.shape)} does not match K={k}"
        )
    meta = None
    if (
        a.dtype == torch.bfloat16
        and a.is_contiguous()
        and k % (2 * _MX_BLOCK) == 0
        and _fast_path_available(a.device.index)
    ):
        meta = _gemm_meta(m, n, k // 2)
    if meta is None:
        q, s = dynamic_mxfp4_quant(a)
        return gemm_afp4wfp4(q, w_fp4, s, w_scales, torch.bfloat16)
    q, s = _quantize(a)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    grid, cfg = meta
    _GEMM[grid](
        a_ptr=q,
        b_ptr=w_fp4,
        c_ptr=y,
        a_scales_ptr=s,
        b_scales_ptr=w_scales,
        M=m,
        N=n,
        K=k // 2,
        stride_am=q.stride(0),
        stride_ak=q.stride(1),
        stride_bk=w_fp4.stride(1),
        stride_bn=w_fp4.stride(0),
        stride_ck=0,
        stride_cm=y.stride(0),
        stride_cn=y.stride(1),
        stride_asm=s.stride(0),
        stride_ask=s.stride(1),
        stride_bsn=w_scales.stride(0),
        stride_bsk=w_scales.stride(1),
        **cfg,
    )
    return y
