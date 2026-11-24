# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP producers and OPUS Wgrad for the fused pre-routed MXFP4 MoE chain (gfx950).

The quant kernels fuse each SwiGLU producer with rowwise and expert-padded colwise
MXFP4 quantization, bit-exact with the Triton dual-quant path. The Wgrad kernel
consumes the colwise packs. Callers own validation; see
``aiter.ops.triton.moe.sonic_mxfp4_opus``.
"""

from __future__ import annotations

import torch

from ...jit.core import compile_ops

QUANT_X = 0
QUANT_SWIGLU = 1
QUANT_SWIGLU_BWD = 2
QUANT_DOUT = 3

# Padded tokens per compact route block; the Wgrad K chunk divides it.
ROUTE_BLOCK = 256
WGRAD_TILE = 128


@compile_ops(
    "module_mxfp4_moe_fast", fc_name="mxfp4_moe_fused_quant", ffi_type="ctypes"
)
def _mxfp4_moe_fused_quant(
    prod: int,
    s0: int,
    s1: int,
    score: int,
    rpay: int,
    rsc: int,
    cpay: int,
    csc: int,
    routes: int,
    F: int,
    P: int,
    I: int,
    nblocks: int,
) -> None: ...


@compile_ops("module_mxfp4_moe_fast", fc_name="opus_mxfp4_wgrad", ffi_type="ctypes")
def _opus_mxfp4_wgrad(
    lp: int,
    ls: int,
    rp: int,
    rs: int,
    out: int,
    poff: int,
    pstride: int,
    sstride: int,
    M: int,
    N: int,
    E: int,
    accumulate: int,
) -> None: ...


def _allocate_dual(rows, features, padded_rows, device):
    """One buffer split into rowwise and colwise payload/scales, 256-byte aligned."""
    sizes = (
        rows * features // 2,
        rows * features // 32,
        features * padded_rows // 2,
        features * padded_rows // 32,
    )
    shapes = (
        (rows, features // 2),
        (rows, features // 32),
        (features, padded_rows // 2),
        (features, padded_rows // 32),
    )
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += -(-size // 256) * 256
    buf = torch.empty(total, dtype=torch.uint8, device=device)
    return [buf[o : o + s].view(shape) for o, s, shape in zip(offsets, sizes, shapes)]


def mxfp4_moe_fused_quant(prod, s0, metadata, features, s1=None, score=None, half=0):
    """Return ((row payload, row scales), (col payload, col scales)) for one producer.

    ``metadata`` is an ``MXFP4WgradMetadata`` padded to ``ROUTE_BLOCK``; rowwise
    outputs are [R, F/2] / [R, F/32] and colwise outputs [F, P/2] / [F, P/32].
    """
    rp, rs, cp, cs = outs = _allocate_dual(
        s0.shape[0], features, metadata.padded_rows, s0.device
    )
    _mxfp4_moe_fused_quant(
        prod,
        s0.data_ptr(),
        0 if s1 is None else s1.data_ptr(),
        0 if score is None else score.data_ptr(),
        rp.data_ptr(),
        rs.data_ptr(),
        cp.data_ptr(),
        cs.data_ptr(),
        metadata.compact_blocks.data_ptr(),
        features,
        metadata.padded_rows,
        half,
        metadata.compact_blocks.shape[0],
    )
    return (outs[0], outs[1]), (outs[2], outs[3])


def opus_mxfp4_wgrad(left, right, padded_offsets, out, accumulate):
    """out[e] (+)= left_e @ right_e^T in FP32 [E, M, N] from colwise MXFP4 packs."""
    lp, ls = left
    rp, rs = right
    e, m, n = out.shape
    _opus_mxfp4_wgrad(
        lp.data_ptr(),
        ls.data_ptr(),
        rp.data_ptr(),
        rs.data_ptr(),
        out.data_ptr(),
        padded_offsets.data_ptr(),
        lp.shape[1],
        ls.shape[1],
        m,
        n,
        e,
        int(accumulate),
    )
    return out
