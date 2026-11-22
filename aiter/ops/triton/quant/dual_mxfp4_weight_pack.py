# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Row- and column-axis MXFP4 packs of a dense [E, M, N] weight from one read."""

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.dual_mxfp4_weight_pack import (
    _dual_mxfp4_weight_pack_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

__all__ = ["quantize_mxfp4_weight_dual"]


def _load_config():
    directory = resolve_config_dir("quant", "DUAL-MXFP4-WEIGHT-PACK", backend="triton")
    return load_config_json(f"{directory}/DEFAULT.json")


def quantize_mxfp4_weight_dual(weight):
    """Quantize contiguous CUDA BF16 [E, M, N] along N and along M in one pass.

    Returns ``((row_payload, row_scales), (col_payload, col_scales))`` with shapes
    [E, M, N/2], [E, M, N/32], [E, N, M/2] and [E, N, M/32], all contiguous
    uint8. They are byte-identical to ``dynamic_mxfp4_quant`` of ``weight`` and of
    ``weight.transpose(1, 2)`` reshaped per row. gfx950 only; M and N must be
    multiples of the configured tile.
    """
    if (
        weight.ndim != 3
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError("weight must be contiguous CUDA BF16 [E,M,N]")
    if get_arch() != "gfx950":
        raise RuntimeError("dual MXFP4 weight pack supports gfx950 only")
    e, m, n = weight.shape
    cfg = _load_config()
    block_m, block_n = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]
    if min(e, m, n) <= 0 or m % block_m or n % block_n:
        raise ValueError(
            f"[E,M,N]={tuple(weight.shape)} must be positive multiples of the "
            f"({block_m},{block_n}) tile"
        )
    row_payload = torch.empty((e, m, n // 2), dtype=torch.uint8, device=weight.device)
    row_scales = torch.empty((e, m, n // 32), dtype=torch.uint8, device=weight.device)
    col_payload = torch.empty((e, n, m // 2), dtype=torch.uint8, device=weight.device)
    col_scales = torch.empty((e, n, m // 32), dtype=torch.uint8, device=weight.device)
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m), e)
    _dual_mxfp4_weight_pack_kernel[grid](
        weight,
        row_payload,
        row_scales,
        col_payload,
        col_scales,
        M=m,
        N=n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        USE_ASM=cfg["USE_ASM"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return (row_payload, row_scales), (col_payload, col_scales)
