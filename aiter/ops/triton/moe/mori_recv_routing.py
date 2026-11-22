# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused receive-side routing for MORI EP dispatch outputs."""

import torch

from aiter.ops.triton._triton_kernels.moe.moe_routing.mori_recv import (
    _mori_recv_local_routing_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

__all__ = ["mori_recv_local_routing"]


def _load_config():
    directory = resolve_config_dir("moe", "MORI-RECV-ROUTING", backend="triton")
    return load_config_json(f"{directory}/DEFAULT.json")


def mori_recv_local_routing(indices, num_valid, local_start, num_local_experts):
    """Local expert ids and per-local-expert counts of received MORI tokens.

    ``indices`` is the contiguous CUDA int32/int64 [R, K] global expert ids a MORI
    dispatch returns; rows at or past ``num_valid[0]`` are padding. Returns
    ``(local_ids, counts)``: int64 [R, K] ids in [0, num_local_experts) with -1 for
    padding rows and non-local slots, and int64 [num_local_experts] counts of each
    local id. Values equal the masked ``torch.where`` plus ``scatter_add_`` form.
    """
    if (
        indices.ndim != 2
        or not indices.is_cuda
        or indices.dtype not in (torch.int32, torch.int64)
        or not indices.is_contiguous()
    ):
        raise ValueError("indices must be contiguous CUDA int32/int64 [R,K]")
    if (
        num_valid.device != indices.device
        or num_valid.dtype not in (torch.int32, torch.int64)
        or num_valid.numel() < 1
        or not num_valid.is_contiguous()
    ):
        raise ValueError("num_valid must be a contiguous integer tensor on indices.device")
    if type(local_start) is not int or local_start < 0:
        raise ValueError("local_start must be a nonnegative int")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive int")
    local = torch.empty(indices.shape, dtype=torch.int64, device=indices.device)
    counts = torch.zeros(num_local_experts, dtype=torch.int64, device=indices.device)
    total = indices.numel()
    if total == 0:
        return local, counts
    cfg = _load_config()
    block = cfg["BLOCK"]
    _mori_recv_local_routing_kernel[(-(-total // block),)](
        indices,
        num_valid,
        local,
        counts,
        total,
        local_start,
        TOPK=indices.shape[1],
        NUM_LOCAL=num_local_experts,
        NUM_BINS=1 << (num_local_experts - 1).bit_length(),
        BLOCK=block,
        num_warps=cfg["num_warps"],
    )
    return local, counts
