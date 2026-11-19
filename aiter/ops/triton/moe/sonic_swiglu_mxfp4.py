# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public fused SonicMoE SwiGLU and grouped MXFP4 producer."""

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.sonic_swiglu_mxfp4 import (
    _sonic_swiglu_mxfp4_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.mxfp4_wgrad import (
    MXFP4ColwisePack,
    MXFP4WgradMetadata,
    record_mxfp4_colwise_quant,
    tensor_version,
    validate_mxfp4_wgrad_metadata,
)

__all__ = ["fused_swiglu_mxfp4"]


def _load_config(device):
    with torch.cuda.device(device):
        directory = resolve_config_dir("quant", "SONIC-SWIGLU-MXFP4", backend="triton")
    return load_config_json(os.path.join(directory, "DEFAULT.json"))


def _validate_tile(config, metadata):
    block_t = config["BLOCK_SIZE_T"]
    block_f = config["BLOCK_SIZE_F"]
    if (
        block_t < 32
        or block_t % 32
        or block_t & (block_t - 1)
        or block_f < 32
        or block_f % 32
        or block_f & (block_f - 1)
    ):
        raise ValueError("invalid fused SwiGLU MXFP4 tile")
    if metadata.pad_multiple not in config["SUPPORTED_PAD_MULTIPLES"]:
        raise ValueError("metadata padding is not configured for the fused producer")
    if metadata.pad_multiple % block_t:
        raise ValueError("metadata padding must be a multiple of the token tile")
    return block_t, block_f


def fused_swiglu_mxfp4(h1, metadata):
    """Apply concat-layout SwiGLU and emit BF16 plus two MXFP4 layouts.

    ``h1`` is finite contiguous CUDA BF16 ``[R,2F]``. ``metadata`` binds the
    expert routing and padding. Return ``(a, row, col)`` where ``a`` is BF16
    ``[R,F]``, ``row`` is the GEMM2 payload/scale pair, and ``col`` is the
    metadata/source-bound Wgrad pack. The configured gfx950 feature sizes apply.
    """
    if (
        h1.ndim != 2
        or not h1.is_cuda
        or h1.dtype != torch.bfloat16
        or not h1.is_contiguous()
        or h1.shape[1] % 64
    ):
        raise ValueError(
            "h1 must be contiguous CUDA BF16 [R,2F] with F divisible by 32"
        )
    if (
        not isinstance(metadata, MXFP4WgradMetadata)
        or metadata.total_rows != h1.shape[0]
    ):
        raise ValueError("metadata must match input rows")
    validate_mxfp4_wgrad_metadata(metadata, metadata.counts, h1.device, 32)
    if get_arch() != "gfx950":
        raise ValueError("fused SonicMoE SwiGLU MXFP4 supports gfx950 only")

    feature_size = h1.shape[1] // 2
    config = _load_config(h1.device)
    block_t, block_f = _validate_tile(config, metadata)
    if feature_size not in config["SUPPORTED_FEATURES"]:
        raise ValueError(
            "fused SonicMoE SwiGLU MXFP4 does not support this feature size"
        )

    padded_rows = metadata.padded_rows
    with torch.inference_mode(False):
        activation = torch.empty(
            (h1.shape[0], feature_size), dtype=h1.dtype, device=h1.device
        )
        row_payload = torch.empty(
            (h1.shape[0], feature_size // 2),
            dtype=torch.uint8,
            device=h1.device,
        )
        row_scales = torch.empty(
            (h1.shape[0], feature_size // 32),
            dtype=torch.uint8,
            device=h1.device,
        )
        col_payload = torch.empty(
            (feature_size, padded_rows // 2),
            dtype=torch.uint8,
            device=h1.device,
        )
        col_scales = torch.empty(
            (feature_size, padded_rows // 32),
            dtype=torch.uint8,
            device=h1.device,
        )

    if padded_rows:
        grid_f = triton.cdiv(feature_size, block_f)
        grid_size = metadata.compact_blocks.shape[0] * grid_f
        _sonic_swiglu_mxfp4_kernel[(grid_size,)](
            h1,
            activation,
            row_payload,
            row_scales,
            col_payload,
            col_scales,
            metadata.compact_blocks,
            activation.stride(0),
            row_payload.stride(0),
            row_scales.stride(0),
            col_payload.stride(0),
            col_scales.stride(0),
            F=feature_size,
            PAD_MULTIPLE=metadata.pad_multiple,
            BLOCK_SIZE_T=block_t,
            BLOCK_SIZE_F=block_f,
            GRID_F=grid_f,
            GRID_SIZE=grid_size,
            NUM_XCDS=config["NUM_XCDS"],
            XCD_CHUNK=config["XCD_CHUNK"],
            USE_ASM=config["USE_ASM"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            CONFIG_NUM_WARPS=config["num_warps"],
            CONFIG_NUM_STAGES=config["num_stages"],
        )

    record_mxfp4_colwise_quant()
    col = MXFP4ColwisePack(
        col_payload,
        col_scales,
        metadata,
        activation,
        tensor_version(activation),
        (tensor_version(col_payload), tensor_version(col_scales)),
    )
    return activation, (row_payload, row_scales), col
