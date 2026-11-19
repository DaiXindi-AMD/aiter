# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public segmented columnwise MXFP4 quantization."""

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.grouped_mxfp4_colwise import (
    _grouped_mxfp4_colwise_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.mxfp4_wgrad import (
    MXFP4ColwisePack,
    MXFP4WgradMetadata,
    record_mxfp4_colwise_quant,
    tensor_version,
    validate_mxfp4_wgrad_metadata,
)

__all__ = ["quantize_grouped_mxfp4_colwise"]


def quantize_grouped_mxfp4_colwise(x, metadata):
    """Quantize finite BF16 x[R,F] along each expert's token axis in blocks of 32.

    Return a metadata/source-bound pack with [0]=uint8 payload[F,P/2] and
    [1]=E8M0 scales[F,P/32]. Every padded token is zero; feature tails are valid.
    Source and metadata must not change, including raw storage writes. External
    prepack reuse requires a source version counter; no cross-update cache.
    """
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous CUDA BF16 [R,F]")
    if (
        not isinstance(metadata, MXFP4WgradMetadata)
        or metadata.total_rows != x.shape[0]
    ):
        raise ValueError("metadata must match input rows")
    validate_mxfp4_wgrad_metadata(metadata, metadata.counts, x.device, 32)
    with torch.cuda.device(x.device):
        directory = resolve_config_dir(
            "quant", "GROUPED-MXFP4-COLWISE", backend="triton"
        )
        cfg = load_config_json(os.path.join(directory, "DEFAULT.json"))
    bt, bf = cfg["BLOCK_SIZE_T"], cfg["BLOCK_SIZE_F"]
    if bt < 32 or bt % 32 or bt & (bt - 1) or bf <= 0 or bf & (bf - 1):
        raise ValueError("invalid columnwise quant tile")
    if metadata.pad_multiple % bt:
        raise ValueError("metadata padding must be a multiple of the quant tile")
    f, p = x.shape[1], metadata.padded_rows
    if f <= 0:
        raise ValueError("positive feature count required")
    with torch.inference_mode(False):
        payload = torch.empty((f, p // 2), dtype=torch.uint8, device=x.device)
        scales = torch.empty((f, p // 32), dtype=torch.uint8, device=x.device)
    if p:
        _grouped_mxfp4_colwise_kernel[
            (metadata.prep_blocks.shape[0], triton.cdiv(f, bf))
        ](
            x,
            payload,
            scales,
            metadata.real_offsets,
            metadata.padded_offsets,
            metadata.prep_blocks,
            payload.stride(0),
            scales.stride(0),
            F=f,
            PAD_MULTIPLE=metadata.pad_multiple,
            BLOCK_SIZE_T=bt,
            BLOCK_SIZE_F=bf,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            CONFIG_NUM_WARPS=cfg["num_warps"],
            CONFIG_NUM_STAGES=cfg["num_stages"],
        )
    record_mxfp4_colwise_quant()
    return MXFP4ColwisePack(
        payload,
        scales,
        metadata,
        x,
        tensor_version(x),
        (tensor_version(payload), tensor_version(scales)),
    )
