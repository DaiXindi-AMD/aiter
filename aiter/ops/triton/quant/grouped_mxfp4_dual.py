# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public grouped row/column MXFP4 activation producer."""

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.grouped_mxfp4_dual import (
    _grouped_mxfp4_dual_kernel,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import (
    quantize_grouped_mxfp4_colwise,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.mxfp4_wgrad import (
    MXFP4ColwisePack,
    MXFP4RowwisePack,
    MXFP4WgradMetadata,
    record_mxfp4_colwise_quant,
    tensor_structure,
    tensor_version,
    validate_mxfp4_wgrad_metadata,
)

__all__ = [
    "quantize_grouped_mxfp4_colwise_async",
    "quantize_grouped_mxfp4_concurrent",
    "quantize_grouped_mxfp4_dual",
    "quantize_grouped_mxfp4_dual_mapped",
    "quantize_grouped_mxfp4_dual_v2",
]

_SIDE_STREAMS = {}


def _side_stream(device):
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    stream = _SIDE_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index)
        _SIDE_STREAMS[index] = stream
    return stream


def quantize_grouped_mxfp4_colwise_async(x, metadata):
    """Launch columnwise quant on a side stream and return (pack, ready_event).

    The caller must wait for the event on the launch stream before it uses the
    pack. The wrapper records both streams for allocator-safe tensor lifetimes.
    """
    current = torch.cuda.current_stream(x.device)
    side = _side_stream(x.device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        col = quantize_grouped_mxfp4_colwise(x, metadata)
        ready = torch.cuda.Event()
        ready.record(side)
    for tensor in (
        x,
        metadata.real_offsets,
        metadata.padded_offsets,
        metadata.prep_blocks,
    ):
        tensor.record_stream(side)
    return col, ready


def quantize_grouped_mxfp4_concurrent(x, metadata):
    """Run row quant on the current stream while column quant runs on a side stream."""
    col, ready = quantize_grouped_mxfp4_colwise_async(x, metadata)
    if x.shape[0]:
        payload, scales = dynamic_mxfp4_quant(x)
        row = payload, scales.contiguous()
    else:
        with torch.inference_mode(False):
            row = (
                torch.empty((0, x.shape[1] // 2), dtype=torch.uint8, device=x.device),
                torch.empty((0, x.shape[1] // 32), dtype=torch.uint8, device=x.device),
            )
    return row, col, ready


def _validate_dual_inputs(x, metadata, source_map=None):
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[1] % 32
    ):
        raise ValueError("x must be contiguous CUDA BF16 [R,F] with F divisible by 32")
    if source_map is not None and (
        not isinstance(source_map, torch.Tensor)
        or source_map.ndim != 1
        or source_map.dtype != torch.int32
        or source_map.device != x.device
        or not source_map.is_contiguous()
    ):
        raise ValueError("source_map must be contiguous CUDA int32 [R] on x.device")
    if not isinstance(metadata, MXFP4WgradMetadata) or metadata.total_rows != (
        x.shape[0] if source_map is None else source_map.numel()
    ):
        raise ValueError("metadata must match input rows")
    validate_mxfp4_wgrad_metadata(metadata, metadata.counts, x.device, 32)
    if source_map is None:
        return
    if source_map.numel() and not x.shape[0]:
        raise ValueError("a nonempty source_map requires source rows")
    if torch._C._overlaps(x, source_map):
        raise ValueError("source and source_map must not overlap")
    if tensor_version(x) is None or tensor_version(source_map) is None:
        raise ValueError("mapped source and source_map require version counters")


def _load_dual_config(config_name, device):
    with torch.cuda.device(device):
        directory = resolve_config_dir("quant", config_name, backend="triton")
    return load_config_json(os.path.join(directory, "DEFAULT.json"))


def _validate_dual_tile(cfg, metadata):
    bt, bf = cfg["BLOCK_SIZE_T"], cfg["BLOCK_SIZE_F"]
    if bt < 32 or bt % 32 or bt & (bt - 1) or bf < 32 or bf % 32 or bf & (bf - 1):
        raise ValueError("invalid grouped dual quant tile")
    if metadata.pad_multiple % bt:
        raise ValueError("metadata padding must be a multiple of the quant tile")
    return bt, bf


def _allocate_dual_outputs(x, metadata):
    f, p = x.shape[1], metadata.padded_rows
    with torch.inference_mode(False):
        row_payload = torch.empty(
            (metadata.total_rows, f // 2), dtype=torch.uint8, device=x.device
        )
        row_scales = torch.empty(
            (metadata.total_rows, f // 32), dtype=torch.uint8, device=x.device
        )
        col_payload = torch.empty((f, p // 2), dtype=torch.uint8, device=x.device)
        col_scales = torch.empty((f, p // 32), dtype=torch.uint8, device=x.device)
    return row_payload, row_scales, col_payload, col_scales


def _wrap_dual_outputs(x, metadata, outputs, source_map=None):
    row_payload, row_scales, col_payload, col_scales = outputs
    source_version = tensor_version(x)
    source_map_version = None if source_map is None else tensor_version(source_map)
    record_mxfp4_colwise_quant()
    row_pack = (
        (row_payload, row_scales)
        if source_map is None
        else MXFP4RowwisePack(
            row_payload,
            row_scales,
            x,
            source_version,
            (tensor_version(row_payload), tensor_version(row_scales)),
            source_map,
            source_map_version,
        )
    )
    col_pack = MXFP4ColwisePack(
        col_payload,
        col_scales,
        metadata,
        x,
        source_version,
        (tensor_version(col_payload), tensor_version(col_scales)),
        source_map,
        source_map_version,
        source_map is not None,
        tensor_structure(x),
    )
    return row_pack, col_pack


def quantize_grouped_mxfp4_dual(x, metadata):
    """Return rowwise GEMM operands and a metadata-bound columnwise Wgrad pack.

    Set ``AITER_GROUPED_MXFP4_DUAL_V2=1`` to use the guarded gfx950 prototype.
    The default path preserves the original two-dimensional launch.
    """
    mode = os.environ.get("AITER_GROUPED_MXFP4_DUAL_V2", "0")
    if mode not in ("0", "1"):
        raise ValueError("AITER_GROUPED_MXFP4_DUAL_V2 must be 0 or 1")
    if mode == "1":
        return quantize_grouped_mxfp4_dual_v2(x, metadata)

    _validate_dual_inputs(x, metadata)
    cfg = _load_dual_config("GROUPED-MXFP4-COLWISE", x.device)
    bt, bf = _validate_dual_tile(cfg, metadata)
    outputs = _allocate_dual_outputs(x, metadata)
    row_payload, row_scales, col_payload, col_scales = outputs
    f, p = x.shape[1], metadata.padded_rows
    if p:
        _grouped_mxfp4_dual_kernel[(metadata.prep_blocks.shape[0], triton.cdiv(f, bf))](
            x,
            metadata.real_offsets,
            row_payload,
            row_scales,
            col_payload,
            col_scales,
            metadata.real_offsets,
            metadata.padded_offsets,
            metadata.prep_blocks,
            row_payload.stride(0),
            row_scales.stride(0),
            col_payload.stride(0),
            col_scales.stride(0),
            x.shape[0],
            F=f,
            PAD_MULTIPLE=metadata.pad_multiple,
            BLOCK_SIZE_T=bt,
            BLOCK_SIZE_F=bf,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            CONFIG_NUM_WARPS=cfg["num_warps"],
            CONFIG_NUM_STAGES=cfg["num_stages"],
            GATHER=False,
        )
    return _wrap_dual_outputs(x, metadata, outputs)


def quantize_grouped_mxfp4_dual_v2(x, metadata):
    """Run the gfx950 compact-route and XCD-aware grouped dual prototype.

    The input is finite contiguous BF16 ``[R,F]``. The current training contract
    uses 256-row expert padding and the configured feature sizes. The return value
    matches :func:`quantize_grouped_mxfp4_dual` byte for byte.
    """
    _validate_dual_inputs(x, metadata)
    if get_arch() != "gfx950":
        raise ValueError("grouped MXFP4 dual v2 supports gfx950 only")
    cfg = _load_dual_config("GROUPED-MXFP4-DUAL-V2", x.device)
    bt, bf = _validate_dual_tile(cfg, metadata)
    if metadata.pad_multiple != cfg["PAD_MULTIPLE"]:
        raise ValueError("grouped MXFP4 dual v2 requires configured expert padding")
    if x.shape[1] not in cfg["SUPPORTED_FEATURES"]:
        raise ValueError("grouped MXFP4 dual v2 does not support this feature size")

    outputs = _allocate_dual_outputs(x, metadata)
    row_payload, row_scales, col_payload, col_scales = outputs
    f, p = x.shape[1], metadata.padded_rows
    grid_f = triton.cdiv(f, bf)
    grid_size = metadata.compact_blocks.shape[0] * grid_f
    if p:
        _grouped_mxfp4_dual_kernel[(grid_size,)](
            x,
            metadata.real_offsets,
            row_payload,
            row_scales,
            col_payload,
            col_scales,
            metadata.real_offsets,
            metadata.padded_offsets,
            metadata.compact_blocks,
            row_payload.stride(0),
            row_scales.stride(0),
            col_payload.stride(0),
            col_scales.stride(0),
            x.shape[0],
            F=f,
            PAD_MULTIPLE=metadata.pad_multiple,
            BLOCK_SIZE_T=bt,
            BLOCK_SIZE_F=bf,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            CONFIG_NUM_WARPS=cfg["num_warps"],
            CONFIG_NUM_STAGES=cfg["num_stages"],
            GRID_F=grid_f,
            GRID_SIZE=grid_size,
            FLAT_GRID=True,
            NUM_XCDS=cfg["NUM_XCDS"],
            XCD_CHUNK=cfg["XCD_CHUNK"],
            USE_ASM=cfg["USE_ASM"],
            GATHER=False,
        )
    return _wrap_dual_outputs(x, metadata, outputs)


def quantize_grouped_mxfp4_dual_mapped(source, source_map, metadata):
    """Gather and quantize an unsorted BF16 source into expert-major dual packs.

    ``source_map`` is contiguous CUDA int32 ``[R]`` with values in
    ``[0, source.shape[0])``. It maps each expert-major destination row to a
    source row. Set ``AITER_GROUPED_MXFP4_MAPPED=1`` to enable this gfx950
    path. The tuple-compatible packs retain source/map identity and versions;
    mapped columnwise packs are valid Wgrad operands. Supported feature counts
    are 768, 1536, and 2048.
    """
    enabled = os.environ.get("AITER_GROUPED_MXFP4_MAPPED", "0")
    if enabled not in ("0", "1"):
        raise ValueError("AITER_GROUPED_MXFP4_MAPPED must be 0 or 1")
    if enabled != "1":
        raise RuntimeError("mapped grouped MXFP4 quantization is disabled")
    _validate_dual_inputs(source, metadata, source_map)
    if get_arch() != "gfx950":
        raise ValueError("mapped grouped MXFP4 dual quant supports gfx950 only")
    cfg = _load_dual_config("GROUPED-MXFP4-DUAL-V2", source.device)
    bt, bf = _validate_dual_tile(cfg, metadata)
    if metadata.pad_multiple != cfg["PAD_MULTIPLE"]:
        raise ValueError("mapped grouped MXFP4 dual quant requires configured padding")
    if source.shape[1] not in cfg["SUPPORTED_FEATURES"]:
        raise ValueError(
            "mapped grouped MXFP4 dual quant does not support this feature size"
        )

    outputs = _allocate_dual_outputs(source, metadata)
    row_payload, row_scales, col_payload, col_scales = outputs
    f, p = source.shape[1], metadata.padded_rows
    grid_f = triton.cdiv(f, bf)
    grid_size = metadata.compact_blocks.shape[0] * grid_f
    if p:
        _grouped_mxfp4_dual_kernel[(grid_size,)](
            source,
            source_map,
            row_payload,
            row_scales,
            col_payload,
            col_scales,
            metadata.real_offsets,
            metadata.padded_offsets,
            metadata.compact_blocks,
            row_payload.stride(0),
            row_scales.stride(0),
            col_payload.stride(0),
            col_scales.stride(0),
            source.shape[0],
            F=f,
            PAD_MULTIPLE=metadata.pad_multiple,
            BLOCK_SIZE_T=bt,
            BLOCK_SIZE_F=bf,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            CONFIG_NUM_WARPS=cfg["num_warps"],
            CONFIG_NUM_STAGES=cfg["num_stages"],
            GRID_F=grid_f,
            GRID_SIZE=grid_size,
            FLAT_GRID=True,
            NUM_XCDS=cfg["NUM_XCDS"],
            XCD_CHUNK=cfg["XCD_CHUNK"],
            USE_ASM=cfg["USE_ASM"],
            GATHER=True,
        )
    stream = torch.cuda.current_stream(source.device)
    source.record_stream(stream)
    source_map.record_stream(stream)
    return _wrap_dual_outputs(source, metadata, outputs, source_map)
