# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared host metadata and counters for segmented MXFP4 operands."""

import copy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MXFP4WgradMetadata:
    """Read-only by contract; raw storage writes and .data mutation are unsupported."""

    counts: tuple[int, ...]
    pad_multiple: int
    total_rows: int
    padded_rows: int
    real_offsets: torch.Tensor
    padded_offsets: torch.Tensor
    prep_blocks: torch.Tensor
    storage_base: int
    tensor_versions: tuple[int, ...]

    @property
    def num_experts(self):
        return len(self.counts)


@dataclass(frozen=True)
class MXFP4ColwisePack:
    """Two tensor views bound to their exact producer routing and source version."""

    payload: torch.Tensor
    scales: torch.Tensor
    metadata: MXFP4WgradMetadata
    source: torch.Tensor
    source_version: int | None
    tensor_versions: tuple[int | None, int | None]

    def __getitem__(self, index):
        return (self.payload, self.scales)[index]

    def __iter__(self):
        return iter((self.payload, self.scales))

    def __len__(self):
        return 2

    def record_stream(self, stream):
        """Keep pack storage alive through work already queued on stream."""
        self.payload.record_stream(stream)
        self.scales.record_stream(stream)


def tensor_version(tensor):
    """Read a host version counter; inference tensors have no counter."""
    try:
        return tensor._version
    except RuntimeError:
        return None


_STATS = {
    "quant_calls": 0,
    "wgrad_calls": 0,
    "empty_experts": 0,
    "backend_calls": {},
    "fallbacks": {},
    "last_config": None,
}


def normalize_counts(counts, rows=None):
    """Validate immutable host counts; do not copy device metadata to the host."""
    if not isinstance(counts, (tuple, list)) or not counts:
        raise ValueError("counts must be a non-empty host tuple/list")
    counts = tuple(counts)
    if any(type(c) is not int or c < 0 for c in counts):
        raise ValueError("counts must contain nonnegative Python integers")
    if sum(counts) > 2**31 - 1:
        raise ValueError("token count exceeds int32 metadata")
    if rows is not None and sum(counts) != rows:
        raise ValueError("counts must sum to input rows")
    return counts


def build_mxfp4_wgrad_metadata(counts, device, pad_multiple):
    """Build offset tensors with one H2D transfer and retain structural provenance."""
    counts = normalize_counts(counts)
    if type(pad_multiple) is not int or pad_multiple <= 0 or pad_multiple % 32:
        raise ValueError("pad_multiple must be a positive multiple of 32")
    real, padded, blocks = [0], [0], []
    for expert, count in enumerate(counts):
        length = ((count + pad_multiple - 1) // pad_multiple) * pad_multiple
        real.append(real[-1] + count)
        padded.append(padded[-1] + length)
        blocks.extend((expert, i) for i in range(length // pad_multiple))
    if padded[-1] > 2**31 - 1:
        raise ValueError("padded token count exceeds int32 metadata")
    flat = real + padded + [v for pair in blocks for v in pair]
    # Keep version counters even when the caller is in inference_mode.
    with torch.inference_mode(False):
        data = torch.tensor(flat, dtype=torch.int32, device=device)
        n = len(real)
        tensors = (data[:n], data[n : 2 * n], data[2 * n :].reshape(-1, 2))
    return MXFP4WgradMetadata(
        counts,
        pad_multiple,
        real[-1],
        padded[-1],
        *tensors,
        data.data_ptr(),
        tuple(tensor_version(t) for t in tensors),
    )


def validate_mxfp4_wgrad_metadata(metadata, counts, device, block_k):
    """Reject stale routing, structural corruption and tracked in-place mutation."""
    if not isinstance(metadata, MXFP4WgradMetadata):
        raise TypeError("metadata must come from prepare_mxfp4_wgrad_metadata")
    counts = normalize_counts(counts)
    if metadata.counts != counts:
        raise ValueError("metadata routing does not match inputs")
    pad = metadata.pad_multiple
    if type(pad) is not int or pad <= 0 or pad % 32 or pad % block_k:
        raise ValueError("metadata padding must divide into consumer BLOCK_SIZE_K")
    total = sum(counts)
    padded = sum(((count + pad - 1) // pad) * pad for count in counts)
    if metadata.total_rows != total or metadata.padded_rows != padded:
        raise ValueError("metadata real/padded totals do not match counts")
    e = len(counts)
    specs = (
        (metadata.real_offsets, (e + 1,), 0),
        (metadata.padded_offsets, (e + 1,), e + 1),
        (metadata.prep_blocks, (padded // pad, 2), 2 * (e + 1)),
    )
    for (tensor, shape, offset), version in zip(specs, metadata.tensor_versions):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != shape
            or tensor.dtype != torch.int32
            or tensor.device != device
            or not tensor.is_contiguous()
            or tensor.storage_offset() != offset
        ):
            raise ValueError(
                "metadata tensors must retain their int32 shape/device/layout"
            )
        if tensor.untyped_storage().data_ptr() != metadata.storage_base:
            raise ValueError("metadata tensor storage does not match its producer")
        if version is None or tensor_version(tensor) != version:
            raise ValueError("metadata tensor was mutated after creation")
    if len(metadata.tensor_versions) != len(specs):
        raise ValueError("metadata version provenance is incomplete")
    return metadata


def get_mxfp4_wgrad_stats():
    """Return host enqueue counters; trace/ISA evidence proves actual GPU execution."""
    return copy.deepcopy(_STATS)


def reset_mxfp4_wgrad_stats():
    """Reset counters without changing tensors or kernel configuration."""
    for key in ("quant_calls", "wgrad_calls", "empty_experts"):
        _STATS[key] = 0
    _STATS["backend_calls"].clear()
    _STATS["fallbacks"].clear()
    _STATS["last_config"] = None


def record_mxfp4_wgrad_fallback(reason):
    """Record a caller's explicit preflight fallback; this module never retries."""
    reason = str(reason)
    _STATS["fallbacks"][reason] = _STATS["fallbacks"].get(reason, 0) + 1


def record_mxfp4_wgrad_launch(
    config,
    is_tuned,
    metadata,
    launched=True,
    backend="triton",
    logical_shape=None,
    physical_shape=None,
    transpose_output=None,
    operand_roles=None,
):
    _STATS["wgrad_calls"] += 1
    _STATS["empty_experts"] += metadata.counts.count(0)
    calls = _STATS["backend_calls"]
    label = backend + "_e2m1_e2m1" if launched else "empty_no_launch"
    calls[label] = calls.get(label, 0) + 1
    _STATS["last_config"] = {
        "config": dict(config),
        "backend": backend,
        "is_tuned": is_tuned,
        "padding": metadata.pad_multiple,
        "real_rows": metadata.total_rows,
        "padded_rows": metadata.padded_rows,
        "logical_shape": logical_shape,
        "physical_shape": physical_shape,
        "logical_transpose_output": transpose_output,
        "physical_operand_roles": operand_roles,
    }


def record_mxfp4_colwise_quant():
    _STATS["quant_calls"] += 1
