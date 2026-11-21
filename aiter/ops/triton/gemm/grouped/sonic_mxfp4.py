# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental MXFP4-weight grouped GEMM with BF16, MXFP4 or MXFP8 activations.

Weight operand is always E2M1 (microscaling FP4) packed as uint8 along K.
Activation formats:
  * ``bf16``  -> A16W4, weight-only. The weight dequant is folded into a BF16
                 MFMA, so this path gets no tensor-core throughput benefit.
  * ``mxfp4`` -> A4W4, native E2M1 x E2M1 f8f6f4 MFMA (4x BF16 peak).
  * ``mxfp8`` -> A8W4, native E4M3 x E2M1 f8f6f4 MFMA (2x BF16 peak).
"""

import functools
import types

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
    _grouped_gemm_kernel,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant, dynamic_mxfp8_quant
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.mxfp4_wgrad import (
    MXFP4RowwisePack,
    tensor_version,
)

__all__ = [
    "ACTIVATION_FORMATS",
    "block_shape",
    "family_for",
    "grouped_mxfp4_weight_gemm",
    "pack_grouped_mxfp4_weights",
    "prepare_expert_offsets",
    "prepare_grouped_gemm_metadata",
    "quantize_grouped_activation",
]

ACTIVATION_FORMATS = ("bf16", "mxfp4", "mxfp8")

# One config family per activation format so A8W4 can be tuned independently.
_FAMILY = {
    "bf16": "GEMM-SONIC-A16W4",
    "mxfp4": "GEMM-SONIC-A4W4",
    "mxfp8": "GEMM-SONIC-A8W4",
}


@functools.lru_cache(maxsize=1024)
def _resolve_cfg(family, m, n, k, e):
    """Cache the resolved launch config.

    ``m`` must be a *bucket* (see :func:`_m_bucket`), not the live max token count.
    The dispatcher produces a different max-per-expert every layer and every
    microbatch, so keying on the raw value thrashed this cache and re-ran
    get_gemm_config (JSON load + deep copy) up to 8 times per module call, which
    was measurable host time. Only the tile shape matters here and it comes from
    the family's DEFAULT entry whenever no specialised key matches.
    """
    cfg, _is_tuned = get_gemm_config(family, m, n, k, B=e, backend="triton")
    return types.MappingProxyType(dict(cfg))


def _m_bucket(m):
    """Coarse, stable M for config lookup (the config never depends on exact M)."""
    return triton.next_power_of_2(max(int(m), 32))


def block_shape(family, m, n, k, e):
    """(BLOCK_SIZE_M, BLOCK_SIZE_N) the family will select for this shape."""
    cfg = _resolve_cfg(family, _m_bucket(m), n, k, e)
    return cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]


def family_for(activation_format):
    """Config family name for an activation format (bf16 / mxfp4 / mxfp8)."""
    if activation_format not in ACTIVATION_FORMATS:
        raise ValueError(f"activation_format must be one of {ACTIVATION_FORMATS}")
    return _FAMILY[activation_format]


def pack_grouped_mxfp4_weights(weight):
    """Pack CUDA BF16 [E,N,K] along K into uint8 payload/scales; K must be a multiple of32."""
    if weight.ndim != 3 or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        raise ValueError("weight must be CUDA BF16 [E,N,K]")
    e, n, k = weight.shape
    if min(e, n, k) <= 0 or k % 32:
        raise ValueError("positive E/N/K required; K must be a multiple of32")
    payload, scales = dynamic_mxfp4_quant(weight.contiguous().reshape(-1, k))
    return (
        payload.reshape(e, n, k // 2).contiguous(),
        scales.reshape(e, n, k // 32).contiguous(),
    )


def quantize_grouped_activation(x, activation_format):
    """Quantize BF16 [R,K] activations for the requested MX format.

    Returns the tensor the kernel consumes plus its scale tensor (None for BF16).
    """
    if activation_format not in ACTIVATION_FORMATS:
        raise ValueError(f"activation_format must be one of {ACTIVATION_FORMATS}")
    if activation_format == "bf16":
        return x, None
    if activation_format == "mxfp4":
        payload, scales = dynamic_mxfp4_quant(x)
        return payload, scales.contiguous()
    payload, scales = dynamic_mxfp8_quant(x)
    # dynamic_mxfp8_quant returns (E4M3 [R,K], e8m0 [R,K/32]); the scale layout is
    # already row-major but the helper does not promise contiguity.
    return payload, scales.contiguous()


def prepare_expert_offsets(counts, device, device_counts=None):
    """Cumulative [E+1] row offsets for host ``counts``.

    Independent of the GEMM block shape, so one call serves every GEMM that shares
    the same routing (forward F1/F2 and backward D1/D2). A validated device copy
    avoids a blocking H2D transfer of host counts.
    """
    e = len(counts)
    if e <= 0:
        raise ValueError("counts must be non-empty")
    if device_counts is None:
        counts_dev = torch.as_tensor(counts, dtype=torch.int32, device=device)
    else:
        if (
            not isinstance(device_counts, torch.Tensor)
            or device_counts.shape != (e,)
            or device_counts.device != torch.device(device)
            or device_counts.dtype not in (torch.int32, torch.int64)
            or not device_counts.is_contiguous()
        ):
            raise ValueError("device_counts must be a contiguous GPU integer vector")
        counts_dev = device_counts.to(dtype=torch.int32)
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(counts_dev, 0, dtype=torch.int32),
        )
    )


def prepare_grouped_gemm_metadata(counts, device, block_m, block_n, offsets=None):
    """Build the kernel's expert block map from host counts.

    Callers that launch several GEMMs with the same block shape for one layer
    should build this once and pass it as ``metadata=`` to every call; that removes
    one blocking ``torch.as_tensor`` plus two tiny cumsum launches per GEMM.

    The returned tensors are read-only scratch: they are not reused across calls
    with different ``counts``.
    """
    if offsets is None:
        offsets = prepare_expert_offsets(counts, device)
    return {
        "counts_dev": offsets[1:] - offsets[:-1],
        "offsets": offsets,
        "block_m": block_m,
        "block_n": block_n,
        "e": len(counts),
    }


def _expert_block_base(meta, n):
    """Cumulative per-expert block counts for this N, cached on the metadata."""
    key = ("block_base", n)
    cached = meta.get(key)
    if cached is not None:
        return cached
    block_n = meta["block_n"]
    blocks_per_expert = torch.div(
        meta["counts_dev"] + meta["block_m"] - 1, meta["block_m"], rounding_mode="floor"
    ) * triton.cdiv(n, block_n)
    base = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=meta["counts_dev"].device),
            torch.cumsum(blocks_per_expert, 0, dtype=torch.int32),
        )
    )
    meta[key] = base
    return base


def _validate_mapped_activation(pack, source):
    if pack.source is not source:
        raise ValueError("mapped activation source tensor does not match")
    if pack.source_version is None or tensor_version(source) != pack.source_version:
        raise ValueError("mapped activation source was mutated")
    source_map = pack.source_map
    if (
        not isinstance(source_map, torch.Tensor)
        or source_map.ndim != 1
        or source_map.dtype != torch.int32
        or source_map.device != source.device
        or not source_map.is_contiguous()
    ):
        raise ValueError("mapped activation requires a contiguous int32 source map")
    if (
        pack.source_map_version is None
        or tensor_version(source_map) != pack.source_map_version
    ):
        raise ValueError("mapped activation source map was mutated")
    shapes = (
        (source_map.numel(), source.shape[1] // 2),
        (source_map.numel(), source.shape[1] // 32),
    )
    for tensor, shape, version in zip(pack, shapes, pack.tensor_versions):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.uint8
            or tensor.device != source.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("invalid mapped rowwise activation pack")
        if version is None or tensor_version(tensor) != version:
            raise ValueError("mapped rowwise activation pack was mutated")
    if len(pack.tensor_versions) != 2:
        raise ValueError("mapped activation version provenance is incomplete")
    return source_map.numel()


def grouped_mxfp4_weight_gemm(
    x,
    payload,
    scales,
    counts,
    out=None,
    *,
    quantize_activation=False,
    activation_format=None,
    metadata=None,
    quantized_activation=None,
):
    """Compute BF16 [R,K] @ packed [E,N,K].T into BF16 [R,N].

    counts is an immutable host sequence of nonnegative per-expert rows summing to R.
    Inputs/out must be contiguous on one CUDA device; overlapping output is rejected.

    ``activation_format`` selects bf16 / mxfp4 / mxfp8. ``quantize_activation=True``
    is the legacy spelling of ``activation_format="mxfp4"``.
    ``quantized_activation`` lets a caller pass an already-quantized (payload,
    scales) pair, which avoids re-quantizing the same activation. A mapped
    rowwise pack binds an unsorted source to expert-major quantized rows.
    ``metadata`` is a dict from :func:`prepare_grouped_gemm_metadata`; it must have
    been built with the block shape this family resolves for (n, k, e).
    """
    if activation_format is None:
        activation_format = "mxfp4" if quantize_activation else "bf16"
    if activation_format not in ACTIVATION_FORMATS:
        raise ValueError(f"activation_format must be one of {ACTIVATION_FORMATS}")
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous CUDA BF16 [R,K]")
    if payload.ndim != 3:
        raise ValueError("payload must be [E,N,K/2]")
    e, n, half_k = payload.shape
    source_rows, k = x.shape
    mapped_activation = isinstance(quantized_activation, MXFP4RowwisePack)
    if mapped_activation:
        if activation_format != "mxfp4":
            raise ValueError("mapped activation packs require activation_format=mxfp4")
        rows = _validate_mapped_activation(quantized_activation, x)
    else:
        rows = source_rows
    if min(e, n, k) <= 0 or k % 32 or half_k * 2 != k:
        raise ValueError("invalid packed dimensions")
    if scales.shape != (e, n, k // 32):
        raise ValueError("scales must be [E,N,K/32]")
    for name, tensor in (("payload", payload), ("scales", scales)):
        if (
            tensor.dtype != torch.uint8
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous uint8 on x.device")
    if not isinstance(counts, (tuple, list)) or len(counts) != e:
        raise ValueError("counts must contain E host integers")
    if any(type(c) is not int or c < 0 for c in counts) or sum(counts) != rows:
        raise ValueError("counts must be nonnegative and sum to R")
    if out is None:
        out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if (
        out.shape != (rows, n)
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 [R,N] on x.device")
    overlap_inputs = [x, payload, scales]
    if mapped_activation:
        overlap_inputs.extend((*quantized_activation, quantized_activation.source_map))
    if any(torch._C._overlaps(out, t) for t in overlap_inputs):
        raise ValueError("out must not overlap any input")
    if rows == 0:
        return out
    family = _FAMILY[activation_format]
    cfg = _resolve_cfg(family, _m_bucket(max(counts)), n, k, e)
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    if bk % 32:
        raise ValueError("BLOCK_SIZE_K must be a multiple of32")
    if cfg["NUM_KSPLIT"] != 1:
        raise ValueError("split-K is unsupported in this experiment")

    if quantized_activation is not None:
        kernel_x, a_scales = quantized_activation
    elif activation_format == "bf16":
        kernel_x, a_scales = x, None
    else:
        kernel_x, a_scales = quantize_grouped_activation(x, activation_format)
    if activation_format == "mxfp4" and kernel_x.shape != (rows, k // 2):
        raise ValueError("quantized activation must be [R,K/2] for mxfp4")
    if activation_format == "mxfp8" and kernel_x.shape != (rows, k):
        raise ValueError("quantized activation must be [R,K] for mxfp8")

    if metadata is None:
        metadata = prepare_grouped_gemm_metadata(counts, x.device, bm, bn)
    elif (metadata["block_m"], metadata["block_n"]) != (bm, bn):
        raise ValueError("metadata block shape does not match the resolved config")
    offsets = metadata["offsets"]
    expert_block_base = _expert_block_base(metadata, n)
    block_map_iters = max(1, (e - 1).bit_length())
    grid = sum(triton.cdiv(c, bm) * triton.cdiv(n, bn) for c in counts)
    _grouped_gemm_kernel[(grid,)](
        kernel_x,
        payload,
        out,
        offsets,
        out,
        offsets,
        offsets,
        kernel_x.stride(0),
        kernel_x.stride(1),
        payload.stride(0),
        payload.stride(2),
        payload.stride(1),
        out.stride(0),
        out.stride(1),
        0,
        0,
        N=n,
        K=k,
        E=e,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        HAS_BIAS=False,
        HAS_GATHER_IDX=False,
        HAS_SCATTER_IDX=False,
        B_scales_ptr=scales,
        W4_WEIGHT=True,
        CACHE_MODIFIER=cfg["cache_modifier"],
        A_scales_ptr=a_scales,
        A4_ACTIVATION=(activation_format == "mxfp4"),
        A8_ACTIVATION=(activation_format == "mxfp8"),
        expert_block_base_ptr=expert_block_base,
        BLOCK_MAP_ITERS=block_map_iters,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
    )
    return out
