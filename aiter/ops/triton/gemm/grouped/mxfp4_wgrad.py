# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public variable-K Wgrad from two MXFP4 columnwise operands."""

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.grouped.mxfp4_wgrad import (
    _grouped_mxfp4_wgrad_kernel,
)
from aiter.ops.triton.quant.grouped_mxfp4_colwise import (
    quantize_grouped_mxfp4_colwise,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.mxfp4_wgrad import (
    MXFP4ColwisePack,
    build_mxfp4_wgrad_metadata,
    normalize_counts,
    record_mxfp4_wgrad_launch,
    tensor_version,
    validate_mxfp4_wgrad_metadata,
)

try:
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.grouped.mxfp4_wgrad import (
        _grouped_mxfp4_wgrad_gluon_kernel,
    )
except ImportError:
    _grouped_mxfp4_wgrad_gluon_kernel = None

MXFP4_WGRAD_BACKENDS = ("triton", "gluon")

__all__ = [
    "MXFP4_WGRAD_BACKENDS",
    "check_mxfp4_wgrad_support",
    "grouped_mxfp4_wgrad",
    "prepare_mxfp4_wgrad_metadata",
]


def _get_config(m, n=None, backend="triton"):
    return get_gemm_config("GEMM-MXFP4-WGRAD", m, N=n, backend=backend)


def check_mxfp4_wgrad_support(
    device, m, n, *, backend="triton", transpose_output=False
):
    """Validate native gfx950 support and the external config before output writes.

    M is dY's feature count; N is X's feature count. Return (config, is_tuned).
    This is a host preflight, not a kernel-execution or ISA check.
    The Gluon candidate supports BM256/BN128/BK256/eight waves; config selection uses physical orientation.
    """
    if backend not in MXFP4_WGRAD_BACKENDS:
        raise ValueError("backend must be triton or gluon")
    if backend == "gluon" and _grouped_mxfp4_wgrad_gluon_kernel is None:
        raise NotImplementedError("Gluon scaled-MFMA backend is not available")
    if m <= 0 or n <= 0:
        raise ValueError("positive feature counts required")
    if type(transpose_output) is not bool:
        raise TypeError("transpose_output must be bool")
    with torch.cuda.device(device):
        if get_arch() != "gfx950":
            raise NotImplementedError("native MXFP4 Wgrad requires gfx950")
        physical_m, physical_n = (
            (n, m) if backend == "gluon" and transpose_output else (m, n)
        )
        cfg, tuned = _get_config(physical_m, physical_n, backend=backend)
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
        value = cfg[key]
        if type(value) is not int or value <= 0 or value & (value - 1):
            raise ValueError("Wgrad tile sizes must be positive powers of two")
    if cfg["BLOCK_SIZE_K"] < 128 or cfg["BLOCK_SIZE_K"] % 32:
        raise ValueError(
            "native scaled-dot requires BLOCK_SIZE_K >= 128, divisible by 32"
        )
    if cfg["NUM_KSPLIT"] != 1:
        raise NotImplementedError("MXFP4 Wgrad has no split-K path")
    if cfg["GROUP_SIZE_M"] <= 0 or cfg["PERSISTENT"] not in (0, 1):
        raise ValueError("invalid Wgrad tile walk config")
    if cfg["GRID_MULTIPLIER"] <= 0:
        raise ValueError("GRID_MULTIPLIER must be positive")
    if cfg["ALIGNMENT_HINTS"] not in (0, 1):
        raise ValueError("ALIGNMENT_HINTS must be 0 or 1")
    if backend == "gluon":
        fixed = (
            cfg["BLOCK_SIZE_M"],
            cfg["BLOCK_SIZE_N"],
            cfg["BLOCK_SIZE_K"],
            cfg["num_warps"],
            cfg["num_stages"],
            cfg["matrix_instr_nonkdim"],
        )
        if fixed != (256, 128, 256, 8, 2, 32) or cfg["PERSISTENT"]:
            raise NotImplementedError(
                "Gluon BN128 candidate requires BM256/BN128/BK256, 8 warps, stage2, nonKdim32"
            )
    return cfg, tuned


def prepare_mxfp4_wgrad_metadata(
    counts, device, *, pad_multiple=None, backend="triton"
):
    """Build immutable routing metadata, padding each expert to the consumer K tile.

    counts is a non-empty host integer sequence. Empty experts have zero padded
    length. The metadata is per routing/microbatch, never an optimizer-step cache.
    """
    if backend not in MXFP4_WGRAD_BACKENDS:
        raise ValueError("backend must be triton or gluon")
    if pad_multiple is None:
        with torch.cuda.device(device):
            cfg, _ = _get_config(0, backend=backend)
        pad_multiple = cfg["BLOCK_SIZE_K"]
    return build_mxfp4_wgrad_metadata(counts, device, pad_multiple)


def _validate_pack(pair, source, metadata):
    if not isinstance(pair, MXFP4ColwisePack):
        raise TypeError("quantized operand must be a metadata-bound colwise pack")
    if pair.metadata is not metadata:
        raise ValueError("quantized operand producer metadata does not match")
    if pair.source is not source:
        raise ValueError("quantized operand source tensor does not match")
    if pair.source_version is None or tensor_version(source) != pair.source_version:
        raise ValueError(
            "quantized operand source was mutated or has no version counter"
        )
    source_map = pair.source_map
    if source_map is None:
        if source.shape[0] != metadata.total_rows:
            raise ValueError("unmapped quantized operand rows do not match routing")
        if pair.source_map_version is not None:
            raise ValueError("quantized operand map provenance is incomplete")
    else:
        if (
            not isinstance(source_map, torch.Tensor)
            or source_map.shape != (metadata.total_rows,)
            or source_map.dtype != torch.int32
            or source_map.device != source.device
            or not source_map.is_contiguous()
        ):
            raise ValueError("invalid contiguous int32 quantized operand source map")
        if (
            pair.source_map_version is None
            or tensor_version(source_map) != pair.source_map_version
        ):
            raise ValueError("quantized operand source map was mutated")
    shapes = (
        (source.shape[1], metadata.padded_rows // 2),
        (source.shape[1], metadata.padded_rows // 32),
    )
    for tensor, shape, version in zip(pair, shapes, pair.tensor_versions):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != shape
            or tensor.dtype != torch.uint8
            or tensor.device != source.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("invalid contiguous uint8 columnwise operand")
        if version is None or tensor_version(tensor) != version:
            raise ValueError("quantized operand payload/scales were mutated")
    if len(pair.tensor_versions) != 2:
        raise ValueError("quantized operand version provenance is incomplete")
    return pair


def grouped_mxfp4_wgrad(
    x,
    dy,
    counts,
    *,
    out=None,
    accumulate=False,
    metadata=None,
    quantized_operands=None,
    transpose_output=False,
    backend="triton",
):
    """Compute dW[e] = dY[e].T @ X[e] with E2M1 data and independent E8M0 scales.

    X[RX,N] and dY[RY,M] are contiguous finite CUDA BF16. Without prequantized
    operands RX=RY=sum(counts). Mapped producer packs can bind each source to
    an expert-major row map of length sum(counts). Return FP32 [E,M,N], or
    [E,N,M] with transpose_output. out is non-overlapping contiguous FP32.
    accumulate=True requires out. Empty experts overwrite with zero or retain
    their accumulated value.

    quantized_operands=(X_pack,dY_pack) must be bound producer objects with the
    same metadata and unchanged source tensors/versions. Raw storage/.data writes
    are unsupported; these packs are single-routing-lived, not generation caches.
    Padding/tile walk come from JSON. Calls on one output require one ordered stream.
    backend selects triton or gluon without cross-backend fallback. Gluon v4
    requires BM256/BN128/BK256, eight warps, buffers below 4 GiB. No precision fallback.
    """
    if type(accumulate) is not bool or type(transpose_output) is not bool:
        raise TypeError("accumulate and transpose_output must be bool")
    for tensor in (x, dy):
        if (
            tensor.ndim != 2
            or not tensor.is_cuda
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            raise ValueError("operands must be contiguous CUDA BF16 matrices")
    if x.device != dy.device:
        raise ValueError("X and dY must be on the same device")
    counts = normalize_counts(counts)
    m, n, e = dy.shape[1], x.shape[1], len(counts)
    cfg, tuned = check_mxfp4_wgrad_support(
        x.device, m, n, backend=backend, transpose_output=transpose_output
    )
    if metadata is None:
        if quantized_operands is not None:
            raise ValueError("prequantized operands require their producer metadata")
        if x.shape[0] != dy.shape[0] or x.shape[0] != sum(counts):
            raise ValueError("unquantized X and dY rows must match counts")
        metadata = prepare_mxfp4_wgrad_metadata(
            counts, x.device, pad_multiple=cfg["BLOCK_SIZE_K"], backend=backend
        )
    validate_mxfp4_wgrad_metadata(metadata, counts, x.device, cfg["BLOCK_SIZE_K"])
    if quantized_operands is None and (
        x.shape[0] != dy.shape[0] or x.shape[0] != metadata.total_rows
    ):
        raise ValueError("unquantized X and dY rows must match routing")
    shape = (e, n, m) if transpose_output else (e, m, n)
    if out is None:
        if accumulate:
            raise ValueError("accumulate requires initialized FP32 out")
        out = torch.empty(shape, dtype=torch.float32, device=x.device)
    if (
        out.shape != shape
        or out.dtype != torch.float32
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous FP32 with the requested shape/device")
    inputs = [
        x,
        dy,
        metadata.real_offsets,
        metadata.padded_offsets,
        metadata.prep_blocks,
    ]
    if quantized_operands is not None:
        if (
            not isinstance(quantized_operands, (tuple, list))
            or len(quantized_operands) != 2
        ):
            raise ValueError("quantized_operands must be (X_pack, dY_pack)")
        xp = _validate_pack(quantized_operands[0], x, metadata)
        dyp = _validate_pack(quantized_operands[1], dy, metadata)
        if xp.source_map is not dyp.source_map:
            raise ValueError("quantized operand source maps do not match")
        inputs.extend((*xp, *dyp))
        for pair in (xp, dyp):
            if pair.source_map is not None:
                inputs.append(pair.source_map)
    if any(torch._C._overlaps(out, tensor) for tensor in inputs):
        raise ValueError("out must not overlap inputs or metadata")
    if metadata.padded_rows == 0:
        if not accumulate:
            out.zero_()
        record_mxfp4_wgrad_launch(cfg, tuned, metadata, launched=False, backend=backend)
        return out
    if quantized_operands is None:
        xp = quantize_grouped_mxfp4_colwise(x, metadata)
        dyp = quantize_grouped_mxfp4_colwise(dy, metadata)
    physical_m, physical_n = (
        (n, m) if backend == "gluon" and transpose_output else (m, n)
    )
    tiles = (
        e
        * triton.cdiv(physical_m, cfg["BLOCK_SIZE_M"])
        * triton.cdiv(physical_n, cfg["BLOCK_SIZE_N"])
    )
    if backend == "gluon":
        if any(t.numel() * t.element_size() >= 2**32 for t in (*xp, *dyp, out)):
            raise NotImplementedError(
                "Gluon buffer ops require each contiguous buffer span below 4 GiB"
            )
        # X.T @ dY writes the transposed logical result without transposing a FP32 tile.
        left, right = (xp, dyp) if transpose_output else (dyp, xp)
        _grouped_mxfp4_wgrad_gluon_kernel[(tiles,)](
            left[0],
            left[1],
            right[0],
            right[1],
            out,
            metadata.padded_offsets,
            xp[0].stride(0),
            xp[1].stride(0),
            M=physical_m,
            N=physical_n,
            E=e,
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            ACCUMULATE=accumulate,
            TRANSPOSE_OUTPUT=False,
            ALIGNMENT_HINTS=bool(cfg["ALIGNMENT_HINTS"]),
            CONFIG_NUM_WARPS=cfg["num_warps"],
            CONFIG_NUM_STAGES=cfg["num_stages"],
            CONFIG_MATRIX_INSTR_NONKDIM=cfg["matrix_instr_nonkdim"],
            CACHE_MODIFIER=cfg["cache_modifier"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            waves_per_eu=cfg["waves_per_eu"],
            matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
        )
        record_mxfp4_wgrad_launch(
            cfg,
            tuned,
            metadata,
            backend=backend,
            logical_shape=(e, m, n),
            physical_shape=(e, physical_m, physical_n),
            transpose_output=transpose_output,
            operand_roles=(
                ("X_colwise", "dY_colwise")
                if transpose_output
                else ("dY_colwise", "X_colwise")
            ),
        )
        return out
    programs = tiles
    if cfg["PERSISTENT"]:
        with torch.cuda.device(x.device):
            programs = min(tiles, get_num_sms() * cfg["GRID_MULTIPLIER"])
    _grouped_mxfp4_wgrad_kernel[(programs,)](
        dyp[0],
        dyp[1],
        xp[0],
        xp[1],
        out,
        metadata.padded_offsets,
        xp[0].stride(0),
        xp[1].stride(0),
        programs,
        M=m,
        N=n,
        E=e,
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        ACCUMULATE=accumulate,
        TRANSPOSE_OUTPUT=transpose_output,
        PERSISTENT=bool(cfg["PERSISTENT"]),
        ALIGNMENT_HINTS=bool(cfg["ALIGNMENT_HINTS"]),
        CACHE_MODIFIER=cfg["cache_modifier"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
        CONFIG_NUM_WARPS=cfg["num_warps"],
        CONFIG_NUM_STAGES=cfg["num_stages"],
        CONFIG_WAVES_PER_EU=cfg["waves_per_eu"],
        CONFIG_MATRIX_INSTR_NONKDIM=cfg["matrix_instr_nonkdim"],
    )
    record_mxfp4_wgrad_launch(cfg, tuned, metadata)
    return out
