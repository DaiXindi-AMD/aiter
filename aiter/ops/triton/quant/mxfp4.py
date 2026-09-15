# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.quant.mxfp4 import (
    _dequant_hadamard_quant_mxfp4_kernel,
)
from aiter.utility import dtypes

__all__ = ["dequant_hadamard_quant_mxfp4"]

_SIGN_VALIDATED_VERSION = "_aiter_mxfp4_h16_sign_validated_version"


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except (AttributeError, RuntimeError):
        return None


def _prepare_h16_sign(
    sign_vector: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(sign_vector, torch.Tensor):
        raise TypeError("sign_vector must be a torch.Tensor")
    if sign_vector.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            "sign_vector must have dtype torch.float32 or torch.bfloat16, "
            f"got {sign_vector.dtype}"
        )
    if sign_vector.device != reference.device:
        raise ValueError("sign_vector must be on the same device as data_fp4")
    if sign_vector.numel() != 16:
        raise ValueError("sign_vector must contain 16 values")

    version = _tensor_version(sign_vector)
    if (
        version is None
        or getattr(sign_vector, _SIGN_VALIDATED_VERSION, None) != version
    ):
        valid = torch.all(
            torch.isfinite(sign_vector) & ((sign_vector == 1) | (sign_vector == -1))
        )
        message = "sign_vector values must be finite and exactly -1 or 1"
        with torch.cuda.device(reference.device):
            is_capturing = torch.cuda.is_current_stream_capturing()
        if is_capturing:
            assert_async = getattr(torch, "_assert_async", None)
            if assert_async is None:
                raise RuntimeError(
                    "sign_vector must be validated before CUDA graph capture"
                )
            assert_async(valid, message)
        elif not valid.item():
            raise ValueError(message)
        if version is not None:
            setattr(sign_vector, _SIGN_VALIDATED_VERSION, version)

    return sign_vector.reshape(-1).contiguous()


def dequant_hadamard_quant_mxfp4(
    data_fp4: torch.Tensor,
    scales: torch.Tensor,
    sign_vector: torch.Tensor,
    block_size: int = 32,
    g: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse packed MXFP4 dequantization, transpose, H16, and RTN requantization.

    The input represents a logical ``(M, K)`` matrix with one row-major E8M0
    scale per 32 values. The result represents ``H16(data.T)`` with signed,
    normalized Hadamard groups along the transposed matrix's final dimension.

    Args:
        data_fp4: 2-D packed E2M1 tensor of shape ``(M, K // 2)``.
            Raw ``uint8`` and canonical ``float4_e2m1fn_x2`` are accepted.
        scales: Row-major E8M0 tensor of shape ``(M, K // 32)``.
            Raw ``uint8`` and canonical ``float8_e8m0fnu`` are accepted.
        sign_vector: A same-device float32 or bfloat16 vector containing exactly
            16 finite values, each equal to ``-1`` or ``1``.
        block_size: MXFP4 scale block size. Only 32 is supported.
        g: Hadamard group size. Only 16 is supported.

    Returns:
        ``(output_fp4, output_scales)`` in canonical row-major byte layouts,
        with shapes ``(K, M // 2)`` and ``(K, M // 32)`` respectively.

    The operation uses deterministic E8M0 EVEN scale selection and E2M1
    round-to-nearest-even payload conversion. Stochastic rounding and GEMM
    layout shuffles are intentionally outside this operator. Decoded values
    are narrowed to bfloat16 before H16, matching Lumen's fused training path.
    An input E8M0 NaN sentinel (raw 255) poisons the corresponding complete
    32-value output scale block as raw 255 with a zero payload.
    """
    if block_size != 32:
        raise ValueError(f"block_size must be 32, got {block_size}")
    if g != 16:
        raise ValueError(f"g must be 16, got {g}")
    if data_fp4.dim() != 2:
        raise ValueError(f"data_fp4 must be 2-D, got {data_fp4.dim()}-D")
    if data_fp4.dtype not in (torch.uint8, dtypes.fp4x2):
        raise TypeError(
            "data_fp4 must have dtype torch.uint8 or "
            f"torch.float4_e2m1fn_x2, got {data_fp4.dtype}"
        )
    data_bytes = data_fp4.view(torch.uint8)
    M, K_packed = data_bytes.shape
    K = K_packed * 2
    if M == 0 or K == 0:
        raise ValueError(f"logical input dimensions must be non-zero, got ({M}, {K})")
    if M % block_size != 0 or K % block_size != 0:
        raise ValueError(f"logical input shape must be divisible by 32, got ({M}, {K})")

    if scales.dim() != 2:
        raise ValueError(f"scales must be 2-D, got {scales.dim()}-D")
    if scales.dtype not in (torch.uint8, dtypes.fp8_e8m0):
        raise TypeError(
            "scales must have dtype torch.uint8 or torch.float8_e8m0fnu, "
            f"got {scales.dtype}"
        )
    expected_scale_shape = (M, K // block_size)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales shape must be {expected_scale_shape}, got {tuple(scales.shape)}"
        )
    scale_bytes = scales.view(torch.uint8)
    if scales.device != data_fp4.device:
        raise ValueError("scales must be on the same device as data_fp4")
    if not data_fp4.is_cuda:
        raise ValueError("data_fp4, scales, and sign_vector must be CUDA tensors")

    sign_flat = _prepare_h16_sign(sign_vector, data_fp4)

    output = torch.empty((K, M // 2), dtype=torch.uint8, device=data_fp4.device)
    output_scales = torch.empty(
        (K, M // block_size), dtype=torch.uint8, device=data_fp4.device
    )

    grid = (M // block_size, K // block_size)
    _dequant_hadamard_quant_mxfp4_kernel[grid](
        data_bytes,
        scale_bytes,
        sign_flat,
        output,
        output_scales,
        *data_bytes.stride(),
        *scale_bytes.stride(),
        *output.stride(),
        *output_scales.stride(),
        BLOCK_SIZE=block_size,
        HADAMARD_SIZE=g,
    )
    return output, output_scales
