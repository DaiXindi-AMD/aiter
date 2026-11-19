# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical references shared by Wgrad tests and benchmarks."""

import torch


def decode_columnwise(pair):
    """Decode each nibble and its independent 1x32 E8M0 scale into FP32."""
    payload, scales = pair
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=payload.device,
    )
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(
        payload.shape[0], -1
    )
    return table[codes.long()] * torch.exp2(scales.float() - 127).repeat_interleave(
        32, dim=1
    )


def same_fp4_wgrad_reference(x_pack, dy_pack, metadata, *, transpose_output=False):
    """Multiply the exact packed operands, excluding only zero padded tokens."""
    x, dy = decode_columnwise(x_pack), decode_columnwise(dy_pack)
    result = []
    offset = 0
    for count in metadata.counts:
        z = dy[:, offset : offset + count] @ x[:, offset : offset + count].T
        result.append(z.T if transpose_output else z)
        offset += (
            (count + metadata.pad_multiple - 1) // metadata.pad_multiple
        ) * metadata.pad_multiple
    return torch.stack(result)


def bf16_wgrad_reference(x, dy, counts, *, transpose_output=False):
    """FP32 mathematical reference using the original BF16 values."""
    result, offset = [], 0
    for count in counts:
        z = dy[offset : offset + count].float().T @ x[offset : offset + count].float()
        result.append(z.T if transpose_output else z)
        offset += count
    return torch.stack(result)


def error_metrics(reference, candidate):
    """Return SNR, relative RMS error, max error and gradient norm ratio."""
    reference, candidate = reference.float(), candidate.float()
    signal = reference.square().sum()
    noise = (candidate - reference).square().sum()
    tiny = torch.finfo(torch.float32).tiny
    return {
        "snr_db": float(
            (
                10
                * (
                    torch.log10(signal.clamp_min(tiny))
                    - torch.log10(noise.clamp_min(tiny))
                )
            ).item()
        ),
        "relative_rms": float(torch.sqrt(noise / signal.clamp_min(tiny)).item()),
        "max_abs": float((candidate - reference).abs().max().item()),
        "norm_ratio": float(
            torch.sqrt(candidate.square().sum() / signal.clamp_min(tiny)).item()
        ),
    }
