# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public-bundle fixtures and independent tensor oracles for A0p tests."""

import math
import os

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    mxfp4_dense_bind_v1,
    mxfp4_dense_prepare_kernel_v1,
)
from aiter.ops.triton.utils.config_utils import resolve_config_dir
from aiter.ops.triton.utils.mxfp4_contract import buffer_shapes

ROWS = (0, 1, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 257)


def prepare_linear(k, mode=1, dtype=1):
    device = torch.cuda.current_device()
    cu = torch.cuda.get_device_properties(device).multi_processor_count
    spec = (1, 0, 950, cu, 1, 4096, k, 257, 0, mode, dtype, 1, 2, 1, 1, 1, 2, 0, 1)
    path = os.path.join(
        resolve_config_dir("gemm", "GEMM-AFP4WFP4"),
        "GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json",
    )
    return mxfp4_dense_prepare_kernel_v1(spec, path)


@pytest.fixture(scope="module", params=(2048, 4096))
def prepared_linear(request):
    return prepare_linear(request.param)


def rng_receipt(seed, base, count):
    mask = (1 << 32) - 1
    return (
        1,
        seed & mask,
        seed >> 32,
        base & mask,
        base >> 32,
        count & mask,
        count >> 32,
    )


def allocate_bundle(prepared, rows):
    n, k = prepared.static_spec[5:7]
    buffers = {}
    for name, shape in buffer_shapes(rows, n, k):
        dtype = torch.uint8
        if name in ("y_out", "dx_out"):
            dtype = torch.bfloat16
        elif name == "w_out":
            dtype = (torch.bfloat16, torch.float32)[prepared.static_spec[10]]
        elif name == "status":
            dtype = torch.int32
        buffers[name] = torch.zeros(shape, dtype=dtype, device=prepared.device)
    return buffers


def bind_bundle(prepared, x, buffers, rrow=None, rt=None):
    m, n = x.shape[0], prepared.static_spec[5]
    p = (m + 31) // 32 * 32
    if rrow is None:
        rrow = rng_receipt((1 << 63) + 19, (1 << 34) + 3, m * n)
    if rt is None:
        rt = rng_receipt((1 << 63) + 23, (1 << 35) + 1, p * n)
    return mxfp4_dense_bind_v1(
        prepared,
        m,
        x,
        *buffers.values(),
        (0, m, n, n, 1, 2, prepared.device.index),
        rrow,
        rt,
    )


def decode_packed_reference(payload, scales, group=1):
    """Decode in FP64 so raw254 values need not overflow before multiplication."""
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
    levels = torch.tensor(
        (0, 0.5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float64, device=payload.device
    )
    values = levels[(codes & 7).long()]
    values = torch.where(codes >= 8, -values, values)
    expanded = scales.repeat_interleave(group, dim=0).repeat_interleave(32, dim=1)
    assert expanded.shape == values.shape
    assert not torch.any(
        expanded == 255
    ).item(), "poison scales are not a finite oracle"
    return torch.ldexp(values, expanded.to(torch.int32) - 127)


def assert_numeric_metrics(actual, reference, threshold_db, *, coverage=True):
    if actual.numel() == 0:
        assert actual.shape == reference.shape
        return {
            "snr_db": math.inf,
            "coverage": 1.0,
            "relative_norm_error": 0.0,
            "max_error": 0.0,
        }
    actual, reference = actual.double(), reference.double()
    assert torch.isfinite(actual).all().item()
    assert torch.isfinite(reference).all().item()
    error = actual - reference
    signal = reference.square().sum().item()
    noise = error.square().sum().item()
    snr = (
        math.inf
        if noise == 0
        else (-math.inf if signal == 0 else 10 * math.log10(signal / noise))
    )
    fraction = (error.abs() <= 0.5 + 0.02 * reference.abs()).double().mean().item()
    relative = (
        math.sqrt(noise / signal) if signal else (0.0 if noise == 0 else math.inf)
    )
    metrics = {
        "snr_db": snr,
        "coverage": fraction,
        "relative_norm_error": relative,
        "max_error": error.abs().max().item(),
    }
    assert snr >= threshold_db, metrics
    if coverage:
        assert fraction >= 0.99, metrics
    return metrics
