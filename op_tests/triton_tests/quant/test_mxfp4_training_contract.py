# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import struct

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import mxfp4_dense_prepare_kernel_v1
from aiter.ops.triton.quant.dual_layout_mxfp4 import dual_layout_quant_mxfp4_out_v1
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant_out_v1
from aiter.ops.triton.utils.config_utils import resolve_config_dir
from op_tests.triton_tests.utils.mxfp4_reference import h16_reference, row_reference


@pytest.fixture(scope="module")
def prepared():
    cu = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    spec = (1, 0, 950, cu, 1, 4096, 2048, 257, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1)
    path = os.path.join(
        resolve_config_dir("gemm", "GEMM-AFP4WFP4"),
        "GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json",
    )
    return mxfp4_dense_prepare_kernel_v1(spec, path)


def _record(prepared, op, variant, dtype=1):
    return prepared.records[(op, -1, variant, dtype, 0, 1, 0, 0, 0, "any")].config


def _receipt(seed, base, count):
    return (
        1,
        seed & 0xFFFFFFFF,
        seed >> 32,
        base & 0xFFFFFFFF,
        base >> 32,
        count & 0xFFFFFFFF,
        count >> 32,
    )


@pytest.mark.parametrize(
    "bits",
    [
        0,
        0x80000000,
        0x00010000,
        0x00400000,
        0x00800000,
        0x3FDFFFFF,
        0x3FE00000,
        0x7F7FFFFF,
        0x7FC00000,
        0x7F800000,
    ],
)
@pytest.mark.parametrize("rounding", [0, 1])
def test_row_exact_bytes(prepared, bits, rounding):
    source_bits = [bits] * 32
    values = [struct.unpack("<f", struct.pack("<I", b))[0] for b in source_bits]
    x = torch.tensor([values], dtype=torch.float32, device="cuda")
    payload = torch.empty((1, 16), dtype=torch.uint8, device=x.device)
    scales = torch.empty((1, 1), dtype=torch.uint8, device=x.device)
    status = torch.zeros((1,), dtype=torch.int32, device=x.device)
    seed, base = (1 << 63) + 17, (1 << 34) + 3
    receipt = _receipt(seed, base, 32) if rounding else (0,) * 7
    dynamic_mxfp4_quant_out_v1(
        x,
        payload,
        scales,
        status,
        1,
        32,
        1,
        rounding,
        (1, 0, 0, 1, 2, 0),
        receipt,
        _record(prepared, 0, rounding),
    )
    expected, expected_scale = row_reference(
        source_bits, seed if rounding else None, base
    )
    assert payload.cpu().tolist() == [expected]
    assert scales.cpu().tolist() == [expected_scale]
    assert status.item() == (1 if ((bits >> 23) & 255) == 255 else 0)


@pytest.mark.parametrize("m", [0, 1, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 257])
def test_dual_ragged_impulse(prepared, m):
    p = (m + 31) // 32 * 32
    x = torch.zeros((m, 32), dtype=torch.float32, device="cuda")
    if m:
        x[0, 0] = 4
    row = torch.empty((m, 16), dtype=torch.uint8, device=x.device)
    rs = torch.empty((m, 1), dtype=torch.uint8, device=x.device)
    transposed = torch.empty((32, p // 2), dtype=torch.uint8, device=x.device)
    ts = torch.empty((32, p // 32), dtype=torch.uint8, device=x.device)
    status = torch.zeros((1,), dtype=torch.int32, device=x.device)
    dual_layout_quant_mxfp4_out_v1(
        x,
        row,
        rs,
        transposed,
        ts,
        status,
        m,
        32,
        p,
        1,
        1,
        0,
        0,
        (1, 0, 2, 1, 2, 0),
        (0,) * 7,
        (0,) * 7,
        _record(prepared, 2, 2),
    )
    assert status.item() == 0
    if m:
        assert transposed[0, :16].cpu().tolist() == [0x66] * 8 + [0] * 8
        assert ts[0, 0].item() == 125
        assert torch.count_nonzero(transposed[1:]).item() == 0


def test_no_decode_requant_trap(prepared):
    x = torch.zeros((2, 32), dtype=torch.float32, device="cuda")
    x[0, 0], x[1, 0], x[:, 31] = 1.25, 1.75, 6.0
    row = torch.empty((2, 16), dtype=torch.uint8, device=x.device)
    rs = torch.empty((2, 1), dtype=torch.uint8, device=x.device)
    transposed = torch.empty((32, 16), dtype=torch.uint8, device=x.device)
    ts = torch.empty((32, 1), dtype=torch.uint8, device=x.device)
    status = torch.zeros((1,), dtype=torch.int32, device=x.device)
    dual_layout_quant_mxfp4_out_v1(
        x,
        row,
        rs,
        transposed,
        ts,
        status,
        2,
        32,
        32,
        1,
        1,
        0,
        0,
        (1, 0, 2, 1, 2, 0),
        (0,) * 7,
        (0,) * 7,
        _record(prepared, 2, 2),
    )
    assert transposed[0, :8].cpu().tolist() == [0xA7] * 8
    assert ts[0, 0].item() == 124
    assert status.item() == 0


@pytest.mark.parametrize("base", [0, 1, 2, 3, (1 << 34) - 1, (1 << 63) + 7])
def test_dual_independent_element_domains(prepared, base):
    bits = [0x3FA00000] * 32 + [0x3FE00000] * 32
    values = [struct.unpack("<f", struct.pack("<I", b))[0] for b in bits]
    x = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(2, 32)
    row = torch.empty((2, 16), dtype=torch.uint8, device=x.device)
    rs = torch.empty((2, 1), dtype=torch.uint8, device=x.device)
    transposed = torch.empty((32, 16), dtype=torch.uint8, device=x.device)
    ts = torch.empty((32, 1), dtype=torch.uint8, device=x.device)
    status = torch.zeros((1,), dtype=torch.int32, device=x.device)
    seed_row, seed_t = (1 << 63) + 9, (1 << 63) + 23
    dual_layout_quant_mxfp4_out_v1(
        x,
        row,
        rs,
        transposed,
        ts,
        status,
        2,
        32,
        32,
        1,
        1,
        1,
        1,
        (1, 0, 2, 1, 2, 0),
        _receipt(seed_row, base, 64),
        _receipt(seed_t, base, 1024),
        _record(prepared, 2, 3),
    )
    expected_row, expected_rs = row_reference(bits, seed_row, base)
    hbits = h16_reference([0x3FA00000, 0x3FE00000] + [0] * 14) + [0] * 16
    expected_t, expected_ts = row_reference(hbits * 32, seed_t, base)
    assert row.cpu().flatten().tolist() == expected_row
    assert rs.cpu().flatten().tolist() == expected_rs
    assert transposed.cpu().flatten().tolist() == expected_t
    assert ts.cpu().flatten().tolist() == expected_ts
    assert status.item() == 0


@pytest.mark.parametrize(
    "value,expected_status",
    [(float("nan"), 1), (float("inf"), 1), (3.3895313892515355e38, 2)],
)
def test_dual_poison_axes_and_h16_overflow(prepared, value, expected_status):
    x = torch.ones((16, 32), dtype=torch.float32, device="cuda")
    x[:, 0] = value
    row = torch.empty((16, 16), dtype=torch.uint8, device=x.device)
    rs = torch.empty((16, 1), dtype=torch.uint8, device=x.device)
    transposed = torch.empty((32, 16), dtype=torch.uint8, device=x.device)
    ts = torch.empty((32, 1), dtype=torch.uint8, device=x.device)
    status = torch.zeros((1,), dtype=torch.int32, device=x.device)
    dual_layout_quant_mxfp4_out_v1(
        x,
        row,
        rs,
        transposed,
        ts,
        status,
        16,
        32,
        32,
        1,
        1,
        0,
        0,
        (1, 0, 2, 1, 2, 0),
        (0,) * 7,
        (0,) * 7,
        _record(prepared, 2, 2),
    )
    assert status.item() == expected_status
    assert ts[0, 0].item() == 255
    assert torch.count_nonzero(transposed[0]).item() == 0
    assert torch.all(ts[1:] != 255).item()
    if expected_status == 1:
        assert torch.all(rs == 255).item()
        assert torch.count_nonzero(row).item() == 0
    else:
        assert torch.all(rs == 254).item()
