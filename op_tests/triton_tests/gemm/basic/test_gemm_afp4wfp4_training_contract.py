# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import builtins
import importlib
import os
import re

import pytest
import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4_out_v1,
    mxfp4_dense_attach_backward_v1,
    mxfp4_dense_launch_backward_v1,
    mxfp4_dense_launch_forward_v1,
    mxfp4_dense_launch_packed_forward_v1,
    mxfp4_dense_prepare_kernel_v1,
    mxfp4_zero_output_out_v1,
)
from aiter.ops.triton.quant.fast_transpose import transpose_packed_fp4_out_v1
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant_blockscale_out_v1
from aiter.ops.triton.utils.config_utils import resolve_config_dir
from aiter.ops.triton.utils.mxfp4_contract import Mxfp4ContractError
from op_tests.triton_tests.utils.mxfp4_contract_test_utils import (
    ROWS,
    allocate_bundle,
    assert_numeric_metrics,
    bind_bundle,
    decode_packed_reference,
    rng_receipt,
)
from op_tests.triton_tests.utils.mxfp4_contract_test_utils import (
    prepared_linear as prepared_linear_fixture,
)

prepared_linear = prepared_linear_fixture


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


@pytest.mark.parametrize("linear_k", [2048, 4096])
@pytest.mark.parametrize("m", [0, 1, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 257])
@pytest.mark.parametrize("pass_id", [0, 1, 2])
def test_fixed_packed_and_fp32_preseed(prepared, linear_k, m, pass_id):
    n = 4096
    p = (m + 31) // 32 * 32
    mg, ng, kg = ((m, n, linear_k), (m, linear_k, n), (n, linear_k, p))[pass_id]
    group = 1 if pass_id == 2 else 32
    dtype, mode = (1, 1) if pass_id == 2 else (0, 0)
    record = prepared.records[
        (4, pass_id, 0, 0, dtype, group, n, linear_k, mode, "any")
    ]
    a = torch.full((mg, kg // 2), 0x11, dtype=torch.uint8, device="cuda")
    b = torch.full((ng, kg // 2), 0x11, dtype=torch.uint8, device=a.device)
    sa = torch.full((mg, kg // 32), 125, dtype=torch.uint8, device=a.device)
    sb = torch.full((ng // group, kg // 32), 125, dtype=torch.uint8, device=a.device)
    out = torch.full(
        (mg, ng),
        0.0001 if pass_id == 2 else float("nan"),
        dtype=torch.float32 if dtype else torch.bfloat16,
        device=a.device,
    )
    original = out.clone()
    workspace = torch.empty((0,), dtype=torch.uint8, device=a.device)
    status = torch.zeros((1,), dtype=torch.int32, device=a.device)
    for _ in range(2 if pass_id == 2 else 1):
        gemm_afp4wfp4_out_v1(
            a,
            b,
            sa,
            sb,
            out,
            workspace,
            status,
            mg,
            ng,
            kg,
            pass_id,
            group,
            mode,
            dtype,
            (1, 0, 4, 1, 2, 0),
            record.config,
        )
    if pass_id == 2 and m == 0:
        assert torch.equal(out, original)
    elif pass_id == 2:
        expected = (original + kg / 64) + kg / 64
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, torch.full_like(out, kg / 64), rtol=0, atol=0)
    assert status.item() == 0


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("dtype_id", [0, 1])
def test_zero_positive_bits_with_canary(prepared, k, dtype_id):
    dtype = (torch.bfloat16, torch.float32)[dtype_id]
    storage = torch.full((4096 * k + 2,), float("nan"), dtype=dtype, device="cuda")
    out = storage[1:-1].view(4096, k)
    z = prepared.records[(5, -1, 4, 0, dtype_id, 1, 0, 0, 0, "any")].config
    mxfp4_zero_output_out_v1(out, 4096, k, dtype_id, (1, 0, 5, 0, 0, 0), z)
    assert (
        torch.count_nonzero(
            out.view(torch.int16 if dtype_id == 0 else torch.int32)
        ).item()
        == 0
    )
    assert torch.isnan(storage[0]).item() and torch.isnan(storage[-1]).item()


def test_weight_once_and_nibble_transpose(prepared):
    w = torch.arange(64 * 64, dtype=torch.float32, device="cuda").reshape(64, 64) / 4096
    q = torch.empty((64, 32), dtype=torch.uint8, device=w.device)
    scales = torch.empty((2, 2), dtype=torch.uint8, device=w.device)
    status = torch.zeros((1,), dtype=torch.int32, device=w.device)
    b = prepared.records[(1, -1, 0, 1, 0, 1, 0, 0, 0, "any")].config
    dynamic_mxfp4_quant_blockscale_out_v1(
        w, q, scales, status, 64, 64, 1, (1, 0, 1, 1, 1, 0), b
    )
    original = q.clone()
    w.fill_(float("nan"))
    qt = torch.empty_like(q)
    u = prepared.records[(3, -1, 0, 0, 0, 1, 0, 0, 0, "any")].config
    transpose_packed_fp4_out_v1(q, qt, 64, 64, (1, 0, 3, 0, 1, 0), u)
    codes = torch.stack((original & 15, original >> 4), dim=-1).reshape(64, 64)
    expected = codes.T[:, 0::2] | (codes.T[:, 1::2] << 4)
    assert torch.equal(qt, expected)
    assert status.item() == 0


@pytest.mark.parametrize("m", [0, 1, 33])
def test_public_bundle_and_packed_replay(prepared, m):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
        mxfp4_dense_attach_backward_v1,
        mxfp4_dense_bind_v1,
        mxfp4_dense_launch_backward_v1,
        mxfp4_dense_launch_forward_v1,
        mxfp4_dense_launch_packed_forward_v1,
    )
    from aiter.ops.triton.utils.mxfp4_contract import buffer_shapes

    n, k = 4096, 2048
    p = (m + 31) // 32 * 32
    x = torch.ones((m, k), dtype=torch.bfloat16, device="cuda")
    values = {}
    for name, shape in buffer_shapes(m, n, k):
        dtype = torch.int32 if name == "status" else torch.uint8
        if name in ("y_out", "dx_out"):
            dtype = torch.bfloat16
        elif name == "w_out":
            dtype = torch.float32
        values[name] = torch.zeros(shape, dtype=dtype, device=x.device)
    values["w_out"].fill_(0.0001)
    rrow = (1, 19, 1, 3, 0, m * n, 0)
    rt = (1, 23, 1, 3, 0, p * n, 0)
    bound = mxfp4_dense_bind_v1(
        prepared, m, x, *values.values(), (0, m, n, n, 1, 2, x.device.index), rrow, rt
    )
    before = values["w_out"].clone()
    mxfp4_dense_launch_forward_v1(bound, x)
    original_y = values["y_out"].clone()
    original_t = values["x_t_out"].clone()
    mxfp4_dense_launch_packed_forward_v1(bound)
    assert torch.equal(values["y_out"], original_y)
    assert torch.equal(values["x_t_out"], original_t)
    dy = torch.ones((m, n), dtype=x.dtype, device=x.device)
    runtime = mxfp4_dense_attach_backward_v1(bound, dy)
    mxfp4_dense_launch_backward_v1(bound, runtime)
    assert values["status"].item() == 0
    assert torch.count_nonzero(values["y_out"]).item() == 0
    assert torch.count_nonzero(values["dx_out"]).item() == 0
    if m in (0, 1):
        torch.testing.assert_close(values["w_out"], before + m, rtol=0, atol=0)
    else:
        assert torch.all(torch.isfinite(values["w_out"])).item()


def test_mutation_schema_compile_zero(prepared):
    out = torch.empty((4096, 2048), dtype=torch.float32, device="cuda")
    z = prepared.records[(5, -1, 4, 0, 1, 1, 0, 0, 0, "any")].config

    def zero(target):
        mxfp4_zero_output_out_v1(target, 4096, 2048, 1, (1, 0, 5, 0, 0, 0), z)
        return target

    compiled = torch.compile(zero, fullgraph=True)
    assert compiled(out).data_ptr() == out.data_ptr()
    assert torch.count_nonzero(out.view(torch.int32)).item() == 0


@pytest.mark.parametrize("m", ROWS)
def test_random_fixed_packed_and_bf16_recipe(prepared_linear, m, record_property):
    """Check each pass against its same-packed oracle and unquantized BF16 source."""
    prepared = prepared_linear
    n, k = prepared.static_spec[5:7]
    generator = torch.Generator(device=prepared.device).manual_seed(173 + k + m)
    x = (
        torch.randn((m, k), generator=generator, device=prepared.device) * 0.1
    ).bfloat16()
    w = (
        torch.randn((n, k), generator=generator, device=prepared.device) * 0.02
    ).bfloat16()
    dy = (
        torch.randn((m, n), generator=generator, device=prepared.device) * 0.05
    ).bfloat16()
    buffers = allocate_bundle(prepared, m)
    buffers["w_out"].fill_(0.0001)
    preseed = buffers["w_out"].clone()
    weight_cfg = prepared.records[(1, -1, 0, 0, 0, 1, 0, 0, 0, "any")].config
    transpose_cfg = prepared.records[(3, -1, 0, 0, 0, 1, 0, 0, 0, "any")].config
    build_status = torch.zeros((1,), dtype=torch.int32, device=prepared.device)
    dynamic_mxfp4_quant_blockscale_out_v1(
        w,
        buffers["w_fp4"],
        buffers["w_scales"],
        build_status,
        n,
        k,
        0,
        (1, 0, 1, 1, 1, 0),
        weight_cfg,
    )
    transpose_packed_fp4_out_v1(
        buffers["w_fp4"],
        buffers["wt_fp4"],
        n,
        k,
        (1, 0, 3, 0, 1, 0),
        transpose_cfg,
    )
    buffers["wt_scales"] = buffers["w_scales"].T
    assert build_status.item() == 0
    bound = bind_bundle(prepared, x, buffers)
    mxfp4_dense_launch_forward_v1(bound, x)
    runtime = mxfp4_dense_attach_backward_v1(bound, dy)
    mxfp4_dense_launch_backward_v1(bound, runtime)
    assert buffers["status"].item() == 0
    if m == 0:
        assert torch.equal(buffers["w_out"], preseed)
        assert buffers["y_out"].numel() == buffers["dx_out"].numel() == 0
        return
    packed_w = decode_packed_reference(buffers["w_fp4"], buffers["w_scales"], 32)
    packed_wt = decode_packed_reference(buffers["wt_fp4"], buffers["wt_scales"], 32)
    assert torch.equal(packed_wt, packed_w.T)
    packed_x = decode_packed_reference(
        buffers["x_row_out"], buffers["x_row_scales_out"]
    )
    packed_dy = decode_packed_reference(
        buffers["dy_row_out"], buffers["dy_row_scales_out"]
    )
    packed_tx = decode_packed_reference(buffers["x_t_out"], buffers["x_t_scales_out"])
    packed_tdy = decode_packed_reference(
        buffers["dy_t_out"], buffers["dy_t_scales_out"]
    )
    packed_refs = (
        packed_x @ packed_w.T,
        packed_dy @ packed_wt.T,
        packed_tdy @ packed_tx.T,
    )
    recipe_refs = (
        x.float() @ w.float().T,
        dy.float() @ w.float(),
        dy.float().T @ x.float(),
    )
    actuals = (buffers["y_out"], buffers["dx_out"], buffers["w_out"] - preseed)
    for pass_name, actual, packed_ref, recipe_ref, threshold in zip(
        ("F", "D", "W"), actuals, packed_refs, recipe_refs, (12, 12, 10)
    ):
        packed_metrics = assert_numeric_metrics(actual, packed_ref, 30)
        recipe_metrics = assert_numeric_metrics(actual, recipe_ref, threshold)
        record_property(pass_name + "_fixed_packed", str(packed_metrics))
        record_property(pass_name + "_bf16_recipe", str(recipe_metrics))


@pytest.mark.parametrize("pass_id", [0, 1, 2])
@pytest.mark.parametrize("raw_a,raw_b", [(0, 254), (254, 0)])
def test_complementary_scale_endpoints(prepared_linear, pass_id, raw_a, raw_b):
    """Require native scale-dot agreement when tiny and large scales cancel."""
    n, k = prepared_linear.static_spec[5:7]
    mg, ng, kg = ((1, n, k), (1, k, n), (n, k, 32))[pass_id]
    group = 1 if pass_id == 2 else 32
    dtype = 1 if pass_id == 2 else 0
    cfg = prepared_linear.records[
        (4, pass_id, 0, 0, dtype, group, n, k, 0, "any")
    ].config
    a = torch.full(
        (mg, kg // 2), 0x22, dtype=torch.uint8, device=prepared_linear.device
    )
    b = torch.full((ng, kg // 2), 0x22, dtype=torch.uint8, device=a.device)
    sa = torch.full((mg, kg // 32), raw_a, dtype=torch.uint8, device=a.device)
    sb = torch.full((ng // group, kg // 32), raw_b, dtype=torch.uint8, device=a.device)
    out = torch.empty(
        (mg, ng), dtype=torch.float32 if dtype else torch.bfloat16, device=a.device
    )
    scratch = torch.empty((0,), dtype=torch.uint8, device=a.device)
    status = torch.zeros((1,), dtype=torch.int32, device=a.device)
    gemm_afp4wfp4_out_v1(
        a,
        b,
        sa,
        sb,
        out,
        scratch,
        status,
        mg,
        ng,
        kg,
        pass_id,
        group,
        0,
        dtype,
        (1, 0, 4, 1, 2, 0),
        cfg,
    )
    reference = decode_packed_reference(a, sa) @ decode_packed_reference(b, sb, group).T
    assert_numeric_metrics(out, reference, 30)
    assert status.item() == 0
    if group == 32:
        expanded = sb.repeat_interleave(32, dim=0)
        expanded_cfg = prepared_linear.records[
            (4, pass_id, 0, 0, dtype, 1, n, k, 0, "any")
        ].config
        expanded_out = torch.empty_like(out)
        gemm_afp4wfp4_out_v1(
            a,
            b,
            sa,
            expanded,
            expanded_out,
            scratch,
            status,
            mg,
            ng,
            kg,
            pass_id,
            1,
            0,
            dtype,
            (1, 0, 4, 1, 2, 0),
            expanded_cfg,
        )
        assert torch.equal(out, expanded_out)


@pytest.mark.parametrize("failure", ["sticky", "scale255", "preseed", "overflow"])
def test_numeric_status_does_not_fallback(prepared, failure):
    n, k, p = 4096, 2048, 32
    a = torch.full((n, p // 2), 0x22, dtype=torch.uint8, device=prepared.device)
    b = torch.full((k, p // 2), 0x22, dtype=torch.uint8, device=prepared.device)
    sa = torch.full((n, 1), 127, dtype=torch.uint8, device=prepared.device)
    sb = torch.full((k, 1), 127, dtype=torch.uint8, device=prepared.device)
    out = torch.full((n, k), 0.0001, dtype=torch.float32, device=prepared.device)
    status = torch.zeros((1,), dtype=torch.int32, device=prepared.device)
    scratch = torch.empty((0,), dtype=torch.uint8, device=prepared.device)
    if failure == "sticky":
        status.fill_(1)
    elif failure == "scale255":
        sa.fill_(255)
    elif failure == "preseed":
        out.fill_(float("nan"))
    else:
        sa.fill_(254)
        sb.fill_(254)
    old = out.clone()
    cfg = prepared.records[(4, 2, 0, 0, 1, 1, n, k, 1, "any")].config
    gemm_afp4wfp4_out_v1(
        a, b, sa, sb, out, scratch, status, n, k, p, 2, 1, 1, 1, (1, 0, 4, 1, 2, 0), cfg
    )
    if failure == "sticky":
        assert status.item() == 1
        assert torch.equal(out, old)
    else:
        assert status.item() & 4


@pytest.mark.parametrize(
    "failure", ["capacity", "alias", "workspace", "rng", "dtype", "stride", "device"]
)
def test_public_bind_rejects_without_mutation(prepared, failure):
    x = torch.ones((1, 2048), dtype=torch.bfloat16, device=prepared.device)
    buffers = allocate_bundle(prepared, 1)
    rrow = rng_receipt(1, 0, 4096)
    expected_reason = None
    if failure == "capacity":
        x = torch.ones((258, 2048), dtype=x.dtype, device=x.device)
        expected_reason = 103
    elif failure == "alias":
        buffers["dx_out"] = x
        expected_reason = 105
    elif failure == "workspace":
        buffers["workspace"] = torch.empty((1,), dtype=torch.uint8, device=x.device)
    elif failure == "rng":
        rrow = rng_receipt(1, (1 << 64) - 1, 4096)
        expected_reason = 114
    elif failure == "dtype":
        x = x.float()
    elif failure == "stride":
        x = torch.ones((1, 4096), dtype=x.dtype, device=x.device)[:, ::2]
    else:
        buffers["status"] = torch.zeros((1,), dtype=torch.int32, device="cpu")
    before = {name: tensor.clone() for name, tensor in buffers.items()}
    with pytest.raises(Mxfp4ContractError) as caught:
        bind_bundle(prepared, x, buffers, rrow=rrow)
    assert caught.value.disposition == 1
    if expected_reason is not None:
        assert caught.value.reason == expected_reason
    for name, tensor in buffers.items():
        assert torch.equal(tensor, before[name]), name


@pytest.mark.parametrize("failure", ["dtype", "shape", "alias", "stride"])
def test_attach_is_fatal_without_rebinding(prepared, failure):
    x = torch.ones((1, 2048), dtype=torch.bfloat16, device=prepared.device)
    buffers = allocate_bundle(prepared, 1)
    bound = bind_bundle(prepared, x, buffers)
    dy = torch.ones((1, 4096), dtype=x.dtype, device=x.device)
    if failure == "dtype":
        dy = dy.float()
    elif failure == "shape":
        dy = dy[:, :-1]
    elif failure == "alias":
        dy = buffers["y_out"]
    else:
        dy = torch.ones((1, 8192), dtype=x.dtype, device=x.device)[:, ::2]
    with pytest.raises(Mxfp4ContractError) as caught:
        mxfp4_dense_attach_backward_v1(bound, dy)
    assert (caught.value.phase, caught.value.reason, caught.value.disposition) == (
        3,
        109,
        1,
    )
    assert bound.buffers["w_out"] is buffers["w_out"]


def test_new_routes_no_allocation_jit_files_and_no_ck(prepared, monkeypatch):
    """Instrument first-seen P32/64/96 calls after finite-slot preparation."""
    cases = []
    for m in (1, 33, 65):
        x = torch.ones((m, 2048), dtype=torch.bfloat16, device=prepared.device)
        dy = torch.ones((m, 4096), dtype=x.dtype, device=x.device)
        cases.append((x, dy, allocate_bundle(prepared, m)))

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation, JIT, or file access after prepare")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        with monkeypatch.context() as patch:
            for name in (
                "empty",
                "empty_like",
                "zeros",
                "zeros_like",
                "ones",
                "ones_like",
            ):
                patch.setattr(torch, name, forbidden)
            patch.setattr(torch.Tensor, "resize_", forbidden)
            patch.setattr(triton, "compile", forbidden)
            patch.setattr(triton.runtime.jit.JITFunction, "run", forbidden)
            patch.setattr(builtins, "open", forbidden)
            for x, dy, buffers in cases:
                bound = bind_bundle(prepared, x, buffers)
                runtime = mxfp4_dense_attach_backward_v1(bound, dy)
                mxfp4_dense_launch_forward_v1(bound, x)
                mxfp4_dense_launch_packed_forward_v1(bound)
                mxfp4_dense_launch_backward_v1(bound, runtime)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.cuda.memory_reserved() == reserved
    names = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert any("_dual_layout_quant_mxfp4_kernel" in name for name in names), names
    assert any("_gemm_afp4wfp4_kernel" in name for name in names), names
    assert not any(
        re.search(r"(?:^|[ :/])ck[_:]|composable|gluon", name.lower()) for name in names
    ), names
    for _, _, buffers in cases:
        assert buffers["status"].item() == 0


def test_zero_specialization_has_no_global_read(prepared):
    """Check emitted zero ISA, not just the mathematical masked-load expression."""
    for dtype in (0, 1):
        record = prepared.records[(5, -1, 4, 0, dtype, 1, 0, 0, 0, "any")]
        assembly = record.executable.asm["amdgcn"]
        assert not re.search(r"\b(?:global|flat|buffer)_load\w*", assembly), assembly


def test_empty_bound_never_launches_gemm(prepared, monkeypatch):
    x = torch.empty((0, 2048), dtype=torch.bfloat16, device=prepared.device)
    dy = torch.empty((0, 4096), dtype=torch.bfloat16, device=prepared.device)
    buffers = allocate_bundle(prepared, 0)
    buffers["w_out"].fill_(-0.0)
    before = buffers["w_out"].view(torch.int32).clone()
    bound = bind_bundle(prepared, x, buffers)
    runtime = mxfp4_dense_attach_backward_v1(bound, dy)

    def forbidden(*args, **kwargs):
        raise AssertionError("empty accumulate must launch neither GEMM nor zero")

    module = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_afp4wfp4")
    monkeypatch.setattr(module, "_launch_bound_gemm", forbidden)
    monkeypatch.setattr(module, "mxfp4_zero_output_out_v1", forbidden)
    mxfp4_dense_launch_forward_v1(bound, x)
    mxfp4_dense_launch_packed_forward_v1(bound)
    mxfp4_dense_launch_backward_v1(bound, runtime)
    assert torch.equal(buffers["w_out"].view(torch.int32), before)
