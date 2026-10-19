# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Candidate public-API timings; run only after correctness qualification."""

import argparse
import os

import torch
import triton.testing

from aiter import logger
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4_out_v1,
    mxfp4_dense_prepare_kernel_v1,
    mxfp4_zero_output_out_v1,
)
from aiter.ops.triton.quant.dual_layout_mxfp4 import dual_layout_quant_mxfp4_out_v1
from aiter.ops.triton.quant.fast_transpose import transpose_packed_fp4_out_v1
from aiter.ops.triton.quant.quant import (
    dynamic_mxfp4_quant_blockscale_out_v1,
    dynamic_mxfp4_quant_out_v1,
)
from aiter.ops.triton.utils.config_utils import resolve_config_dir


def benchmark(rows, k):
    cu = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    spec = (1, 0, 950, cu, 1, 4096, k, max(rows, 1), 0, 0, 1, 1, 2, 1, 1, 1, 2, 0, 1)
    artifact = os.path.join(
        resolve_config_dir("gemm", "GEMM-AFP4WFP4"),
        "GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json",
    )
    prepared = mxfp4_dense_prepare_kernel_v1(spec, artifact)
    n, p = 4096, (rows + 31) // 32 * 32
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device=x.device, dtype=x.dtype)
    q = torch.empty((n, k // 2), device=x.device, dtype=torch.uint8)
    qs = torch.empty((n // 32, k // 32), device=x.device, dtype=torch.uint8)
    qt = torch.empty((k, n // 2), device=x.device, dtype=torch.uint8)
    row = torch.empty((rows, k // 2), device=x.device, dtype=torch.uint8)
    rs = torch.empty((rows, k // 32), device=x.device, dtype=torch.uint8)
    xt = torch.empty((k, p // 2), device=x.device, dtype=torch.uint8)
    xts = torch.empty((k, p // 32), device=x.device, dtype=torch.uint8)
    y = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    dw = torch.empty((n, k), device=x.device, dtype=torch.float32)
    status = torch.zeros((1,), device=x.device, dtype=torch.int32)
    workspace = torch.empty((0,), device=x.device, dtype=torch.uint8)
    records = prepared.records
    row_cfg = records[(0, -1, 0, 0, 0, 1, 0, 0, 0, "any")].config
    weight_cfg = records[(1, -1, 0, 0, 0, 1, 0, 0, 0, "any")].config
    dual_cfg = records[(2, -1, 2, 0, 0, 1, 0, 0, 0, "any")].config
    trans_cfg = records[(3, -1, 0, 0, 0, 1, 0, 0, 0, "any")].config
    gemm_cfg = records[(4, 0, 0, 0, 0, 32, n, k, 0, "any")].config
    zero_cfg = records[(5, -1, 4, 0, 1, 1, 0, 0, 0, "any")].config
    calls = {
        "row": lambda: dynamic_mxfp4_quant_out_v1(
            x, row, rs, status, rows, k, 0, 0, (1, 0, 0, 1, 2, 0), (0,) * 7, row_cfg
        ),
        "weight": lambda: dynamic_mxfp4_quant_blockscale_out_v1(
            w, q, qs, status, n, k, 0, (1, 0, 1, 1, 1, 0), weight_cfg
        ),
        "dual": lambda: dual_layout_quant_mxfp4_out_v1(
            x,
            row,
            rs,
            xt,
            xts,
            status,
            rows,
            k,
            p,
            0,
            1,
            0,
            0,
            (1, 0, 2, 1, 2, 0),
            (0,) * 7,
            (0,) * 7,
            dual_cfg,
        ),
        "transpose": lambda: transpose_packed_fp4_out_v1(
            q, qt, n, k, (1, 0, 3, 0, 1, 0), trans_cfg
        ),
        "gemm": lambda: gemm_afp4wfp4_out_v1(
            row,
            q,
            rs,
            qs,
            y,
            workspace,
            status,
            rows,
            n,
            k,
            0,
            32,
            0,
            0,
            (1, 0, 4, 1, 2, 0),
            gemm_cfg,
        ),
        "zero": lambda: mxfp4_zero_output_out_v1(
            dw, n, k, 1, (1, 0, 5, 0, 0, 0), zero_cfg
        ),
    }
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        if status.item():
            raise RuntimeError("numeric status prevents benchmark")
        ms = triton.testing.do_bench(call, warmup=20, rep=200)
        logger.info(
            "contract=%s M=%s K=%s latency_ms=%s config_artifact=%s",
            name,
            rows,
            k,
            ms,
            prepared.artifact_digest,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--k", type=int, choices=(2048, 4096), default=2048)
    args = parser.parse_args()
    benchmark(args.rows, args.k)
