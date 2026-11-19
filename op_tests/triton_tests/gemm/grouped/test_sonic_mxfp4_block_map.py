# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced AMD, Inc. All rights reserved.
"""Bit-exact A/B: legacy O(E) expert scan vs binary-search block map."""

import json

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm_triton import (
    _grouped_gemm_kernel,
)
from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
    grouped_mxfp4_weight_gemm,
    pack_grouped_mxfp4_weights,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant

N, K, E = 1536, 2048, 32
# BLOCK_K=256 only fits the activation-quantized (packed uint8) path; the BF16-A
# path needs half of that to stay inside the 160 KiB LDS limit.
CFG_A4 = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 8,
    "cache_modifier": ".cg",
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 32,
}
CFG_A16 = dict(CFG_A4, BLOCK_SIZE_K=128)


def launch(x, payload, scales, counts, quantize, *, use_map):
    CFG = CFG_A4 if quantize else CFG_A16
    BM, BN, BK = CFG["BLOCK_SIZE_M"], CFG["BLOCK_SIZE_N"], CFG["BLOCK_SIZE_K"]
    rows = sum(counts)
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=x.device)
    offsets = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(counts), 0).tolist()),
        dtype=torch.int32,
        device=x.device,
    )
    if quantize:
        kernel_x, a_scales = dynamic_mxfp4_quant(x)
        a_scales = a_scales.contiguous()
    else:
        kernel_x, a_scales = x, None
    grid = sum(triton.cdiv(c, BM) * triton.cdiv(N, BN) for c in counts)
    kwargs = {}
    if use_map:
        blocks = ((offsets[1:] - offsets[:-1]) + BM - 1) // BM * triton.cdiv(N, BN)
        base = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=x.device),
                torch.cumsum(blocks, 0, dtype=torch.int32),
            )
        )
        kwargs = {
            "expert_block_base_ptr": base,
            "BLOCK_MAP_ITERS": max(1, (E - 1).bit_length()),
        }
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
        N=N,
        K=K,
        E=E,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_SIZE_M=CFG["GROUP_SIZE_M"],
        HAS_BIAS=False,
        HAS_GATHER_IDX=False,
        HAS_SCATTER_IDX=False,
        B_scales_ptr=scales,
        W4_WEIGHT=True,
        CACHE_MODIFIER=CFG["cache_modifier"],
        A_scales_ptr=a_scales,
        A4_ACTIVATION=quantize,
        num_warps=CFG["num_warps"],
        num_stages=CFG["num_stages"],
        waves_per_eu=CFG["waves_per_eu"],
        matrix_instr_nonkdim=CFG["matrix_instr_nonkdim"],
        **kwargs,
    )
    torch.cuda.synchronize()
    return out


def main():
    torch.manual_seed(4321)
    w = torch.randn((E, N, K), dtype=torch.bfloat16, device="cuda") * 0.02
    payload, scales = pack_grouped_mxfp4_weights(w)
    cases = {
        "uniform_128": [128] * E,
        "all_empty": [0] * E,
        "ragged_with_zeros": [0, 3, 0, 129, 1, 0, 256, 33] + [0] * (E - 8),
        "tail_33": [33] * E,
        "skewed": [1024] + [1] * (E - 1),
    }
    report = {}
    for name, counts in cases.items():
        rows = sum(counts)
        x = (
            torch.randn((rows, K), dtype=torch.bfloat16, device="cuda")
            if rows
            else torch.empty((0, K), dtype=torch.bfloat16, device="cuda")
        )
        # The MXFP4 activation quantizer rejects zero-row input, and the public
        # wrapper returns before quantizing in that case, so skip only the
        # quantized variant of the empty-expert case.
        for quantize in ((False,) if rows == 0 else (False, True)):
            legacy = launch(x, payload, scales, counts, quantize, use_map=False)
            mapped = launch(x, payload, scales, counts, quantize, use_map=True)
            same = torch.equal(legacy, mapped)
            wrapped = (
                grouped_mxfp4_weight_gemm(
                    x, payload, scales, tuple(counts), quantize_activation=quantize
                )
                if rows
                else mapped
            )
            same_wrapper = torch.equal(legacy, wrapped) if rows else True
            key = f"{name}_{'a4w4' if quantize else 'a16w4'}"
            report[key] = {
                "rows": rows,
                "legacy_equals_blockmap": bool(same),
                "legacy_equals_public_wrapper": bool(same_wrapper),
            }
            print(json.dumps({key: report[key]}), flush=True)
    print(json.dumps(report))
    assert all(
        v["legacy_equals_blockmap"] and v["legacy_equals_public_wrapper"]
        for v in report.values()
    ), report


if __name__ == "__main__":
    main()
