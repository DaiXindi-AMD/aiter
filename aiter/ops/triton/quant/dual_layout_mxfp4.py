# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton

from aiter.ops.triton.utils import mxfp4_contract as contract


@contract.register_contract_op(
    ("x", "row_out", "row_scales_out", "t_out", "t_scales_out", "status"),
    ("row_out", "row_scales_out", "t_out", "t_scales_out", "status"),
    {"H": 6, "R_row": 7, "R_t": 7, "B": 4},
)
def dual_layout_quant_mxfp4_out_v1(
    x,
    row_out,
    row_scales_out,
    t_out,
    t_scales_out,
    status,
    M,
    C,
    P,
    source_dtype,
    transform,
    row_rounding,
    t_rounding,
    H,
    R_row,
    R_t,
    B,
):
    """Write independent row and pad32/H16T quantization from the same source.

    All buffers belong to the caller. The two SR tuples use separate ranges.
    """
    contract.header(H, 2, phase=contract.Phase.LAUNCH)
    source = contract.primitive_source(x, M, C, source_dtype)
    contract.integer(transform, 1, 1, phase=contract.Phase.LAUNCH)
    contract.integer(P, phase=contract.Phase.LAUNCH)
    if P != contract.pad32(M):
        contract.fail(contract.Phase.LAUNCH)
    row_seed, row_base = contract.rng_words(row_rounding, R_row, M * C)
    t_seed, t_base = contract.rng_words(t_rounding, R_t, C * P)
    variant = {(0, 0): 2, (1, 1): 3, (1, 0): 5, (0, 1): 6}[(row_rounding, t_rounding)]
    record = contract.ready_record(B, 2)
    contract.check_record(record, x.device, source_dtype, variant)
    row = contract.check_tensor(row_out, (M, C // 2), "torch.uint8", x.device)
    rs = contract.check_tensor(row_scales_out, (M, C // 32), "torch.uint8", x.device)
    transposed = contract.check_tensor(t_out, (C, P // 2), "torch.uint8", x.device)
    ts = contract.check_tensor(t_scales_out, (C, P // 32), "torch.uint8", x.device)
    state = contract.status_metadata(status, x.device)
    contract.disjoint((row, rs, transposed, ts, state), (source,))
    if M == 0:
        return
    record.executable[(triton.cdiv(M, 32), C // 32, 1)](
        x,
        row_out,
        row_scales_out,
        t_out,
        t_scales_out,
        status,
        M,
        C,
        P,
        row_seed,
        row_base,
        t_seed,
        t_base,
    )
    return
