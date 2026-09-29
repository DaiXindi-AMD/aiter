# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import (
    _mxfp4_rtn_pack,
    _mxfp4_scale_from_amax,
    _mxfp4_sr_pack,
    _mxfp4_sr_random_words,
)
from aiter.ops.triton.utils._triton.activation import _sigmoid_exp2, _silu_exp2
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.shuffle import (
    _mxfp4_shuffled_offsets,
    _mxfp4_swizzled_scale_offsets,
)


@triton.jit
def _store_dual_layout_mxfp4(
    values,
    row_ptr,
    row_scale_ptr,
    col_ptr,
    col_scale_ptr,
    hmat,
    stride_row_m_in,
    stride_row_n_in,
    stride_row_scale_m_in,
    stride_row_scale_n_in,
    stride_col_m_in,
    stride_col_n_in,
    stride_col_scale_m_in,
    stride_col_scale_n_in,
    M,
    N,
    philox_seed,
    philox_offset_row,
    philox_offset_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    RNG_BLOCK_M: tl.constexpr,
    RNG_BLOCK_N: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    SHUFFLE_COL: tl.constexpr,
    SCALE_STRIPE: tl.constexpr,
    SCALE_KCHUNK: tl.constexpr,
    SHUFFLE_TILE_ROWS: tl.constexpr,
    SHUFFLE_UNIT_BYTES: tl.constexpr,
):
    """Store row-wise and signed-H16-transposed MXFP4 views of one BF16 tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows_i64 = rows.to(tl.int64)
    cols_i64 = cols.to(tl.int64)
    source_tile_m = (pid_m * BLOCK_M) // RNG_BLOCK_M
    source_tile_n = (pid_n * BLOCK_N) // RNG_BLOCK_N
    source_tile_id = source_tile_m * (N // RNG_BLOCK_N) + source_tile_n

    stride_row_m = tl.cast(stride_row_m_in, tl.int64)
    stride_row_n = tl.cast(stride_row_n_in, tl.int64)
    stride_row_scale_m = tl.cast(stride_row_scale_m_in, tl.int64)
    stride_row_scale_n = tl.cast(stride_row_scale_n_in, tl.int64)
    stride_col_m = tl.cast(stride_col_m_in, tl.int64)
    stride_col_n = tl.cast(stride_col_n_in, tl.int64)
    stride_col_scale_m = tl.cast(stride_col_scale_m_in, tl.int64)
    stride_col_scale_n = tl.cast(stride_col_scale_n_in, tl.int64)

    ROW_SCALE_COLS: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    row_grouped = values.to(tl.float32).reshape(
        BLOCK_M, ROW_SCALE_COLS, QUANT_BLOCK_SIZE
    )
    row_amax = tl.max(tl.abs(row_grouped), axis=2, keep_dims=True)
    row_scales, _ = _mxfp4_scale_from_amax(row_amax)
    row_scales = row_scales.reshape(BLOCK_M, ROW_SCALE_COLS)
    if USE_SR:
        row_random_words = _mxfp4_sr_random_words(
            philox_seed,
            philox_offset_row,
            source_tile_id,
            pid_m * BLOCK_M,
            pid_n * BLOCK_N,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            RNG_BLOCK_M=RNG_BLOCK_M,
            RNG_BLOCK_N=RNG_BLOCK_N,
            PHILOX_ROUNDS=PHILOX_ROUNDS,
        )
        row_packed = _mxfp4_sr_pack(
            values.to(tl.float32),
            row_scales,
            row_random_words,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            MXFP4_QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        )
    else:
        row_packed = _mxfp4_rtn_pack(
            values,
            row_scales,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            MXFP4_QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        )

    row_packed_cols = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    row_mask = (rows < M)[:, None] & (row_packed_cols < N // 2)[None, :]
    tl.store(
        row_ptr
        + rows_i64[:, None] * stride_row_m
        + row_packed_cols.to(tl.int64)[None, :] * stride_row_n,
        row_packed,
        mask=row_mask,
    )

    row_scale_cols = pid_n * ROW_SCALE_COLS + tl.arange(0, ROW_SCALE_COLS)
    row_scale_mask = (rows < M)[:, None] & (row_scale_cols < N // QUANT_BLOCK_SIZE)[
        None, :
    ]
    if SWIZZLE_SCALE:
        row_scale_offsets = _mxfp4_swizzled_scale_offsets(
            rows_i64[:, None],
            row_scale_cols.to(tl.int64)[None, :],
            N // QUANT_BLOCK_SIZE,
            STRIPE=SCALE_STRIPE,
            KCHUNK=SCALE_KCHUNK,
        )
    else:
        row_scale_offsets = (
            rows_i64[:, None] * stride_row_scale_m
            + row_scale_cols.to(tl.int64)[None, :] * stride_row_scale_n
        )
    tl.store(row_scale_ptr + row_scale_offsets, row_scales, mask=row_scale_mask)

    HADAMARD_ROWS: tl.constexpr = BLOCK_N * (BLOCK_M // HADAMARD_SIZE)
    col_values = tl.dot(
        tl.trans(values).reshape(HADAMARD_ROWS, HADAMARD_SIZE),
        hmat,
        out_dtype=tl.float32,
    ).reshape(BLOCK_N, BLOCK_M)

    COL_SCALE_COLS: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    col_grouped = col_values.reshape(BLOCK_N, COL_SCALE_COLS, QUANT_BLOCK_SIZE)
    col_amax = tl.max(tl.abs(col_grouped), axis=2, keep_dims=True)
    col_scales, _ = _mxfp4_scale_from_amax(col_amax)
    col_scales = col_scales.reshape(BLOCK_N, COL_SCALE_COLS)
    if USE_SR:
        col_random_words = _mxfp4_sr_random_words(
            philox_seed,
            philox_offset_col,
            source_tile_id,
            pid_n * BLOCK_N,
            pid_m * BLOCK_M,
            BLOCK_SIZE_M=BLOCK_N,
            BLOCK_SIZE_N=BLOCK_M,
            RNG_BLOCK_M=RNG_BLOCK_N,
            RNG_BLOCK_N=RNG_BLOCK_M,
            PHILOX_ROUNDS=PHILOX_ROUNDS,
        )
        col_packed = _mxfp4_sr_pack(
            col_values,
            col_scales,
            col_random_words,
            BLOCK_SIZE_M=BLOCK_N,
            BLOCK_SIZE_N=BLOCK_M,
            MXFP4_QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        )
    else:
        col_packed = _mxfp4_rtn_pack(
            col_values,
            col_scales,
            BLOCK_SIZE_M=BLOCK_N,
            BLOCK_SIZE_N=BLOCK_M,
            MXFP4_QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        )

    col_packed_cols = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
    col_packed_mask = (cols < N)[:, None] & (col_packed_cols < M // 2)[None, :]
    if SHUFFLE_COL:
        col_offsets = _mxfp4_shuffled_offsets(
            cols_i64[:, None],
            col_packed_cols.to(tl.int64)[None, :],
            M // 2,
            TILE_ROWS=SHUFFLE_TILE_ROWS,
            UNIT_BYTES=SHUFFLE_UNIT_BYTES,
        )
    else:
        col_offsets = (
            cols_i64[:, None] * stride_col_m
            + col_packed_cols.to(tl.int64)[None, :] * stride_col_n
        )
    tl.store(col_ptr + col_offsets, col_packed, mask=col_packed_mask)

    col_scale_cols = pid_m * COL_SCALE_COLS + tl.arange(0, COL_SCALE_COLS)
    col_scale_mask = (cols < N)[:, None] & (col_scale_cols < M // QUANT_BLOCK_SIZE)[
        None, :
    ]
    if SWIZZLE_SCALE:
        col_scale_offsets = _mxfp4_swizzled_scale_offsets(
            cols_i64[:, None],
            col_scale_cols.to(tl.int64)[None, :],
            M // QUANT_BLOCK_SIZE,
            STRIPE=SCALE_STRIPE,
            KCHUNK=SCALE_KCHUNK,
        )
    else:
        col_scale_offsets = (
            cols_i64[:, None] * stride_col_scale_m
            + col_scale_cols.to(tl.int64)[None, :] * stride_col_scale_n
        )
    tl.store(col_scale_ptr + col_scale_offsets, col_scales, mask=col_scale_mask)


_dual_layout_quant_mxfp4_repr = make_kernel_repr(
    "dual_layout_quant_mxfp4_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "QUANT_BLOCK_SIZE",
        "HADAMARD_SIZE",
        "USE_SR",
        "PHILOX_ROUNDS",
        "RNG_BLOCK_M",
        "RNG_BLOCK_N",
        "SWIZZLE_SCALE",
        "SHUFFLE_COL",
    ],
)


@triton.jit(repr=_dual_layout_quant_mxfp4_repr)
def _dual_layout_quant_mxfp4_kernel(
    x_ptr,
    row_ptr,
    row_scale_ptr,
    col_ptr,
    col_scale_ptr,
    hmat_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_row_m_in,
    stride_row_n_in,
    stride_row_scale_m_in,
    stride_row_scale_n_in,
    stride_col_m_in,
    stride_col_n_in,
    stride_col_scale_m_in,
    stride_col_scale_n_in,
    M,
    N,
    philox_seed,
    philox_offset_row,
    philox_offset_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    RNG_BLOCK_M: tl.constexpr,
    RNG_BLOCK_N: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    SHUFFLE_COL: tl.constexpr,
    SCALE_STRIPE: tl.constexpr,
    SCALE_KCHUNK: tl.constexpr,
    SHUFFLE_TILE_ROWS: tl.constexpr,
    SHUFFLE_UNIT_BYTES: tl.constexpr,
):
    """Load one BF16 tile and emit its two MXFP4 training layouts."""
    tl.static_assert(x_ptr.type.element_ty == tl.bfloat16)
    tl.static_assert(QUANT_BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)
    tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows_i64 = rows.to(tl.int64)
    cols_i64 = cols.to(tl.int64)
    value_mask = (rows < M)[:, None] & (cols < N)[None, :]
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    values = tl.load(
        x_ptr + rows_i64[:, None] * stride_x_m + cols_i64[None, :] * stride_x_n,
        mask=value_mask,
        other=0.0,
        cache_modifier=".cg",
    )
    h_row = tl.arange(0, HADAMARD_SIZE)
    h_col = tl.arange(0, HADAMARD_SIZE)
    hmat = tl.load(hmat_ptr + h_row[:, None] * HADAMARD_SIZE + h_col[None, :])

    _store_dual_layout_mxfp4(
        values,
        row_ptr,
        row_scale_ptr,
        col_ptr,
        col_scale_ptr,
        hmat,
        stride_row_m_in,
        stride_row_n_in,
        stride_row_scale_m_in,
        stride_row_scale_n_in,
        stride_col_m_in,
        stride_col_n_in,
        stride_col_scale_m_in,
        stride_col_scale_n_in,
        M,
        N,
        philox_seed,
        philox_offset_row,
        philox_offset_col,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        HADAMARD_SIZE=HADAMARD_SIZE,
        USE_SR=USE_SR,
        PHILOX_ROUNDS=PHILOX_ROUNDS,
        RNG_BLOCK_M=RNG_BLOCK_M,
        RNG_BLOCK_N=RNG_BLOCK_N,
        SWIZZLE_SCALE=SWIZZLE_SCALE,
        SHUFFLE_COL=SHUFFLE_COL,
        SCALE_STRIPE=SCALE_STRIPE,
        SCALE_KCHUNK=SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=SHUFFLE_UNIT_BYTES,
    )


_fused_swiglu_dual_layout_mxfp4_repr = make_kernel_repr(
    "fused_swiglu_dual_layout_mxfp4_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "QUANT_BLOCK_SIZE",
        "HADAMARD_SIZE",
        "SWIZZLE_SCALE",
        "SHUFFLE_COL",
    ],
)


@triton.jit(repr=_fused_swiglu_dual_layout_mxfp4_repr)
def _fused_swiglu_dual_layout_mxfp4_kernel(
    gate_ptr,
    up_ptr,
    activation_ptr,
    row_ptr,
    row_scale_ptr,
    col_ptr,
    col_scale_ptr,
    hmat_ptr,
    stride_gate_m_in,
    stride_gate_n_in,
    stride_up_m_in,
    stride_up_n_in,
    stride_activation_m_in,
    stride_activation_n_in,
    stride_row_m_in,
    stride_row_n_in,
    stride_row_scale_m_in,
    stride_row_scale_n_in,
    stride_col_m_in,
    stride_col_n_in,
    stride_col_scale_m_in,
    stride_col_scale_n_in,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    SHUFFLE_COL: tl.constexpr,
    SCALE_STRIPE: tl.constexpr,
    SCALE_KCHUNK: tl.constexpr,
    SHUFFLE_TILE_ROWS: tl.constexpr,
    SHUFFLE_UNIT_BYTES: tl.constexpr,
):
    """Emit BF16 SwiGLU plus row and H16-transposed RTN MXFP4 layouts."""
    tl.static_assert(QUANT_BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)
    tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows_i64 = rows.to(tl.int64)
    cols_i64 = cols.to(tl.int64)
    value_mask = (rows < M)[:, None] & (cols < N)[None, :]

    stride_gate_m = tl.cast(stride_gate_m_in, tl.int64)
    stride_gate_n = tl.cast(stride_gate_n_in, tl.int64)
    stride_up_m = tl.cast(stride_up_m_in, tl.int64)
    stride_up_n = tl.cast(stride_up_n_in, tl.int64)
    stride_activation_m = tl.cast(stride_activation_m_in, tl.int64)
    stride_activation_n = tl.cast(stride_activation_n_in, tl.int64)
    gate = tl.load(
        gate_ptr
        + rows_i64[:, None] * stride_gate_m
        + cols_i64[None, :] * stride_gate_n,
        mask=value_mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    up = tl.load(
        up_ptr + rows_i64[:, None] * stride_up_m + cols_i64[None, :] * stride_up_n,
        mask=value_mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    silu_bf16 = _silu_exp2(gate).to(tl.bfloat16)
    activation_bf16 = (silu_bf16.to(tl.float32) * up).to(tl.bfloat16)
    tl.store(
        activation_ptr
        + rows_i64[:, None] * stride_activation_m
        + cols_i64[None, :] * stride_activation_n,
        activation_bf16,
        mask=value_mask,
    )

    h_row = tl.arange(0, HADAMARD_SIZE)
    h_col = tl.arange(0, HADAMARD_SIZE)
    hmat = tl.load(hmat_ptr + h_row[:, None] * HADAMARD_SIZE + h_col[None, :])
    _store_dual_layout_mxfp4(
        activation_bf16,
        row_ptr,
        row_scale_ptr,
        col_ptr,
        col_scale_ptr,
        hmat,
        stride_row_m_in,
        stride_row_n_in,
        stride_row_scale_m_in,
        stride_row_scale_n_in,
        stride_col_m_in,
        stride_col_n_in,
        stride_col_scale_m_in,
        stride_col_scale_n_in,
        M,
        N,
        0,
        0,
        0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        HADAMARD_SIZE=HADAMARD_SIZE,
        USE_SR=False,
        PHILOX_ROUNDS=7,
        RNG_BLOCK_M=BLOCK_M,
        RNG_BLOCK_N=BLOCK_N,
        SWIZZLE_SCALE=SWIZZLE_SCALE,
        SHUFFLE_COL=SHUFFLE_COL,
        SCALE_STRIPE=SCALE_STRIPE,
        SCALE_KCHUNK=SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=SHUFFLE_UNIT_BYTES,
    )


_fused_swiglu_bwd_dual_layout_mxfp4_repr = make_kernel_repr(
    "fused_swiglu_bwd_dual_layout_mxfp4_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "QUANT_BLOCK_SIZE",
        "HADAMARD_SIZE",
        "USE_SR",
        "PHILOX_ROUNDS",
        "RNG_BLOCK_M",
        "RNG_BLOCK_N",
        "SWIZZLE_SCALE",
    ],
)


@triton.jit(repr=_fused_swiglu_bwd_dual_layout_mxfp4_repr)
def _fused_swiglu_bwd_dual_layout_mxfp4_kernel(
    grad_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    dgate_row_ptr,
    dgate_row_scale_ptr,
    dgate_col_ptr,
    dgate_col_scale_ptr,
    dup_row_ptr,
    dup_row_scale_ptr,
    dup_col_ptr,
    dup_col_scale_ptr,
    hmat_ptr,
    stride_grad_m_in,
    stride_grad_n_in,
    stride_gate_m_in,
    stride_gate_n_in,
    stride_up_m_in,
    stride_up_n_in,
    stride_dgate_m_in,
    stride_dgate_n_in,
    stride_dup_m_in,
    stride_dup_n_in,
    stride_dgate_row_m_in,
    stride_dgate_row_n_in,
    stride_dgate_row_scale_m_in,
    stride_dgate_row_scale_n_in,
    stride_dgate_col_m_in,
    stride_dgate_col_n_in,
    stride_dgate_col_scale_m_in,
    stride_dgate_col_scale_n_in,
    stride_dup_row_m_in,
    stride_dup_row_n_in,
    stride_dup_row_scale_m_in,
    stride_dup_row_scale_n_in,
    stride_dup_col_m_in,
    stride_dup_col_n_in,
    stride_dup_col_scale_m_in,
    stride_dup_col_scale_n_in,
    M,
    N,
    gate_philox_seed,
    gate_philox_offset_row,
    gate_philox_offset_col,
    up_philox_seed,
    up_philox_offset_row,
    up_philox_offset_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    USE_SR: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    RNG_BLOCK_M: tl.constexpr,
    RNG_BLOCK_N: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    SCALE_STRIPE: tl.constexpr,
    SCALE_KCHUNK: tl.constexpr,
    SHUFFLE_TILE_ROWS: tl.constexpr,
    SHUFFLE_UNIT_BYTES: tl.constexpr,
):
    """Emit eager-compatible BF16 SwiGLU gradients and both MXFP4 layouts."""
    tl.static_assert(QUANT_BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)
    tl.static_assert(BLOCK_M % QUANT_BLOCK_SIZE == 0)
    tl.static_assert(BLOCK_N % QUANT_BLOCK_SIZE == 0)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows_i64 = rows.to(tl.int64)
    cols_i64 = cols.to(tl.int64)
    value_mask = (rows < M)[:, None] & (cols < N)[None, :]

    stride_grad_m = tl.cast(stride_grad_m_in, tl.int64)
    stride_grad_n = tl.cast(stride_grad_n_in, tl.int64)
    stride_gate_m = tl.cast(stride_gate_m_in, tl.int64)
    stride_gate_n = tl.cast(stride_gate_n_in, tl.int64)
    stride_up_m = tl.cast(stride_up_m_in, tl.int64)
    stride_up_n = tl.cast(stride_up_n_in, tl.int64)
    stride_dgate_m = tl.cast(stride_dgate_m_in, tl.int64)
    stride_dgate_n = tl.cast(stride_dgate_n_in, tl.int64)
    stride_dup_m = tl.cast(stride_dup_m_in, tl.int64)
    stride_dup_n = tl.cast(stride_dup_n_in, tl.int64)
    grad = tl.load(
        grad_ptr
        + rows_i64[:, None] * stride_grad_m
        + cols_i64[None, :] * stride_grad_n,
        mask=value_mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr
        + rows_i64[:, None] * stride_gate_m
        + cols_i64[None, :] * stride_gate_n,
        mask=value_mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    up = tl.load(
        up_ptr + rows_i64[:, None] * stride_up_m + cols_i64[None, :] * stride_up_n,
        mask=value_mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    sigmoid = _sigmoid_exp2(gate)
    silu_bf16 = (gate * sigmoid).to(tl.bfloat16)
    grad_up_bf16 = (grad * up).to(tl.bfloat16)
    dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    dgate_bf16 = (grad_up_bf16.to(tl.float32) * dsilu).to(tl.bfloat16)
    dup_bf16 = (grad * silu_bf16.to(tl.float32)).to(tl.bfloat16)
    tl.store(
        dgate_ptr
        + rows_i64[:, None] * stride_dgate_m
        + cols_i64[None, :] * stride_dgate_n,
        dgate_bf16,
        mask=value_mask,
    )
    tl.store(
        dup_ptr + rows_i64[:, None] * stride_dup_m + cols_i64[None, :] * stride_dup_n,
        dup_bf16,
        mask=value_mask,
    )

    h_row = tl.arange(0, HADAMARD_SIZE)
    h_col = tl.arange(0, HADAMARD_SIZE)
    hmat = tl.load(hmat_ptr + h_row[:, None] * HADAMARD_SIZE + h_col[None, :])
    _store_dual_layout_mxfp4(
        dgate_bf16,
        dgate_row_ptr,
        dgate_row_scale_ptr,
        dgate_col_ptr,
        dgate_col_scale_ptr,
        hmat,
        stride_dgate_row_m_in,
        stride_dgate_row_n_in,
        stride_dgate_row_scale_m_in,
        stride_dgate_row_scale_n_in,
        stride_dgate_col_m_in,
        stride_dgate_col_n_in,
        stride_dgate_col_scale_m_in,
        stride_dgate_col_scale_n_in,
        M,
        N,
        gate_philox_seed,
        gate_philox_offset_row,
        gate_philox_offset_col,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        HADAMARD_SIZE=HADAMARD_SIZE,
        USE_SR=USE_SR,
        PHILOX_ROUNDS=PHILOX_ROUNDS,
        RNG_BLOCK_M=RNG_BLOCK_M,
        RNG_BLOCK_N=RNG_BLOCK_N,
        SWIZZLE_SCALE=SWIZZLE_SCALE,
        SHUFFLE_COL=False,
        SCALE_STRIPE=SCALE_STRIPE,
        SCALE_KCHUNK=SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=SHUFFLE_UNIT_BYTES,
    )
    _store_dual_layout_mxfp4(
        dup_bf16,
        dup_row_ptr,
        dup_row_scale_ptr,
        dup_col_ptr,
        dup_col_scale_ptr,
        hmat,
        stride_dup_row_m_in,
        stride_dup_row_n_in,
        stride_dup_row_scale_m_in,
        stride_dup_row_scale_n_in,
        stride_dup_col_m_in,
        stride_dup_col_n_in,
        stride_dup_col_scale_m_in,
        stride_dup_col_scale_n_in,
        M,
        N,
        up_philox_seed,
        up_philox_offset_row,
        up_philox_offset_col,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        HADAMARD_SIZE=HADAMARD_SIZE,
        USE_SR=USE_SR,
        PHILOX_ROUNDS=PHILOX_ROUNDS,
        RNG_BLOCK_M=RNG_BLOCK_M,
        RNG_BLOCK_N=RNG_BLOCK_N,
        SWIZZLE_SCALE=SWIZZLE_SCALE,
        SHUFFLE_COL=False,
        SCALE_STRIPE=SCALE_STRIPE,
        SCALE_KCHUNK=SCALE_KCHUNK,
        SHUFFLE_TILE_ROWS=SHUFFLE_TILE_ROWS,
        SHUFFLE_UNIT_BYTES=SHUFFLE_UNIT_BYTES,
    )
