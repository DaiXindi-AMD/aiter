from .quant.quant import _mxfp4_quant_op
from .quant.fused_fp8_quant import _fp8_quant_op
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.activation import _silu_exp2
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _silu(x):
    return _silu_exp2(x)


_fused_silu_mul_kernel_repr = make_kernel_repr(
    "fused_silu_mul_kernel",
    ["BLOCK_M", "BLOCK_N", "EAGER_ROUNDING"],
)


@triton.jit(repr=_fused_silu_mul_kernel_repr)
def fused_silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride_gate,
    col_stride_gate,
    row_stride_up,
    col_stride_up,
    row_stride_out,
    col_stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EAGER_ROUNDING: tl.constexpr,
):
    """
    Apply SiLU to ``gate`` and multiply by ``up`` for matching 2D tiles.
    Each input row has ``n_cols`` elements and the output has the same shape.
    2D grid: axis 0 tiles rows (BLOCK_M), axis 1 tiles columns (BLOCK_N).
    """
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    row_idx = m_pid * BLOCK_M + m_offs
    col_idx = n_pid * BLOCK_N + n_offs

    row_idx_i64 = row_idx.to(tl.int64)
    col_idx_i64 = col_idx.to(tl.int64)
    row_stride_gate_i64 = tl.cast(row_stride_gate, tl.int64)
    col_stride_gate_i64 = tl.cast(col_stride_gate, tl.int64)
    row_stride_up_i64 = tl.cast(row_stride_up, tl.int64)
    col_stride_up_i64 = tl.cast(col_stride_up, tl.int64)
    row_stride_out_i64 = tl.cast(row_stride_out, tl.int64)
    col_stride_out_i64 = tl.cast(col_stride_out, tl.int64)

    gate_ptrs = (
        gate_ptr
        + row_idx_i64[:, None] * row_stride_gate_i64
        + col_idx_i64[None, :] * col_stride_gate_i64
    )
    up_ptrs = (
        up_ptr
        + row_idx_i64[:, None] * row_stride_up_i64
        + col_idx_i64[None, :] * col_stride_up_i64
    )
    out_ptrs = (
        out_ptr
        + row_idx_i64[:, None] * row_stride_out_i64
        + col_idx_i64[None, :] * col_stride_out_i64
    )

    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]
    a = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    if EAGER_ROUNDING:
        silu_a = _silu_exp2(a).to(gate_ptr.dtype.element_ty)
    else:
        silu_a = a * tl.sigmoid(a)
    b = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    o = (silu_a * b).to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, o, mask=mask)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + tl.erf(x * ALPHA))


@triton.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def _relu(x):
    return tl.maximum(0.0, x)


@triton.jit
def _relu6(x):
    return tl.minimum(tl.maximum(0.0, x), 6.0)


def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
        "relu6": _relu6,
    }
    return mapping[activation]


@triton.jit
def _apply_activation_from_str(x, activation: tl.constexpr):
    if activation == "gelu":
        return _gelu(x)
    elif activation == "gelu_tanh":
        return _gelu_tanh(x)
    elif activation == "silu":
        return _silu(x)
    elif activation == "silu_exp2":
        return _silu_exp2(x)
    elif activation == "relu":
        return _relu(x)
    elif activation == "relu6":
        return _relu6(x)
    else:
        return x  # No activation if it is not recognized


@triton.heuristics(
    {
        "EVEN_M_N": lambda args: (
            args["M"] % args["BLOCK_SIZE_M"] == 0
            and args["N"] % (args["BLOCK_SIZE_N"] * args["NUM_ITER"]) == 0
        ),
    }
)
@triton.jit
def _act_mul_and_dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    scaleN: tl.constexpr,
    scaleM_pad: tl.constexpr,
    scaleN_pad: tl.constexpr,
    SHUFFLE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in tl.range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            a = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
            b = tl.load(x_ptr + x_offs + stride_x_n * N, cache_modifier=".cg").to(
                tl.float32
            )
        else:
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            a = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
                tl.float32
            )
            # a and b can share the same mask
            b = tl.load(
                x_ptr + x_offs + stride_x_n * N, mask=x_mask, cache_modifier=".cg"
            ).to(tl.float32)

        x = _apply_activation_from_str(a, ACTIVATION) * b

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        out_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        if SHUFFLE:
            bs_offs_0 = bs_offs_m[:, None] // 32
            bs_offs_1 = bs_offs_m[:, None] % 32
            bs_offs_2 = bs_offs_1 % 16
            bs_offs_1 = bs_offs_1 // 16
            bs_offs_3 = bs_offs_n[None, :] // 8
            bs_offs_4 = bs_offs_n[None, :] % 8
            bs_offs_5 = bs_offs_4 % 4
            bs_offs_4 = bs_offs_4 // 4
            bs_offs = (
                bs_offs_1
                + bs_offs_4 * 2
                + bs_offs_2 * 2 * 2
                + bs_offs_5 * 2 * 2 * 16
                + bs_offs_3 * 2 * 2 * 16 * 4
                + bs_offs_0 * 2 * 16 * scaleN
            )
            bs_mask1 = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
            bs_mask = (bs_offs_m < scaleM_pad)[:, None] & (bs_offs_n < scaleN_pad)[
                None, :
            ]
            bs_e8m0 = tl.where(bs_mask1, bs_e8m0, 127)
        else:
            bs_offs = (
                bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
            )
            bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            tl.store(
                bs_ptr + bs_offs,
                bs_e8m0,
                mask=bs_mask,
            )


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    }
)
@triton.jit
def _act_mul_and_dynamic_fp8_group_quant_kernel(
    x_ptr,
    x_fp8_ptr,
    x_bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp8_m_in,
    stride_x_fp8_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    N,
    ACTIVATION: tl.constexpr,
    scaleN: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # cast strides to int64, in case M*N > max int32
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp8_m = tl.cast(stride_x_fp8_m_in, tl.int64)
    stride_x_fp8_n = tl.cast(stride_x_fp8_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE

    x_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_offs = pid_m * stride_x_m + x_offs_n * stride_x_n

    if EVEN_N:
        a = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        b = tl.load(x_ptr + x_offs + stride_x_n * N, cache_modifier=".cg").to(
            tl.float32
        )
    else:
        x_mask = x_offs_n < N
        a = tl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(tl.float32)
        # a and b can share the same mask
        b = tl.load(
            x_ptr + x_offs + stride_x_n * N, mask=x_mask, cache_modifier=".cg"
        ).to(tl.float32)

    x = _apply_activation_from_str(a, ACTIVATION) * b

    x_fp8, x_bs = _fp8_quant_op(
        x, 1, BLOCK_SIZE_N, QUANT_BLOCK_SIZE, DTYPE_MAX, DTYPE_MIN
    )
    x_fp8 = tl.ravel(x_fp8)
    x_bs = tl.ravel(x_bs)

    out_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_offs = pid_m * stride_x_fp8_m + out_offs_n * stride_x_fp8_n

    if EVEN_N:
        tl.store(x_fp8_ptr + out_offs, x_fp8.to(x_fp8_ptr.dtype.element_ty))
    else:
        out_mask = out_offs_n < N
        tl.store(
            x_fp8_ptr + out_offs, x_fp8.to(x_fp8_ptr.dtype.element_ty), mask=out_mask
        )

    bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
    bs_offs = pid_m * stride_bs_m + bs_offs_n * stride_bs_n
    if EVEN_N:
        tl.store(x_bs_ptr + bs_offs, x_bs.to(x_bs_ptr.dtype.element_ty))
    else:
        bs_mask = bs_offs_n < scaleN
        tl.store(
            x_bs_ptr + bs_offs,
            x_bs.to(x_bs_ptr.dtype.element_ty),
            mask=bs_mask,
        )


# ── Fused SwiGLU forward / backward ─────────────────────────────────────

_swiglu_bwd_kernel_repr = make_kernel_repr(
    "swiglu_bwd_kernel",
    ["BLOCK_M", "BLOCK_N", "EAGER_ROUNDING"],
)


@triton.jit(repr=_swiglu_bwd_kernel_repr)
def _swiglu_bwd_kernel(
    grad_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    n_rows,
    n_cols,
    row_stride_grad,
    col_stride_grad,
    row_stride_gate,
    col_stride_gate,
    row_stride_up,
    col_stride_up,
    row_stride_dgate,
    col_stride_dgate,
    row_stride_dup,
    col_stride_dup,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EAGER_ROUNDING: tl.constexpr,
):
    """Compute SwiGLU gradients from separate gate/up tensors in 2D tiles."""
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    row_idx = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]

    row_idx_i64 = row_idx.to(tl.int64)
    col_idx_i64 = col_idx.to(tl.int64)
    row_stride_grad_i64 = tl.cast(row_stride_grad, tl.int64)
    col_stride_grad_i64 = tl.cast(col_stride_grad, tl.int64)
    row_stride_gate_i64 = tl.cast(row_stride_gate, tl.int64)
    col_stride_gate_i64 = tl.cast(col_stride_gate, tl.int64)
    row_stride_up_i64 = tl.cast(row_stride_up, tl.int64)
    col_stride_up_i64 = tl.cast(col_stride_up, tl.int64)
    row_stride_dgate_i64 = tl.cast(row_stride_dgate, tl.int64)
    col_stride_dgate_i64 = tl.cast(col_stride_dgate, tl.int64)
    row_stride_dup_i64 = tl.cast(row_stride_dup, tl.int64)
    col_stride_dup_i64 = tl.cast(col_stride_dup, tl.int64)

    grad_ptrs = (
        grad_ptr
        + row_idx_i64[:, None] * row_stride_grad_i64
        + col_idx_i64[None, :] * col_stride_grad_i64
    )
    gate_ptrs = (
        gate_ptr
        + row_idx_i64[:, None] * row_stride_gate_i64
        + col_idx_i64[None, :] * col_stride_gate_i64
    )
    up_ptrs = (
        up_ptr
        + row_idx_i64[:, None] * row_stride_up_i64
        + col_idx_i64[None, :] * col_stride_up_i64
    )
    dgate_ptrs = (
        dgate_ptr
        + row_idx_i64[:, None] * row_stride_dgate_i64
        + col_idx_i64[None, :] * col_stride_dgate_i64
    )
    dup_ptrs = (
        dup_ptr
        + row_idx_i64[:, None] * row_stride_dup_i64
        + col_idx_i64[None, :] * col_stride_dup_i64
    )

    grad = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    if EAGER_ROUNDING:
        sigmoid = 1.0 / (1.0 + tl.exp2(-(gate * 1.44269504089)))
    else:
        sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid
    dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))

    if EAGER_ROUNDING:
        silu = silu.to(gate_ptr.dtype.element_ty).to(tl.float32)
        grad_silu = (grad * up).to(grad_ptr.dtype.element_ty).to(tl.float32)
        dgate = grad_silu * dsilu
    else:
        dgate = grad * dsilu * up

    tl.store(dgate_ptrs, dgate, mask=mask)
    tl.store(dup_ptrs, grad * silu, mask=mask)
