import pytest
import torch

from aiter.ops.triton.activation import fused_silu_mul, swiglu_bwd, swiglu_fwd

LOG2_E = 1.44269504089

# GLM-4.7-FP8 MoE (e.g. zai-org/GLM-4.7-FP8): moe_intermediate_size=1536, top_k=8.
# Column-parallel TP4: local d = 1536 // 4 = 384, fused silu-mul input last dim = 768.
_GLM47_TP4_LAST = 768
_GLM47_TOP_K = 8

# Kimi-K2.5 MoE (moonshotai/Kimi-K2.5 text_config): moe_intermediate_size=2048, top_k=8.
# TP4: local d = 2048 // 4 = 512, last dim = 1024.
_KIMI_K25_TP4_LAST = 1024
_KIMI_K25_TOP_K = 8


def silu_exp2_ref(t: torch.Tensor) -> torch.Tensor:
    """Match ``_silu_exp2`` in Triton (same as MoE silu-fused path)."""
    x = t.float()
    return x / (1.0 + torch.exp2(-(x * LOG2_E)))


def torch_silu_mul_last_dim_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1) // 2
    a, b = x[..., :d], x[..., d:]
    return (silu_exp2_ref(a) * b).to(x.dtype)


def torch_swiglu_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1) // 2
    gate, up = x[..., :d].float(), x[..., d:].float()
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


def torch_swiglu_backward_ref(
    grad_output: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    d = x.size(-1) // 2
    gate, up = x[..., :d].float(), x[..., d:].float()
    grad = grad_output.float()
    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    dsilu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    return torch.cat((grad * up * dsilu_gate, grad * silu_gate), dim=-1).to(
        x.dtype
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 64),
        (128, 256),
        (31, 500),
        (2, 16, 128),
        (1, 3, 7, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_explicit_out", [False, True])
def test_fused_silu_mul(shape, dtype, use_explicit_out):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = torch_silu_mul_last_dim_ref(x)
    if use_explicit_out:
        out = torch.empty_like(ref)
        fused_silu_mul(x, out)
    else:
        out = fused_silu_mul(x)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_fused_silu_mul_requires_even_last_dim():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(2, 3, device="cuda")
    with pytest.raises(AssertionError, match="even"):
        fused_silu_mul(x)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 64),
        (128, 256),
        (31, 500),
        (2, 16, 128),
        (1, 3, 7, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous_grad", [False, True])
def test_standard_swiglu_forward_backward(shape, dtype, noncontiguous_grad):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    grad_shape = (*shape[:-1], shape[-1] // 2)
    if noncontiguous_grad:
        grad_storage = torch.randn(
            *shape[:-1], shape[-1], dtype=dtype, device="cuda"
        )
        grad_output = grad_storage[..., ::2]
        assert grad_output.shape == grad_shape
        assert not grad_output.is_contiguous()
    else:
        grad_output = torch.randn(grad_shape, dtype=dtype, device="cuda")

    out_ref = torch_swiglu_ref(x)
    grad_ref = torch_swiglu_backward_ref(grad_output, x)
    out = swiglu_fwd(x)
    grad = swiglu_bwd(grad_output, x)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grad, grad_ref, rtol=1e-2, atol=1e-2)


def test_silu_mul_apis_reject_empty_last_dim():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.empty((2, 0), device="cuda")
    grad_output = torch.empty_like(x)
    with pytest.raises(AssertionError, match="non-zero"):
        fused_silu_mul(x)
    with pytest.raises(AssertionError, match="non-zero"):
        swiglu_fwd(x)
    with pytest.raises(AssertionError, match="non-zero"):
        swiglu_bwd(grad_output, x)


@pytest.mark.parametrize(
    "n_rows,last_dim",
    [
        # Decode M=4 → rows M * top_k
        pytest.param(4 * _GLM47_TOP_K, _GLM47_TP4_LAST, id="glm47_tp4_decode4"),
        pytest.param(
            4 * _KIMI_K25_TOP_K, _KIMI_K25_TP4_LAST, id="kimi_k25_tp4_decode4"
        ),
        # Medium prefill / batched decode
        pytest.param(256 * _GLM47_TOP_K, _GLM47_TP4_LAST, id="glm47_tp4_rows256x8"),
        pytest.param(
            256 * _KIMI_K25_TOP_K, _KIMI_K25_TP4_LAST, id="kimi_k25_tp4_rows256x8"
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_silu_mul_tp4_moe_shapes(n_rows, last_dim, dtype):
    """MoE fused silu×mul tensor as (tokens * top_k, 2 * local_d) under TP4."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    shape = (n_rows, last_dim)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = torch_silu_mul_last_dim_ref(x)
    out = fused_silu_mul(x)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "n_rows,last_dim",
    [
        pytest.param(
            (8190 + 3) * _GLM47_TOP_K,
            _GLM47_TP4_LAST,
            id="glm47_tp4_pref8190_dec3",
        ),
        pytest.param(
            (7235 + 3) * _GLM47_TOP_K,
            _GLM47_TP4_LAST,
            id="glm47_tp4_pref7235_dec3",
        ),
        pytest.param(
            (8190 + 3) * _KIMI_K25_TOP_K,
            _KIMI_K25_TP4_LAST,
            id="kimi_k25_tp4_pref8190_dec3",
        ),
        pytest.param(
            (7235 + 3) * _KIMI_K25_TOP_K,
            _KIMI_K25_TP4_LAST,
            id="kimi_k25_tp4_pref7235_dec3",
        ),
    ],
)
def test_fused_silu_mul_tp4_prefill_bf16(n_rows, last_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    shape = (n_rows, last_dim)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = torch_silu_mul_last_dim_ref(x)
    out = fused_silu_mul(x)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
