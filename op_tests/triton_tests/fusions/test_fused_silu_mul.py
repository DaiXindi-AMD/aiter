import torch
import torch.nn.functional as F
import pytest

from aiter.ops.triton.activation import (
    fused_silu_mul,
    fused_silu_mul_backward,
    fused_silu_mul_two_input,
    swiglu_bwd,
    swiglu_bwd_split,
    swiglu_fwd,
    swiglu_fwd_split,
)

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


def torch_silu_mul_backward_ref(
    grad: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_f32 = gate.float()
    grad_f32 = grad.float()
    up_f32 = up.float()
    sigmoid = torch.sigmoid(gate_f32)
    silu = gate_f32 * sigmoid
    dsilu = sigmoid * (1.0 + gate_f32 * (1.0 - sigmoid))
    return (grad_f32 * dsilu * up_f32).to(gate.dtype), (grad_f32 * silu).to(gate.dtype)


def eager_swiglu_autograd_ref(
    grad: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    out_ref = F.silu(gate_ref) * up_ref
    out_ref.backward(grad)
    assert gate_ref.grad is not None and up_ref.grad is not None
    return out_ref.detach(), gate_ref.grad, up_ref.grad


def compute_snr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    signal = reference.float().square().sum()
    noise = (reference.float() - actual.float()).square().sum()
    if noise == 0:
        return float("inf")
    return float(10.0 * torch.log10(signal / noise))


def assert_matches_eager(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Allow only the sub-ULP difference from Triton's exp2 SiLU approximation."""
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-3)


def _ordered_bf16_bits(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where(
        (bits & 0x8000) != 0,
        0x8000 - magnitude,
        0x8000 + magnitude,
    )


def assert_silu_backward_matches_eager(
    actual: torch.Tensor, expected: torch.Tensor
) -> None:
    """Allow rare one-ULP BF16 ties from different SiLU evaluation orders."""
    if actual.dtype != torch.bfloat16:
        assert_matches_eager(actual, expected)
        return

    assert actual.shape == expected.shape
    assert expected.dtype == torch.bfloat16
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    if actual.numel() == 0:
        return

    ulp = (_ordered_bf16_bits(actual) - _ordered_bf16_bits(expected)).abs()
    changed = int((ulp != 0).sum())
    max_ulp = int(ulp.max())
    allowed_changed = max(1, (actual.numel() + 999_999) // 1_000_000)
    snr = compute_snr(expected, actual)
    assert max_ulp <= 1, f"BF16 SiLU dgate differs by {max_ulp} ULP"
    assert changed <= allowed_changed, (
        f"BF16 SiLU dgate has {changed}/{actual.numel()} changed elements; "
        f"allowed {allowed_changed}"
    )
    assert snr >= 40.0, f"BF16 SiLU dgate SNR is only {snr:.3f} dB"


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
    x = torch.randn(2, 3, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="even"):
        fused_silu_mul(x)


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
    dtype = torch.bfloat16
    shape = (n_rows, last_dim)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = torch_silu_mul_last_dim_ref(x)
    out = fused_silu_mul(x)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        (31, 250),
        (2, 17, 384),
        pytest.param((257, 12288), id="qwen3_wide_non_aligned_rows"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_silu_mul_two_input_forward_backward(shape, dtype):
    """The split API matches eager forward and autograd cut points exactly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(shape, dtype=dtype, device="cuda")
    up = torch.randn_like(gate)
    grad = torch.randn_like(gate)

    expected, expected_dgate, expected_dup = eager_swiglu_autograd_ref(grad, gate, up)
    actual = swiglu_fwd_split(gate, up)
    dgate, dup = swiglu_bwd_split(grad, gate, up)

    assert_matches_eager(actual, expected)
    assert_silu_backward_matches_eager(dgate, expected_dgate)
    assert_matches_eager(dup, expected_dup)

    assert_matches_eager(fused_silu_mul_two_input(gate, up), expected)
    alias_dgate, alias_dup = fused_silu_mul_backward(grad, gate, up)
    assert_silu_backward_matches_eager(alias_dgate, expected_dgate)
    assert_matches_eager(alias_dup, expected_dup)


def test_split_backward_preserves_bf16_eager_rounding_points():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(20260920)
    gate = torch.randn(31, 250, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(gate)
    grad = torch.randn_like(gate)

    _, expected_dgate, expected_dup = eager_swiglu_autograd_ref(grad, gate, up)
    full_dgate, full_dup = torch_silu_mul_backward_ref(grad, gate, up)
    actual_dgate, actual_dup = swiglu_bwd_split(grad, gate, up)

    assert not torch.equal(expected_dgate, full_dgate)
    assert not torch.equal(expected_dup, full_dup)
    torch.testing.assert_close(actual_dgate, expected_dgate, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_dup, expected_dup, rtol=0.0, atol=0.0)


def test_split_backward_allows_bf16_rounding_boundary():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.tensor([[0.0023040771484375]], dtype=torch.bfloat16, device="cuda")
    up = torch.tensor([[1.9375]], dtype=torch.bfloat16, device="cuda")
    grad = torch.tensor([[-0.4375]], dtype=torch.bfloat16, device="cuda")
    _, expected_dgate, _ = eager_swiglu_autograd_ref(grad, gate, up)
    actual_dgate, _ = swiglu_bwd_split(grad, gate, up)

    assert not torch.equal(actual_dgate, expected_dgate)
    ulp = (_ordered_bf16_bits(actual_dgate) - _ordered_bf16_bits(expected_dgate)).abs()
    assert int(ulp.max()) == 1
    assert_silu_backward_matches_eager(actual_dgate, expected_dgate)


@pytest.mark.parametrize("shape", [(4, 128), (3, 7, 500)])
def test_packed_swiglu_compatibility_uses_shared_tiled_kernels(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    packed = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    grad = torch.randn(*shape[:-1], shape[-1] // 2, dtype=packed.dtype, device="cuda")

    d = shape[-1] // 2
    gate_f32 = packed[..., :d].float()
    up_f32 = packed[..., d:].float()
    sigmoid = torch.sigmoid(gate_f32)
    silu = gate_f32 * sigmoid
    expected = (silu * up_f32).to(packed.dtype)
    torch.testing.assert_close(swiglu_fwd(packed), expected, rtol=0.0, atol=0.0)

    dpacked = swiglu_bwd(grad, packed)
    grad_f32 = grad.float()
    dsilu = sigmoid * (1.0 + gate_f32 * (1.0 - sigmoid))
    expected_dgate = (grad_f32 * dsilu * up_f32).to(packed.dtype)
    expected_dup = (grad_f32 * silu).to(packed.dtype)
    torch.testing.assert_close(dpacked[..., :d], expected_dgate, rtol=0.0, atol=0.0)
    torch.testing.assert_close(dpacked[..., d:], expected_dup, rtol=0.0, atol=0.0)


def test_packed_swiglu_accepts_noncontiguous_input_and_broadcast_grad():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    packed = torch.randn(2, 3, 128, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    grad = torch.randn(1, 1, 64, dtype=packed.dtype, device=packed.device).expand(
        3, 2, 64
    )
    assert not packed.is_contiguous()
    assert 0 in grad.stride()

    gate_f32 = packed[..., :64].float()
    up_f32 = packed[..., 64:].float()
    sigmoid = torch.sigmoid(gate_f32)
    silu = gate_f32 * sigmoid
    expected = (silu * up_f32).to(packed.dtype)
    grad_f32 = grad.float()
    dsilu = sigmoid * (1.0 + gate_f32 * (1.0 - sigmoid))
    expected_dgate = (grad_f32 * dsilu * up_f32).to(packed.dtype)
    expected_dup = (grad_f32 * silu).to(packed.dtype)

    torch.testing.assert_close(swiglu_fwd(packed), expected, rtol=0.0, atol=0.0)
    dpacked = swiglu_bwd(grad, packed)
    torch.testing.assert_close(dpacked[..., :64], expected_dgate, rtol=0.0, atol=0.0)
    torch.testing.assert_close(dpacked[..., 64:], expected_dup, rtol=0.0, atol=0.0)


def test_fused_silu_mul_two_input_requires_matching_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(2, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="shape"):
        swiglu_fwd_split(gate, up)


@pytest.mark.parametrize("shape", [(64,), (0,), (2, 0), (0, 64), (2, 0, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_swiglu_supports_1d_and_empty_tensors(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(shape, dtype=dtype, device="cuda")
    up = torch.randn_like(gate)
    grad = torch.randn_like(gate)

    out = swiglu_fwd_split(gate, up)
    dgate, dup = swiglu_bwd_split(grad, gate, up)

    assert out.shape == gate.shape
    assert dgate.shape == gate.shape
    assert dup.shape == up.shape
    if gate.numel():
        expected, expected_dgate, expected_dup = eager_swiglu_autograd_ref(
            grad, gate, up
        )
        assert_matches_eager(out, expected)
        assert_silu_backward_matches_eager(dgate, expected_dgate)
        assert_matches_eager(dup, expected_dup)


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 64), (2, 0, 64)])
def test_packed_swiglu_supports_empty_tensors(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    packed = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    grad = torch.empty(
        *shape[:-1], shape[-1] // 2, dtype=packed.dtype, device=packed.device
    )

    assert swiglu_fwd(packed).shape == grad.shape
    assert swiglu_bwd(grad, packed).shape == packed.shape
    assert fused_silu_mul(packed).shape == grad.shape


def test_empty_packed_forward_still_validates_preallocated_output():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    packed = torch.empty(2, 0, dtype=torch.bfloat16, device="cuda")
    wrong_out = torch.empty(1, 0, dtype=packed.dtype, device=packed.device)
    with pytest.raises(ValueError, match="out shape"):
        fused_silu_mul(packed, wrong_out)


def _padded_rows(rows, cols, padding, dtype=torch.bfloat16):
    storage = torch.empty(rows, cols + padding, dtype=dtype, device="cuda")
    view = storage[:, :cols]
    view.copy_(torch.randn_like(view))
    return view


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_swiglu_supports_padded_rows_and_preallocated_outputs(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = _padded_rows(17, 250, 11, dtype)
    up = _padded_rows(17, 250, 7, dtype)
    grad = _padded_rows(17, 250, 5, dtype)
    out = _padded_rows(17, 250, 13, dtype)
    dgate = _padded_rows(17, 250, 17, dtype)
    dup = _padded_rows(17, 250, 19, dtype)
    expected, expected_dgate, expected_dup = eager_swiglu_autograd_ref(grad, gate, up)

    returned_out = swiglu_fwd_split(gate, up, out)
    returned_dgate, returned_dup = swiglu_bwd_split(grad, gate, up, dgate, dup)

    assert returned_out is out
    assert returned_dgate is dgate
    assert returned_dup is dup
    assert_matches_eager(out, expected)
    assert_silu_backward_matches_eager(dgate, expected_dgate)
    assert_matches_eager(dup, expected_dup)


def test_split_swiglu_rejects_nonflattenable_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(2, 3, 64, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    up = torch.randn_like(gate)
    with pytest.raises(ValueError, match="flatten"):
        swiglu_fwd_split(gate, up)


def test_split_swiglu_rejects_unsupported_dtype_and_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(2, 64, dtype=torch.float32, device="cuda")
    with pytest.raises(TypeError, match="float16 or bfloat16"):
        swiglu_fwd_split(gate, torch.randn_like(gate))

    gate_cpu = torch.randn(2, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        swiglu_fwd_split(gate_cpu, torch.randn_like(gate_cpu))


def test_split_swiglu_rejects_true_output_aliases():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(gate)
    grad = torch.randn_like(gate)

    with pytest.raises(ValueError, match="overlap gate"):
        swiglu_fwd_split(gate, up, gate)
    with pytest.raises(ValueError, match="overlap gate"):
        swiglu_bwd_split(grad, gate, up, gate, torch.empty_like(up))

    shared_output = torch.empty_like(gate)
    with pytest.raises(ValueError, match="overlap grad_gate"):
        swiglu_bwd_split(grad, gate, up, shared_output, shared_output)


def test_split_swiglu_rejects_internal_overlap_in_preallocated_outputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(gate)
    grad = torch.randn_like(gate)
    expanded_out = torch.empty(1, 64, dtype=gate.dtype, device=gate.device).expand_as(
        gate
    )

    with pytest.raises(ValueError, match="out must not have internal overlap"):
        swiglu_fwd_split(gate, up, expanded_out)
    with pytest.raises(ValueError, match="grad_gate must not have internal overlap"):
        swiglu_bwd_split(grad, gate, up, expanded_out, torch.empty_like(up))


def test_split_swiglu_allows_disjoint_views_of_shared_storage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    shared_input_output = torch.randn(2, 192, dtype=torch.bfloat16, device="cuda")
    gate = shared_input_output[:, :64]
    out = shared_input_output[:, 64:128]
    up = torch.randn_like(gate)
    grad = torch.randn_like(gate)
    expected, expected_dgate, expected_dup = eager_swiglu_autograd_ref(grad, gate, up)

    returned_out = swiglu_fwd_split(gate, up, out)
    shared_grad_output = torch.empty(2, 128, dtype=gate.dtype, device=gate.device)
    dgate = shared_grad_output[:, :64]
    dup = shared_grad_output[:, 64:]
    returned_dgate, returned_dup = swiglu_bwd_split(grad, gate, up, dgate, dup)

    assert returned_out is out
    assert returned_dgate is dgate
    assert returned_dup is dup
    assert_matches_eager(out, expected)
    assert_silu_backward_matches_eager(dgate, expected_dgate)
    assert_matches_eager(dup, expected_dup)


@pytest.mark.parametrize(
    "bad_grad_shape",
    [(2, 31), (1, 32), (2, 33)],
)
def test_packed_swiglu_backward_rejects_grad_shape_mismatch(bad_grad_shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    packed = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
    grad = torch.randn(bad_grad_shape, dtype=packed.dtype, device=packed.device)
    with pytest.raises(ValueError, match="grad_output must have shape"):
        swiglu_bwd(grad, packed)
