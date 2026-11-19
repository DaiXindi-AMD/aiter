# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pre-routed SwiGLU experts with quantized forward and selectable gradients.

Independent axis-wise MX quantization defines a surrogate-gradient recipe.
Router derivatives use the saved forward output. Native MXFP4 Wgrad is opt-in;
this module makes no training-quality claim from historical cached-weight runs.
"""

import os

import torch

from aiter.ops.triton._triton_kernels.moe.sonicmoe.row_scale import row_scale_fp32
from aiter.ops.triton.gemm.grouped.mxfp4_wgrad import (
    MXFP4_WGRAD_BACKENDS,
    check_mxfp4_wgrad_support,
    grouped_mxfp4_wgrad,
    prepare_mxfp4_wgrad_metadata,
)
from aiter.ops.triton.gemm.grouped.sonic_mxfp4 import (
    block_shape,
    grouped_mxfp4_weight_gemm,
    pack_grouped_mxfp4_weights,
    prepare_expert_offsets,
    prepare_grouped_gemm_metadata,
)
from aiter.ops.triton.moe.sonicmoe import activation_bwd, activation_fwd, grouped_gemm
from aiter.ops.triton.quant.grouped_mxfp4_colwise import (
    quantize_grouped_mxfp4_colwise,
)
from aiter.ops.triton.quant.grouped_mxfp4_dual import (
    quantize_grouped_mxfp4_colwise_async,
    quantize_grouped_mxfp4_concurrent,
    quantize_grouped_mxfp4_dual,
)

__all__ = [
    "MXFP4_WGRAD_API_VERSION",
    "MXFP4_WGRAD_BACKENDS",
    "RECIPES",
    "moe_pre_routed_a4w4",
    "moe_pre_routed_a8w4",
    "moe_pre_routed_a8w4f",
    "moe_pre_routed_quantized",
    "moe_pre_routed_w4a16",
    "pack_pre_routed_weights",
]
MXFP4_WGRAD_API_VERSION = 1

RECIPES = {
    "w4a16": ("bf16", "bf16"),
    "a4w4": ("mxfp4", "mxfp4"),
    "a8w4": ("mxfp8", "mxfp8"),
    "a8w4f": ("mxfp8", "bf16"),
}
_FAMILY = {
    "bf16": "GEMM-SONIC-A16W4",
    "mxfp4": "GEMM-SONIC-A4W4",
    "mxfp8": "GEMM-SONIC-A8W4",
}


def _routed_weight(x, score):
    """Scale rows in FP32 and cast once; the debug fallback is bit-equivalent."""
    if os.environ.get("SONIC_MOE_FUSED_ROUTER_SCALE", "1") != "0":
        return row_scale_fp32(x, score)
    return (x.float() * score[:, None]).to(x.dtype)


def pack_pre_routed_weights(w1, w2):
    """Pack four F/D weight orientations; the caller owns post-update invalidation."""
    return (
        pack_grouped_mxfp4_weights(w1.transpose(1, 2)),
        pack_grouped_mxfp4_weights(w2.transpose(1, 2)),
        pack_grouped_mxfp4_weights(w1),
        pack_grouped_mxfp4_weights(w2),
    )


def _metadata(counts, device, shapes, family, offsets=None):
    if offsets is None:
        offsets = prepare_expert_offsets(counts, device)
    blocks = {block_shape(family, max(counts), n, k, len(counts)) for n, k in shapes}
    if len(blocks) != 1:
        return None, offsets
    block_m, block_n = next(iter(blocks))
    return (
        prepare_grouped_gemm_metadata(
            counts,
            device,
            block_m,
            block_n,
            offsets=offsets,
        ),
        offsets,
    )


def _contiguous_byte_ranges_overlap(left, right):
    """Whether two validated contiguous tensors share any actual bytes."""
    left_start = left.data_ptr()
    right_start = right.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_end = right_start + right.numel() * right.element_size()
    return max(left_start, right_start) < min(left_end, right_end)


def _validate_main_grads(main_grads, weights):
    if not isinstance(main_grads, tuple) or len(main_grads) != 2:
        raise ValueError("main_grads must be a two-tensor tuple")
    for grad, weight in zip(main_grads, weights):
        if weight.grad is not None:
            raise ValueError(
                "main_grads requires weight.grad=None; mixed gradient accumulation is unsupported"
            )
        if (
            not isinstance(grad, torch.Tensor)
            or grad.shape != weight.shape
            or grad.dtype != torch.float32
            or grad.device != weight.device
            or not grad.is_contiguous()
            or grad.requires_grad
        ):
            raise ValueError(
                "main_grads must be contiguous non-grad FP32 weight-shaped buffers"
            )
        if any(_contiguous_byte_ranges_overlap(grad, w) for w in weights):
            raise ValueError("main_grad must not overlap a weight")
    if _contiguous_byte_ranges_overlap(*main_grads):
        raise ValueError("main_grad buffers must not overlap each other")


class _PreRoutedMXFP4(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        scores,
        w1,
        w2,
        counts,
        recipe,
        packed,
        wgrad_mode,
        main_grads,
        wgrad_backend,
    ):
        fwd_format, bwd_format = RECIPES[recipe]
        _e, h, j = w1.shape
        i = j // 2
        meta, offsets = _metadata(
            counts, x.device, ((j, h), (h, i)), _FAMILY[fwd_format]
        )
        if _FAMILY[bwd_format] == _FAMILY[fwd_format]:
            bwd_meta = meta
        else:
            bwd_meta, _ = _metadata(
                counts, x.device, ((i, h), (h, j)), _FAMILY[bwd_format], offsets
            )
        dual_quant_mode = os.environ.get("SONIC_MOE_MXFP4_DUAL_QUANT", "0")
        if dual_quant_mode == "1":
            dual_quant_mode = "kernel"
        if dual_quant_mode not in ("0", "kernel", "concurrent"):
            raise ValueError(
                "SONIC_MOE_MXFP4_DUAL_QUANT must be 0, kernel or concurrent"
            )
        if wgrad_mode != "mxfp4":
            dual_quant_mode = "0"
        if wgrad_mode == "mxfp4":
            cfg, _ = check_mxfp4_wgrad_support(
                x.device, j, h, backend=wgrad_backend, transpose_output=True
            )
            wgrad_meta = prepare_mxfp4_wgrad_metadata(
                counts,
                x.device,
                pad_multiple=cfg["BLOCK_SIZE_K"],
                backend=wgrad_backend,
            )
        else:
            wgrad_meta = None
        if dual_quant_mode == "kernel":
            x_quant, x_col = quantize_grouped_mxfp4_dual(x, wgrad_meta)
            x_ready = None
        elif dual_quant_mode == "concurrent":
            x_quant, x_col, x_ready = quantize_grouped_mxfp4_concurrent(x, wgrad_meta)
        else:
            x_quant, x_col, x_ready = None, None, None
        if packed is None:
            packed = pack_pre_routed_weights(w1, w2)
        w1_t, w2_t, _w1_p, _w2_p = packed
        h1 = grouped_mxfp4_weight_gemm(
            x,
            w1_t[0],
            w1_t[1],
            counts,
            activation_format=fwd_format,
            metadata=meta,
            quantized_activation=x_quant,
        )
        a = activation_fwd(h1, i, "swiglu", concat_layout=True)
        if dual_quant_mode == "kernel":
            a_quant, a_col = quantize_grouped_mxfp4_dual(a, wgrad_meta)
            a_ready = None
        elif dual_quant_mode == "concurrent":
            a_quant, a_col, a_ready = quantize_grouped_mxfp4_concurrent(a, wgrad_meta)
        else:
            a_quant, a_col, a_ready = None, None, None
        y = grouped_mxfp4_weight_gemm(
            a,
            w2_t[0],
            w2_t[1],
            counts,
            activation_format=fwd_format,
            metadata=meta,
            quantized_activation=a_quant,
        )
        ctx.counts, ctx.recipe, ctx.packed = counts, recipe, packed
        ctx.bwd_meta, ctx.offsets = bwd_meta, offsets
        ctx.wgrad_mode, ctx.main_grads = wgrad_mode, main_grads
        ctx.wgrad_backend = wgrad_backend
        ctx.wgrad_meta = wgrad_meta
        ctx.dual_quant_mode = dual_quant_mode
        ctx.wgrad_xp, ctx.wgrad_ap = x_col, a_col
        ctx.wgrad_x_ready, ctx.wgrad_a_ready = x_ready, a_ready
        ctx.save_for_backward(x, scores, w1, w2, h1, a, y)
        return _routed_weight(y, scores)

    @staticmethod
    def backward(ctx, dout):
        x, scores, w1, w2, h1, a, y = ctx.saved_tensors
        if ctx.main_grads is not None:
            _validate_main_grads(ctx.main_grads, (w1, w2))
        bwd_format = RECIPES[ctx.recipe][1]
        counts, meta = ctx.counts, ctx.bwd_meta
        dout = dout.contiguous()
        ds = (dout.float() * y.float()).sum(dim=-1)
        _w1_t, _w2_t, w1_p, w2_p = ctx.packed
        unweighted_da = grouped_mxfp4_weight_gemm(
            dout,
            w2_p[0],
            w2_p[1],
            counts,
            activation_format=bwd_format,
            metadata=meta,
        )
        da = _routed_weight(unweighted_da, scores)
        dh = activation_bwd(h1, da, w2.shape[1], "swiglu", concat_layout=True)
        if ctx.dual_quant_mode == "kernel":
            dh_quant, dh_col = quantize_grouped_mxfp4_dual(dh, ctx.wgrad_meta)
            dh_ready = None
        elif ctx.dual_quant_mode == "concurrent":
            dh_quant, dh_col, dh_ready = quantize_grouped_mxfp4_concurrent(
                dh, ctx.wgrad_meta
            )
        else:
            dh_quant, dh_col, dh_ready = None, None, None
        dx = grouped_mxfp4_weight_gemm(
            dh,
            w1_p[0],
            w1_p[1],
            counts,
            activation_format=bwd_format,
            metadata=meta,
            quantized_activation=dh_quant,
        )
        dy = _routed_weight(dout, scores)
        if ctx.wgrad_mode == "mxfp4":
            targets = ctx.main_grads or (None, None)
            if ctx.dual_quant_mode == "concurrent":
                dy_col, dy_ready = quantize_grouped_mxfp4_colwise_async(
                    dy, ctx.wgrad_meta
                )
                dw1_quant = (ctx.wgrad_xp, dh_col)
                dw2_quant = (ctx.wgrad_ap, dy_col)
                current = torch.cuda.current_stream(dy.device)
                for ready in (
                    ctx.wgrad_x_ready,
                    ctx.wgrad_a_ready,
                    dh_ready,
                ):
                    current.wait_event(ready)
            elif ctx.dual_quant_mode == "kernel":
                dy_col = quantize_grouped_mxfp4_colwise(dy, ctx.wgrad_meta)
                dw1_quant = (ctx.wgrad_xp, dh_col)
                dw2_quant = (ctx.wgrad_ap, dy_col)
                dy_ready = None
            else:
                dw1_quant = dw2_quant = None
                dy_ready = None
            # No fallback after a write: a failed native call must fail this backward.
            dw1 = grouped_mxfp4_wgrad(
                x,
                dh,
                counts,
                out=targets[0],
                accumulate=ctx.main_grads is not None,
                metadata=ctx.wgrad_meta,
                quantized_operands=dw1_quant,
                transpose_output=True,
                backend=ctx.wgrad_backend,
            )
            if dy_ready is not None:
                current.wait_event(dy_ready)
            dw2 = grouped_mxfp4_wgrad(
                a,
                dy,
                counts,
                out=targets[1],
                accumulate=ctx.main_grads is not None,
                metadata=ctx.wgrad_meta,
                quantized_operands=dw2_quant,
                transpose_output=True,
                backend=ctx.wgrad_backend,
            )
            if ctx.main_grads is not None:
                for weight in (w1, w2):
                    if hasattr(weight, "grad_added_to_main_grad"):
                        weight.grad_added_to_main_grad = True
                dw1, dw2 = None, None
            else:
                dw1, dw2 = dw1.to(w1.dtype), dw2.to(w2.dtype)
        else:
            # Preserve the current BF16 control; zeros give empty experts defined gradients.
            dw1, dw2 = torch.zeros_like(w1), torch.zeros_like(w2)
            grouped_gemm(x, dh, ctx.offsets, out=dw1, A_is_transposed=True)
            grouped_gemm(a, dy, ctx.offsets, out=dw2, A_is_transposed=True)
        return dx, ds, dw1, dw2, None, None, None, None, None, None


def _pre_routed_mxfp4(
    x,
    scores,
    counts,
    w1,
    w2,
    recipe,
    packed=None,
    *,
    wgrad_mode="bf16",
    main_grads=None,
    wgrad_backend="triton",
):
    """Return BF16 expert-major [R,H]; weights are BF16 w1[E,H,2I], w2[E,I,H].

    Native Wgrad accepts caller-owned FP32 main_grads and always adds to them.
    The caller must zero buffers at each optimizer step and invalidate weight packs
    after update/load. With no buffers, autograd returns parameter-dtype gradients.
    """
    if recipe not in RECIPES:
        raise ValueError("invalid recipe")
    if wgrad_backend not in MXFP4_WGRAD_BACKENDS:
        raise ValueError("wgrad_backend must be triton or gluon")
    if wgrad_mode not in ("bf16", "mxfp4") or (
        wgrad_mode == "mxfp4" and recipe != "a4w4"
    ):
        raise ValueError("native MXFP4 Wgrad is supported only for the a4w4 recipe")
    if main_grads is not None and wgrad_mode != "mxfp4":
        raise ValueError("main_grads requires native MXFP4 Wgrad")
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous CUDA BF16 [R,H]")
    if (
        scores.shape != (x.shape[0],)
        or scores.dtype != torch.float32
        or scores.device != x.device
    ):
        raise ValueError("scores must be FP32 [R] on x.device")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("weights must be rank3")
    e, h, j = w1.shape
    if min(e, h, j) <= 0 or j % 2 or w2.shape != (e, j // 2, h) or x.shape[1] != h:
        raise ValueError("expected w1[E,H,2I], w2[E,I,H], x[R,H]")
    if any(
        w.dtype != x.dtype or w.device != x.device or not w.is_contiguous()
        for w in (w1, w2)
    ):
        raise ValueError("weights must be contiguous BF16 on x.device")
    if isinstance(counts, torch.Tensor):
        if counts.ndim != 1 or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("counts must be an integer vector")
        counts = tuple(counts.detach().cpu().tolist())
    if not isinstance(counts, (list, tuple)) or len(counts) != e:
        raise ValueError("counts must have E entries")
    counts = tuple(counts)
    if any(type(c) is not int or c < 0 for c in counts) or sum(counts) != x.shape[0]:
        raise ValueError("counts must be nonnegative and sum to R")
    if wgrad_mode == "mxfp4":
        cfg1, _ = check_mxfp4_wgrad_support(
            x.device, j, h, backend=wgrad_backend, transpose_output=True
        )
        cfg2, _ = check_mxfp4_wgrad_support(
            x.device, h, j // 2, backend=wgrad_backend, transpose_output=True
        )
        if cfg1["BLOCK_SIZE_K"] != cfg2["BLOCK_SIZE_K"]:
            raise NotImplementedError(
                "W1/W2 require the same metadata padding contract"
            )
        if main_grads is not None:
            _validate_main_grads(main_grads, (w1, w2))
    return _PreRoutedMXFP4.apply(
        x, scores, w1, w2, counts, recipe, packed, wgrad_mode, main_grads, wgrad_backend
    )


def moe_pre_routed_w4a16(x, scores, counts, w1, w2, packed=None):
    """W4A16 F/D with BF16 Wgrad, returning expert-major BF16 [R,H]."""
    return _pre_routed_mxfp4(x, scores, counts, w1, w2, "w4a16", packed)


def moe_pre_routed_a4w4(
    x,
    scores,
    counts,
    w1,
    w2,
    packed=None,
    *,
    wgrad_mode="bf16",
    main_grads=None,
    wgrad_backend="triton",
):
    """A4W4 F/D; optional native Wgrad adds into two FP32 weight-shaped buffers."""
    return _pre_routed_mxfp4(
        x,
        scores,
        counts,
        w1,
        w2,
        "a4w4",
        packed,
        wgrad_mode=wgrad_mode,
        main_grads=main_grads,
        wgrad_backend=wgrad_backend,
    )


def moe_pre_routed_a8w4(x, scores, counts, w1, w2, packed=None):
    """A8W4 F/D with BF16 Wgrad, returning expert-major BF16 [R,H]."""
    return _pre_routed_mxfp4(x, scores, counts, w1, w2, "a8w4", packed)


def moe_pre_routed_a8w4f(x, scores, counts, w1, w2, packed=None):
    """A8W4 forward, weight-only dgrad, and BF16 Wgrad."""
    return _pre_routed_mxfp4(x, scores, counts, w1, w2, "a8w4f", packed)


def moe_pre_routed_quantized(x, scores, counts, w1, w2, recipe="a8w4f", packed=None):
    """Explicit recipe entry; the historical default stays BF16 Wgrad."""
    return _pre_routed_mxfp4(x, scores, counts, w1, w2, recipe, packed)
