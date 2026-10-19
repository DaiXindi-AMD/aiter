# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import hashlib
import importlib
import inspect
import json
import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_kernel as _triton_gemm_afp4wfp4_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_kernel_preshuffle_scales as _triton_gemm_afp4wfp4_kernel_preshuffle_scales,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_preshuffle_kernel as _triton_gemm_afp4wfp4_preshuffle_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _get_config,
)
from aiter.ops.triton.utils import mxfp4_contract as contract
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.common_utils import deserialize_str, serialize_dict
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_USE_GEMM_SPLITK_BF16 = False

_GLUON_SUPPORTED_ARCHS = ("gfx950",)
_GLUON_PRESHUFFLE_ARCHS = ("gfx1250",)
_GLUON_DEFAULT_ARCHS = ("gfx1250",)


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value


def get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
    # heuristics for make "EVEN_K == True" as much as possible
    NUM_KSPLIT_STEP = 2
    BLOCK_SIZE_K_STEP = 2
    # Both decrements floor-divide, so they must be clamped: a NUM_KSPLIT below
    # NUM_KSPLIT_STEP would go to 0 and the next cdiv(K, NUM_KSPLIT) would
    # divide by zero, and BLOCK_SIZE_K can fall under the 16 that
    # compute_splitk_params() enforces everywhere else.
    SPLITK_BLOCK_SIZE = (
        triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
    )
    while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
        if (
            K % (SPLITK_BLOCK_SIZE // 2) == 0
            and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
            and K % (BLOCK_SIZE_K // 2) == 0
        ):
            break
        elif K % (SPLITK_BLOCK_SIZE // 2) != 0 and NUM_KSPLIT > 1:
            NUM_KSPLIT = max(NUM_KSPLIT // NUM_KSPLIT_STEP, 1)
        elif SPLITK_BLOCK_SIZE % BLOCK_SIZE_K != 0:
            if NUM_KSPLIT > 1:
                NUM_KSPLIT = max(NUM_KSPLIT // NUM_KSPLIT_STEP, 1)
            elif BLOCK_SIZE_K > 16:
                BLOCK_SIZE_K = max(BLOCK_SIZE_K // BLOCK_SIZE_K_STEP, 16)
        elif K % (BLOCK_SIZE_K // 2) != 0 and BLOCK_SIZE_K > 16:
            BLOCK_SIZE_K = max(BLOCK_SIZE_K // BLOCK_SIZE_K_STEP, 16)
        else:
            break

        SPLITK_BLOCK_SIZE = (
            triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
        )

    # re-ensuring NUM_KSPLIT is the correct value
    NUM_KSPLIT = triton.cdiv(K, (SPLITK_BLOCK_SIZE // 2))

    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


def gemm_afp4wfp4_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: str | None = None,
    skip_reduce: bool | None = False,
    backend: str = "triton",
) -> torch.Tensor:
    M, K = x.shape
    N, _ = w.shape

    config = deserialize_str(config)

    num_ksplit = config["NUM_KSPLIT"]
    block_size_k = config["BLOCK_SIZE_K"]

    if num_ksplit > 1:
        _, block_size_k, num_ksplit = get_splitk(K, block_size_k, num_ksplit)

    if block_size_k >= 2 * K:
        num_ksplit = 1

    return_y_pp = num_ksplit > 1 and skip_reduce
    if return_y_pp:
        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty((num_ksplit, M, N), dtype=y.dtype, device=x.device)
        else:
            y_pp = torch.empty((num_ksplit, M, N), dtype=torch.float32, device=x.device)
        return y_pp

    return torch.empty((M, N), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=gemm_afp4wfp4_fake_tensor)
def gemm_afp4wfp4_(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: str | None = None,
    skip_reduce: bool | None = False,
    backend: str = "triton",
) -> torch.Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K//2).
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K//2), internally transposed.
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M, K//32).
            One scale per 32 elements in K dimension.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        skip_reduce (Optional[bool]): skip reduction, y becomes (SPK, M, N) where SPK is determined by config

    Returns:
        y (torch.Tensor): Output with shape (M, N) or (SPK, M, N).
    """
    _LOGGER.info(
        "GEMM_AFP4WFP4: x.shape=%s w.shape=%s x_scale=%s w_scale=%s ",
        tuple(x.shape),
        tuple(w.shape),
        tuple(x_scales.shape),
        tuple(w_scales.shape),
    )

    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    # gfx1150 and gfx1151 (RDNA3.5) can lower this Triton dot_scaled kernel
    # without native FP4 instructions. Keep the exception local: other FP4
    # entry points need their own configs and validation before they can run
    # on these targets.
    assert arch_info.is_fp4_avail() or (
        backend == "triton" and arch_info.get_arch() in ("gfx1150", "gfx1151")
    ), "MXFP4 is not available on your device"

    if backend == "gluon":
        arch = arch_info.get_arch()
        assert arch in _GLUON_SUPPORTED_ARCHS, (
            f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{arch}'. "
            "gfx1250 gluon MXFP4 goes through gemm_afp4wfp4_preshuffle."
        )
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_afp4wfp4 import (
            _gemm_afp4wfp4_kernel as _gluon_gemm_afp4wfp4_gfx950_kernel,
        )

        impl = _gluon_gemm_afp4wfp4_gfx950_kernel
    else:
        impl = _triton_gemm_afp4wfp4_kernel

    M, K = x.shape
    N, K = w.shape

    # Transpose w
    w = w.T

    if config is None:
        if backend == "gluon":
            config, _ = _get_config(M, N, K, backend="gluon")
        else:
            config, _ = _get_config(M, N, K)
    else:
        config = deserialize_str(config)

    if backend == "gluon":
        # The gluon kernel rounds SPLITK_BLOCK_SIZE up to a BLOCK_SIZE_K
        # multiple and leaves BLOCK_SIZE_K alone; get_splitk's shrinking is
        # tuned for the triton kernel.
        if config["BLOCK_SIZE_K"] >= 2 * K:
            config["NUM_KSPLIT"] = 1
        if config["NUM_KSPLIT"] > 1:
            config["SPLITK_BLOCK_SIZE"] = (
                triton.cdiv(
                    2 * triton.cdiv(K, config["NUM_KSPLIT"]), config["BLOCK_SIZE_K"]
                )
                * config["BLOCK_SIZE_K"]
            )
    elif config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

    if backend == "triton":
        if config["BLOCK_SIZE_K"] >= 2 * K:
            config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
            config["SPLITK_BLOCK_SIZE"] = 2 * K
            config["NUM_KSPLIT"] = 1
        config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 128)

    return_y_pp = config["NUM_KSPLIT"] > 1 and skip_reduce

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=dtype if _USE_GEMM_SPLITK_BF16 else torch.float32,
            device=x.device,
        )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if y is None and not return_y_pp:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    impl[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    if return_y_pp:
        return y_pp
    elif config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_afp4wfp4_reduce_kernel",
        )

    return y


def gemm_afp4wfp4(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    backend: str | None = None,
) -> torch.Tensor:
    arch = arch_info.get_arch()
    if backend is None:
        backend = "gluon" if arch in _GLUON_DEFAULT_ARCHS else "triton"

    if backend == "gluon" and arch not in _GLUON_SUPPORTED_ARCHS:
        # This entry point takes w as (N, K//2) row-major. gfx1250's gluon MXFP4
        # kernel takes preshuffled weights, so it cannot serve it.
        _LOGGER.info(
            "GEMM_AFP4WFP4: no gluon kernel for unshuffled weights on %s, using triton",
            arch,
        )
        backend = "triton"

    if config is None:
        M, K = x.shape
        N, _ = w.shape
        if backend == "gluon":
            config, _ = _get_config(M, N, K, backend="gluon")
        else:
            config, _ = _get_config(M, N, K)
    config_hashable = serialize_dict(config)
    return gemm_afp4wfp4_(
        x, w, x_scales, w_scales, dtype, y, config_hashable, skip_reduce, backend
    )


def gemm_afp4wfp4_preshuffled_scales(
    x,
    w,
    x_scales,
    w_scales,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
):
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights using preshuffled scales.
    Scales are arranged with M/N dimension grouped by 32 instead of K dimension.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K). M >= 32 required.
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K), internally transposed.
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M//32, K).
            Groups of 32 rows in M dimension share K scales.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N//32, K).
            Groups of 32 rows in N dimension share K scales.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape

    # Transpose w
    w = w.T

    assert M >= 32, f"M >= 32 is required, but got {M=}"

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config, _ = _get_config(M, N, K)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
            )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    _triton_gemm_afp4wfp4_kernel_preshuffle_scales[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_afp4wfp4_reduce_kernel",
        )

    return y


def gemm_afp4wfp4_preshuffle(
    x_fp4: torch.Tensor,
    w_preshuf: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
) -> torch.Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights.
    Weight matrix and scales are stored in optimized layout for improved performance.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K//2).
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N//16, K*16),
            preshuffled via aiter.ops.shuffle.shuffle_weight_gfx1250 (or
            equivalently shuffle_weight(layout=(16, 16)) reshaped to
            (N//16, K*16) — both produce the same bytes). Internally transposed.
        x_scales (torch.Tensor): E8M0 per-group scale for x, one scale per 32
            elements in K. For M >= 32: preshuffled via
            aiter.ops.shuffle.shuffle_scale (identical layout on gfx950 and
            gfx1250), then viewed as one row per 32-row stripe, i.e.
            (M_pad//32, (K//32)*32) where M_pad = M rounded up to a multiple
            of 256. shuffle_scale returns the buffer flat as
            (M_pad, K//32), so the caller owns the reshape:
            ``s = shuffle_scale(x_scales); s.view(-1, s.shape[-1] * 32)``
            (take the shape off the shuffled tensor -- shuffle_scale pads both
            dims). For M < 32: un-shuffled (M, K//32) row-major.
        w_scales (torch.Tensor): E8M0 per-group scale for w, one scale per 32
            elements in K: same shuffle_scale + stripe view as x_scales,
            i.e. (N_pad//32, (K//32)*32). Always shuffled, including M < 32.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        skip_reduce (Optional[bool]): skip reduction, y becomes (SPK, M, N) where SPK is determined by config

    Returns:
        y (torch.Tensor): Output with shape (M, N) or (SPK, M, N).
    """

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"
    use_gluon = arch_info.get_arch() in _GLUON_PRESHUFFLE_ARCHS

    M, K_bytes = x_fp4.shape
    n16, _ = w_preshuf.shape
    N = n16 * 16
    K_elems = 2 * K_bytes

    if config is None:
        # _get_config doubles K itself (logical K = 2 * K_bytes) — pass bytes,
        # matching the non-preshuffled path. The two backends take disjoint
        # params (gluon: NUM_BUFFERS, triton: NUM_KSPLIT/GROUP_SIZE_M/...), so
        # the config must come from the dir of the backend we actually launch.
        config, _ = _get_config(
            M, N, K_bytes, True, backend="gluon" if use_gluon else "triton"
        )

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)
    if M < 32:
        assert (
            config["BLOCK_SIZE_M"] <= 16
        ), "for M < 32, BLOCK_SIZE_M must be 16 or less as x_scale are assumed to be un-shuffled"
    else:
        assert (
            config["BLOCK_SIZE_M"] >= 32
        ), "for M >= 32, BLOCK_SIZE_M must be 32 or more as x_scale are assumed to be preshuffled"

    # shuffle_scale pads K//32 up to a multiple of 8 (its k-chunk), but the
    # kernels bound the scale reads at (K//32)*32 columns per stripe row. A
    # k-chunk interleaves its 8 k-groups across all 256 of its bytes, so a
    # bound that lands mid-chunk silently masks *real* k-groups to zero rather
    # than only the padding -- e.g. K=896 loses k-groups 26 and 27. Require
    # whole k-chunks. w_scales are always preshuffled, so this holds for M < 32
    # too, where x_scales themselves are un-shuffled.
    assert K_elems % 256 == 0, (
        f"preshuffled scales require K % 256 == 0 (K//32 a multiple of 8), got K={K_elems}. "
        "shuffle_scale pads K//32 to a multiple of 8 and the kernels cannot mask a "
        "partial k-chunk"
    )

    if use_gluon:
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp4 import (
            cluster_shape,
            get_gemm_afp4wfp4_preshuffle_layouts,
        )
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp4 import (
            gemm_mxfp4_preshuffle_gfx1250 as _gluon_gemm_mxfp4_preshuffle_gfx1250,
        )

        grid = lambda META: (
            (
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"])
            ),
        )
        # gluon path does not support splitk; config has no NUM_KSPLIT / SPLITK_BLOCK_SIZE
        if y is None:
            y = torch.empty((M, N), dtype=dtype, device=x_fp4.device)

        # Clamp NUM_BUFFERS so  prologue never advances TDM descriptors past the end of K when k_tiles < NUM_BUFFERS (BLOCK_K_BYTES = BLOCK_SIZE_K // 2)
        BLOCK_K_BYTES = config["BLOCK_SIZE_K"] // 2
        k_tiles = triton.cdiv(K_bytes, BLOCK_K_BYTES)
        config["NUM_BUFFERS"] = min(config["NUM_BUFFERS"], k_tiles)

        num_ctas = config["num_ctas"]
        ctas_m, _ = cluster_shape(
            num_ctas, config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
        )
        # The kernel takes a CTA's A scales as preshuffled iff it has 32+ rows.
        assert M < 32 or config["BLOCK_SIZE_M"] // ctas_m >= 32, (
            f"for M >= 32, each CTA needs 32 or more rows of BLOCK_SIZE_M="
            f"{config['BLOCK_SIZE_M']}, which num_ctas={num_ctas} splits {ctas_m} ways"
        )

        layouts = get_gemm_afp4wfp4_preshuffle_layouts(
            config["num_warps"],
            config["BLOCK_SIZE_M"],
            config["BLOCK_SIZE_N"],
            config["BLOCK_SIZE_K"],
            num_ctas,
        )

        _gluon_gemm_mxfp4_preshuffle_gfx1250[grid](
            x_fp4,
            w_preshuf,
            y,
            x_scales,
            w_scales,
            M,
            N,
            K_elems,
            x_fp4.stride(0),
            x_fp4.stride(1),
            w_preshuf.stride(0),
            w_preshuf.stride(1),
            y.stride(0),
            y.stride(-2),
            y.stride(-1),
            x_scales.stride(0),
            x_scales.stride(1),
            w_scales.stride(0),
            w_scales.stride(1),
            **config,
            **layouts,
        )
        return y

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K_bytes, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=x_fp4.device
            )
        else:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x_fp4.device
            )
    else:
        config["SPLITK_BLOCK_SIZE"] = K_elems
        y_pp = None

    return_y_pp = config["NUM_KSPLIT"] > 1 and skip_reduce

    if y is None and not return_y_pp:
        y = torch.empty((M, N), dtype=dtype, device=x_fp4.device)

    if config["BLOCK_SIZE_K"] >= K_elems:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(K_elems)
        config["SPLITK_BLOCK_SIZE"] = K_elems

    M_POW2 = triton.next_power_of_2(M)
    if M < 32 and M_POW2 > 16:
        M_POW2 = 16

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    config.pop("NUM_BUFFERS", None)
    _triton_gemm_afp4wfp4_preshuffle_kernel[grid](
        x_fp4,
        w_preshuf,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K_bytes,
        x_fp4.stride(0),
        x_fp4.stride(1),
        w_preshuf.stride(0),
        w_preshuf.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    if return_y_pp:
        return y_pp
    elif config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K_bytes, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_afp4wfp4_reduce_kernel",
        )

    return y


def gemm_afp4wfp4_preshuffled_weight_scales(
    x,
    w,
    x_scales,
    w_scales,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
):
    """
    This this a backward-compatible API and will be deprecated in future release
    """
    _LOGGER.info(
        "gemm_afp4wfp4_preshuffled_weight_scales will be deprecated in future AITER release, please switch to gemm_afp4wfp4_preshuffle"
    )
    return gemm_afp4wfp4_preshuffle(x, w, x_scales, w_scales, dtype, y, config)


@dataclass(frozen=True)
class PreparedDenseV1:
    static_spec: tuple
    capability: contract.DenseCapabilityV1
    records: object
    device: object
    artifact_digest: str
    config_view: contract.DenseConfigViewV1 = dataclass_field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "config_view",
            contract._prepared_config_view_v1(
                self.static_spec, self.records, self.artifact_digest, self.device
            ),
        )


@dataclass(frozen=True)
class BoundDenseV1:
    prepared: PreparedDenseV1
    logical_M: int
    buffers: object
    metadata: object
    x_identity: int
    x_metadata: contract.TensorMetadataV1
    dy_schema: tuple
    R_dy_row: tuple
    R_dy_t: tuple


@dataclass(frozen=True)
class RuntimeOperandsV1:
    bound: BoundDenseV1
    dy: object
    metadata: contract.TensorMetadataV1


def mxfp4_dense_query_static_v1(static_spec):
    """Query the two dense-training linear shapes without files or device access."""
    s = contract.validate_static_spec(static_spec)
    supported = (
        s[2] == 950 and s[4] == 1 and (s[5], s[6]) in ((4096, 4096), (4096, 2048))
    )
    shapes = contract.buffer_shapes(s[7], s[5], s[6])
    for name, shape in shapes:
        size = (
            4
            if name in ("w_out", "status")
            else (2 if name in ("y_out", "dx_out") else 1)
        )
        elements = 1
        for dim in shape:
            elements *= dim
        if elements * size > contract.SPAN_MAX:
            contract.fail(
                contract.Phase.QUERY, contract.Fatal.CAPACITY_OR_INDEX_OVERFLOW
            )
    if 2 * s[7] * max(s[5], s[6]) > contract.SPAN_MAX:
        contract.fail(contract.Phase.QUERY, contract.Fatal.CAPACITY_OR_INDEX_OVERFLOW)
    return contract.DenseCapabilityV1(
        0 if supported else 1,
        0 if supported else 1,
        0,
        0,
        shapes,
        (("bf16", 2), ("fp32", 4), ("payload", 1), ("scale", 1), ("status", 4)),
        contract.SPAN_MAX,
        (0, 1),
        False,
        True,
    )


def _record_key(
    op, pass_id=-1, variant=0, source=0, output=0, group=1, n=0, k=0, mode=0
):
    return (op, pass_id, variant, source, output, group, n, k, mode, "any")


def _bound_record(prepared, key):
    try:
        return prepared.records[key]
    except KeyError:
        contract.fail(contract.Phase.BIND, contract.Fatal.UNPREPARED_SPECIALIZATION)


@contract.register_contract_op(
    ("A", "B", "A_scales", "B_scales", "out", "workspace", "status"),
    ("out", "workspace", "status"),
    {"H": 6, "G": 11},
)
def gemm_afp4wfp4_out_v1(
    A,
    B,
    A_scales,
    B_scales,
    out,
    workspace,
    status,
    M_g,
    N_g,
    K_g,
    pass_id,
    b_scale_row_group,
    output_mode,
    output_dtype,
    H,
    G,
):
    """Enqueue prepared single-split A @ B.T into caller storage; return None.

    F/D write BF16. W can overwrite or accumulate directly in FP32.
    """
    phase = contract.Phase.LAUNCH
    contract.header(H, 4, phase=phase)
    for value in (M_g, N_g, K_g):
        contract.integer(value, high=contract.SPAN_MAX, phase=phase)
    for value in (output_dtype, output_mode):
        contract.integer(value, high=1, phase=phase)
    contract.integer(pass_id, high=2, phase=phase)
    contract.integer(b_scale_row_group, 1, 32, phase=phase)
    if (
        b_scale_row_group not in (1, 32)
        or K_g % 32
        or (output_mode and output_dtype != 1)
    ):
        contract.fail(phase)
    record = contract.ready_record(G, 4)
    key = record.semantic_key
    if (
        key[1] != pass_id
        or key[4:6] != (output_dtype, b_scale_row_group)
        or key[8] != output_mode
    ):
        contract.fail(phase, contract.Fatal.UNPREPARED_SPECIALIZATION)
    n, k = key[6:8]
    if pass_id == 0:
        valid = (N_g, K_g, output_mode, output_dtype) == (n, k, 0, 0)
    elif pass_id == 1:
        valid = (N_g, K_g, output_mode, output_dtype) == (k, n, 0, 0)
    else:
        valid = (M_g, N_g, b_scale_row_group) == (n, k, 1)
    if not valid or A.device != record.device:
        contract.fail(phase)
    device = A.device
    a = contract.check_tensor(A, (M_g, K_g // 2), "torch.uint8", device)
    b = contract.check_tensor(B, (N_g, K_g // 2), "torch.uint8", device)
    sa = contract.check_tensor(A_scales, (M_g, K_g // 32), "torch.uint8", device)
    sb = contract.check_tensor(
        B_scales,
        (N_g // b_scale_row_group, K_g // 32),
        "torch.uint8",
        device,
        transpose_scale=True,
    )
    target = contract.check_tensor(
        out, (M_g, N_g), ("torch.bfloat16", "torch.float32")[output_dtype], device
    )
    scratch = contract.check_tensor(workspace, (0,), "torch.uint8", device)
    state = contract.status_metadata(status, device)
    contract.disjoint((target, scratch, state), (a, b, sa, sb))
    if M_g == 0:
        return
    if K_g == 0:
        if pass_id != 2:
            contract.fail(phase)
        if output_mode == 0:
            zero = contract.companion_record(
                record, _record_key(5, variant=4, output=output_dtype)
            )
            mxfp4_zero_output_out_v1(
                out, M_g, N_g, output_dtype, (1, 0, 5, 0, 0, 0), zero.config
            )
        return
    _, bm, bn, _, _, _, _, _, _, _, _ = G
    record.executable[(triton.cdiv(M_g, bm) * triton.cdiv(N_g, bn), 1, 1)](
        A,
        B,
        out,
        A_scales,
        B_scales,
        M_g,
        N_g,
        K_g // 2,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        0,
        out.stride(0),
        out.stride(1),
        *A_scales.stride(),
        *B_scales.stride(),
        status,
    )
    return


@contract.register_contract_op(("out",), ("out",), {"H": 6, "Z": 6})
def mxfp4_zero_output_out_v1(out, rows, cols, output_dtype, H, Z):
    """Write positive zero without an input read for the two parameter shapes."""
    phase = contract.Phase.LAUNCH
    contract.header(H, 5, phase=phase)
    contract.integer(rows, phase=phase)
    contract.integer(cols, phase=phase)
    contract.integer(output_dtype, high=1, phase=phase)
    record = contract.ready_record(Z, 5)
    if (rows, cols) not in ((4096, 4096), (4096, 2048)) or record.semantic_key[
        4
    ] != output_dtype:
        contract.fail(phase)
    contract.check_tensor(
        out,
        (rows, cols),
        ("torch.bfloat16", "torch.float32")[output_dtype],
        record.device,
    )
    _, bm, bn, _, _, _ = Z
    if rows % bm or cols % bn:
        contract.fail(phase)
    record.executable[(rows // bm, cols // bn, 1)](
        out,
        out,
        out,
        rows,
        cols,
        rows * cols,
        cols,
        1,
        cols,
        1,
    )


def mxfp4_dense_bind_v1(
    prepared,
    logical_M,
    x,
    w_fp4,
    w_scales,
    wt_fp4,
    wt_scales,
    x_row_out,
    x_row_scales_out,
    x_t_out,
    x_t_scales_out,
    dy_row_out,
    dy_row_scales_out,
    dy_t_out,
    dy_t_scales_out,
    y_out,
    dx_out,
    w_out,
    workspace,
    status,
    dy_schema,
    R_dy_row,
    R_dy_t,
):
    """Validate and freeze all caller buffers without device mutation or allocation."""
    phase = contract.Phase.BIND
    if not isinstance(prepared, PreparedDenseV1):
        contract.fail(phase)
    s = prepared.static_spec
    contract.integer(logical_M, high=s[7], phase=phase, reason=103)
    n, k = s[5:7]
    p = contract.pad32(logical_M)
    contract.rng_words(1, R_dy_row, logical_M * n)
    contract.rng_words(1, R_dy_t, p * n)
    expected_dy = (0, logical_M, n, n, 1, 2, prepared.device.index)
    contract.flat_tuple(dy_schema, 7)
    if dy_schema != expected_dy:
        contract.fail(phase)
    values = (
        w_fp4,
        w_scales,
        wt_fp4,
        wt_scales,
        x_row_out,
        x_row_scales_out,
        x_t_out,
        x_t_scales_out,
        dy_row_out,
        dy_row_scales_out,
        dy_t_out,
        dy_t_scales_out,
        y_out,
        dx_out,
        w_out,
        workspace,
        status,
    )
    shapes = contract.buffer_shapes(logical_M, n, k)
    buffers = dict(zip((name for name, _ in shapes), values))
    metadata = {}
    if not isinstance(x, torch.Tensor) or any(
        not isinstance(value, torch.Tensor) for value in values
    ):
        contract.fail(phase)
    for (name, shape), tensor in zip(shapes, values):
        dtype = "torch.uint8"
        if name in ("y_out", "dx_out"):
            dtype = "torch.bfloat16"
        elif name == "w_out":
            dtype = ("torch.bfloat16", "torch.float32")[s[10]]
        elif name == "status":
            dtype = "torch.int32"
        metadata[name] = contract.check_tensor(
            tensor, shape, dtype, prepared.device, transpose_scale=name == "wt_scales"
        )
    source = contract.check_tensor(x, (logical_M, k), "torch.bfloat16", prepared.device)
    readonly = tuple(metadata[name] for name, _ in shapes[:4]) + (source,)
    writable = tuple(metadata[name] for name, _ in shapes[4:])
    contract.disjoint(writable, readonly)
    for key in (
        _record_key(2, variant=2),
        _record_key(2, variant=3),
        _record_key(4, 0, group=32, n=n, k=k),
        _record_key(4, 1, group=32, n=n, k=k),
        _record_key(4, 2, output=s[10], n=n, k=k, mode=s[9]),
        _record_key(5, variant=4, output=s[10]),
    ):
        _bound_record(prepared, key)
    return BoundDenseV1(
        prepared,
        logical_M,
        MappingProxyType(buffers),
        MappingProxyType(metadata),
        id(x),
        source,
        dy_schema,
        R_dy_row,
        R_dy_t,
    )


def mxfp4_dense_attach_backward_v1(bound, dy):
    """Attach future dY metadata; destinations, code and RNG remain bound."""
    phase = contract.Phase.ATTACH_POSTCOMMIT
    try:
        if not isinstance(dy, torch.Tensor):
            contract.fail(phase, 109)
        n = bound.prepared.static_spec[5]
        meta = contract.check_tensor(
            dy,
            (bound.logical_M, n),
            "torch.bfloat16",
            bound.prepared.device,
            phase=phase,
        )
        contract.disjoint((meta,), tuple(bound.metadata.values()), phase=phase)
    except (contract.Mxfp4ContractError, AttributeError) as exc:
        raise contract.Mxfp4ContractError(phase, 109) from exc
    return RuntimeOperandsV1(bound, dy, meta)


def _verify_bound_storage(bound):
    if not isinstance(bound, BoundDenseV1):
        contract.fail(contract.Phase.LAUNCH, contract.Fatal.BOUND_KERNEL_FAILURE)
    for name, tensor in bound.buffers.items():
        if (
            contract.tensor_metadata(tensor, phase=contract.Phase.LAUNCH)
            != bound.metadata[name]
        ):
            contract.fail(contract.Phase.LAUNCH, contract.Fatal.BOUND_KERNEL_FAILURE)


def _launch_bound_gemm(bound, pass_id):
    b = bound.buffers
    s = bound.prepared.static_spec
    m, n, k = bound.logical_M, s[5], s[6]
    if pass_id == 0:
        names, dims, group, mode, dtype = (
            ("x_row_out", "w_fp4", "x_row_scales_out", "w_scales", "y_out"),
            (m, n, k),
            32,
            0,
            0,
        )
    elif pass_id == 1:
        names, dims, group, mode, dtype = (
            ("dy_row_out", "wt_fp4", "dy_row_scales_out", "wt_scales", "dx_out"),
            (m, k, n),
            32,
            0,
            0,
        )
    else:
        names, dims, group, mode, dtype = (
            ("dy_t_out", "x_t_out", "dy_t_scales_out", "x_t_scales_out", "w_out"),
            (n, k, contract.pad32(m)),
            1,
            s[9],
            s[10],
        )
    record = _bound_record(
        bound.prepared,
        _record_key(4, pass_id, output=dtype, group=group, n=n, k=k, mode=mode),
    )
    gemm_afp4wfp4_out_v1(
        *(b[name] for name in names),
        b["workspace"],
        b["status"],
        *dims,
        pass_id,
        group,
        mode,
        dtype,
        (1, 0, 4, 1, 2, 0),
        record.config,
    )


def mxfp4_dense_launch_forward_v1(bound, x):
    """Enqueue original X dual quantization and Fprop on the current stream."""
    from aiter.ops.triton.quant.dual_layout_mxfp4 import dual_layout_quant_mxfp4_out_v1

    _verify_bound_storage(bound)
    if id(x) != bound.x_identity or contract.tensor_metadata(x) != bound.x_metadata:
        contract.fail(contract.Phase.LAUNCH, contract.Fatal.BOUND_KERNEL_FAILURE)
    if bound.logical_M == 0:
        return
    b = bound.buffers
    m, k = bound.logical_M, bound.prepared.static_spec[6]
    record = _bound_record(bound.prepared, _record_key(2, variant=2))
    dual_layout_quant_mxfp4_out_v1(
        x,
        b["x_row_out"],
        b["x_row_scales_out"],
        b["x_t_out"],
        b["x_t_scales_out"],
        b["status"],
        m,
        k,
        contract.pad32(m),
        0,
        1,
        0,
        0,
        (1, 0, 2, 1, 2, 0),
        (0,) * 7,
        (0,) * 7,
        record.config,
    )
    _launch_bound_gemm(bound, 0)
    return


def mxfp4_dense_launch_packed_forward_v1(bound):
    """Replay Fprop from retained original packed X without quantization."""
    _verify_bound_storage(bound)
    if bound.logical_M:
        _launch_bound_gemm(bound, 0)


def mxfp4_dense_launch_backward_v1(bound, runtime_operands):
    """Enqueue independent dY row/H16T, DGrad and direct caller WGrad."""
    from aiter.ops.triton.quant.dual_layout_mxfp4 import dual_layout_quant_mxfp4_out_v1

    _verify_bound_storage(bound)
    if (
        not isinstance(runtime_operands, RuntimeOperandsV1)
        or runtime_operands.bound is not bound
        or contract.tensor_metadata(runtime_operands.dy) != runtime_operands.metadata
    ):
        contract.fail(
            contract.Phase.LAUNCH, contract.Fatal.RUNTIME_OPERAND_CONTRACT_VIOLATION
        )
    b, s = bound.buffers, bound.prepared.static_spec
    m, n, k = bound.logical_M, s[5], s[6]
    if m == 0:
        if s[9] == 0:
            record = _bound_record(
                bound.prepared, _record_key(5, variant=4, output=s[10])
            )
            mxfp4_zero_output_out_v1(
                b["w_out"], n, k, s[10], (1, 0, 5, 0, 0, 0), record.config
            )
        return
    record = _bound_record(bound.prepared, _record_key(2, variant=3))
    dual_layout_quant_mxfp4_out_v1(
        runtime_operands.dy,
        b["dy_row_out"],
        b["dy_row_scales_out"],
        b["dy_t_out"],
        b["dy_t_scales_out"],
        b["status"],
        m,
        n,
        contract.pad32(m),
        0,
        1,
        1,
        1,
        (1, 0, 2, 1, 2, 0),
        bound.R_dy_row,
        bound.R_dy_t,
        record.config,
    )
    _launch_bound_gemm(bound, 1)
    _launch_bound_gemm(bound, 2)
    return


def _compile_contract_record(record, values, options, device, artifact_digest):
    from triton.compiler import ASTSource

    op = record["op_id"]
    module_name, symbol = record["kernel_symbol"].rsplit(".", 1)
    body = getattr(importlib.import_module(module_name), symbol)
    while not hasattr(body, "arg_names"):
        body = body.fn
    source_file = inspect.getsourcefile(body.fn)
    with open(source_file, "rb") as stream:
        source_hash = hashlib.sha256(stream.read()).hexdigest()
    if source_hash != record["kernel_source_sha256"]:
        contract.fail(contract.Phase.PREPARE, contract.Fatal.PREPARE_FAILURE)
    source_dtype = ("*bf16", "*fp32")[record["source_dtype"]]
    out_dtype = ("*bf16", "*fp32")[record["output_dtype"]]
    variant = record["variant"]
    constants = {}
    compiler = dict(options)
    config_id = record["config_id"]
    if op == 0:
        constants.update(
            BLOCK_SIZE_M=values["BLOCK_M"],
            BLOCK_SIZE_N=values["BLOCK_N"],
            NUM_ITER=values["NUM_ITER"],
            NUM_STAGES=values["NUM_STAGES"],
            MXFP4_QUANT_BLOCK_SIZE=32,
            EVEN_M_N=False,
            SCALING_MODE="EVEN",
            USE_SR=variant == 1,
            CONTRACT_V1=True,
            CONFIG_ID=config_id,
            num_warps=values["num_warps"],
        )
        compiler["num_stages"] = values["NUM_STAGES"]
        fields = (
            "BLOCK_M",
            "BLOCK_N",
            "NUM_ITER",
            "NUM_STAGES",
            "num_warps",
            "waves_per_eu",
        )
        pointers = {
            "x_ptr": source_dtype,
            "x_fp4_ptr": "*u8",
            "bs_ptr": "*u8",
            "status_ptr": "*i32",
        }
    elif op == 1:
        constants.update(BLOCK_SIZE=32, CONTRACT_V1=True, CONFIG_ID=config_id)
        fields = ("num_warps", "num_stages", "waves_per_eu")
        pointers = {
            "x_ptr": source_dtype,
            "x_fp4_ptr": "*u8",
            "bs_ptr": "*u8",
            "status_ptr": "*i32",
        }
    elif op == 2:
        constants.update(
            BLOCK_SIZE=32,
            HADAMARD_SIZE=16,
            USE_SR_ROW=variant in (3, 5),
            USE_SR_TRANSPOSED=variant in (3, 6),
            CONFIG_ID=config_id,
        )
        fields = ("num_warps", "num_stages", "waves_per_eu")
        pointers = {
            "x_ptr": source_dtype,
            "row_ptr": "*u8",
            "row_scale_ptr": "*u8",
            "transposed_ptr": "*u8",
            "transposed_scale_ptr": "*u8",
            "status_ptr": "*i32",
        }
    elif op == 3:
        constants.update(
            BLOCK_M=values["BLOCK_M"], BLOCK_N_PACKED=values["BLOCK_N_PACKED"]
        )
        fields = (
            "BLOCK_M",
            "BLOCK_N_PACKED",
            "num_warps",
            "num_stages",
            "waves_per_eu",
            "reserved_zero",
        )
        pointers = {"input_ptr": "*u8", "output_ptr": "*u8"}
    elif op == 4:
        fields = (
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "cache_modifier",
            "NUM_KSPLIT",
        )
        constants.update({key: values[key] for key in fields})
        constants["cache_modifier"] = ("", ".cg", ".ca")[values["cache_modifier"]]
        constants.update(
            SPLITK_BLOCK_SIZE=0,
            EVEN_K=False,
            CONTRACT_V1=True,
            B_SCALE_ROW_GROUP=record["b_scale_row_group"],
            OUTPUT_MODE=record["output_mode"],
            CONFIG_ID=config_id,
        )
        pointers = {
            "a_ptr": "*u8",
            "b_ptr": "*u8",
            "c_ptr": out_dtype,
            "a_scales_ptr": "*u8",
            "b_scales_ptr": "*u8",
            "status_ptr": "*i32",
        }
        compiler["matrix_instr_nonkdim"] = values["matrix_instr_nonkdim"]
    else:
        fields = ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "waves_per_eu")
        constants.update(
            BLOCK_SIZE_M=values["BLOCK_M"],
            BLOCK_SIZE_N=values["BLOCK_N"],
            ACTUAL_KSPLIT=0,
            MAX_KSPLIT=1,
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="mxfp4_zero_output",
        )
        pointers = {
            "c_in_ptr": out_dtype,
            "c_out_ptr": out_dtype,
            "bias_ptr": out_dtype,
        }
    if set(values) != set(fields):
        contract.fail(contract.Phase.PREPARE, contract.Fatal.PREPARE_FAILURE)
    for name, value in values.items():
        contract.integer(value, phase=contract.Phase.PREPARE)
        if (name.startswith("BLOCK_") or name == "num_warps") and (
            value == 0 or value & (value - 1)
        ):
            contract.fail(contract.Phase.PREPARE, contract.Fatal.PREPARE_FAILURE)
        if (
            name in ("num_stages", "NUM_STAGES", "NUM_ITER", "GROUP_SIZE_M")
            and value == 0
        ):
            contract.fail(contract.Phase.PREPARE, contract.Fatal.PREPARE_FAILURE)
    if op == 0 and values["BLOCK_N"] % 32:
        contract.fail(contract.Phase.PREPARE, 106)
    if op == 3 and (values["BLOCK_M"] % 2 or values["reserved_zero"] != 0):
        contract.fail(contract.Phase.PREPARE, 106)
    if op == 4 and (
        values["NUM_KSPLIT"] != 1
        or values["BLOCK_SIZE_K"] % 32
        or values["cache_modifier"] > 2
    ):
        contract.fail(contract.Phase.PREPARE, 106)
    if op == 5 and (4096 % values["BLOCK_M"] or 2048 % values["BLOCK_N"]):
        contract.fail(contract.Phase.PREPARE, 106)
    compiler["num_warps"] = values["num_warps"]
    compiler["waves_per_eu"] = values["waves_per_eu"]
    if "num_stages" in values:
        compiler["num_stages"] = values["num_stages"]
    signature = {}
    for name in body.arg_names:
        if name in constants:
            continue
        if name in pointers:
            signature[name] = pointers[name]
        elif name.startswith("stride_"):
            signature[name] = "i64"
        elif "philox_" in name or name in ("row_seed", "row_base", "t_seed", "t_base"):
            signature[name] = "u64"
        else:
            signature[name] = "i32"
    compiled = triton.compile(ASTSource(body, signature, constants), options=compiler)
    metadata = compiled.metadata
    for field in ("global_scratch_size", "profile_scratch_size"):
        if getattr(metadata, field, None) != 0:
            contract.fail(
                contract.Phase.PREPARE, contract.Fatal.PREPARE_RESOURCE_UNSUPPORTED
            )
    compiled._init_handles()
    hsaco = compiled.asm.get("hsaco")
    if not isinstance(hsaco, bytes):
        contract.fail(
            contract.Phase.PREPARE, contract.Fatal.PREPARE_RESOURCE_UNSUPPORTED
        )
    code_digest = hashlib.sha256(hsaco).hexdigest()
    config = (config_id,) + tuple(values[name] for name in fields)
    key = _record_key(
        op,
        record["pass_id"],
        variant,
        record["source_dtype"],
        record["output_dtype"],
        record["b_scale_row_group"],
        record["linear_n"],
        record["linear_k"],
        record["output_mode"],
    )
    return contract.ReadyRecordV1(
        config, key, artifact_digest, code_digest, compiled, device
    )


@contract.prepare_errors
def mxfp4_dense_prepare_kernel_v1(static_spec, config_artifact):
    """Compile finite code-only records before bind; never consume source data or RNG.

    This is a compute-node operation. Missing resource metadata fails closed.
    """
    cap = mxfp4_dense_query_static_v1(static_spec)
    if cap.capability_status:
        contract.fail(contract.Phase.PREPARE, contract.Fatal.PREPARE_FAILURE)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if (
        props.gcnArchName.split(":")[0] != "gfx950"
        or props.multi_processor_count != static_spec[3]
    ):
        contract.fail(contract.Phase.PREPARE, contract.Fatal.PREPARE_FAILURE)
    from aiter.ops.triton.quant.dual_layout_mxfp4 import (
        dual_layout_quant_mxfp4_out_v1,  # noqa: F401
    )
    from aiter.ops.triton.quant.fast_transpose import (
        transpose_packed_fp4_out_v1,  # noqa: F401
    )
    from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant_out_v1  # noqa: F401

    if type(config_artifact) is not str:
        contract.fail(contract.Phase.PREPARE, 106)
    root = resolve_config_dir("gemm", "GEMM-AFP4WFP4", backend="triton")
    expected_path = os.path.join(root, "GEMM-AFP4WFP4-CONTRACT-MANIFEST-V1.json")
    if os.path.abspath(config_artifact) != os.path.abspath(expected_path):
        contract.fail(contract.Phase.PREPARE, 106)
    artifact = load_config_json(expected_path)
    fixed = {
        "schema_id": "aiter.mxfp4_dense_config.v1",
        "abi_major": 1,
        "abi_minor": 0,
        "arch": "gfx950",
        "kernel_family": 1,
        "selector_id": 1,
        "physical_layouts": [1, 2],
        "prepared_layout": 0,
        "rng_mapping": 1,
        "allowed_linear_shapes": [[4096, 4096], [4096, 2048]],
        "output_modes": [0, 1],
        "production_w_dtype": 1,
        "split_k": 1,
        "capture_safe": False,
        "max_relative_span_bytes": contract.SPAN_MAX,
        "compile_options": {"allow_flush_denorm": False, "enable_fp_fusion": False},
        "runtime_axes": ["M", "P", "strides"],
    }
    if set(artifact) != set(fixed) | {"records"}:
        contract.fail(contract.Phase.PREPARE, 106)
    for name, value in fixed.items():
        if json.dumps(artifact[name], sort_keys=True) != json.dumps(
            value, sort_keys=True
        ):
            contract.fail(contract.Phase.PREPARE, 106)
    digest = hashlib.sha256(
        json.dumps(
            artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    records = artifact["records"]
    if type(records) is not list or not records:
        contract.fail(contract.Phase.PREPARE, 106)
    required = {
        "config_id",
        "op_id",
        "pass_id",
        "variant",
        "source_dtype",
        "output_dtype",
        "b_scale_row_group",
        "linear_n",
        "linear_k",
        "output_mode",
        "profile_ref",
        "profile_sha256",
        "kernel_symbol",
        "kernel_source_sha256",
        "workspace_bytes",
        "workspace_alignment",
    }
    symbols = (
        "aiter.ops.triton._triton_kernels.quant.quant._dynamic_mxfp4_quant_kernel",
        "aiter.ops.triton._triton_kernels.quant.quant._dynamic_mxfp4_quant_blockscale_kernel",
        "aiter.ops.triton._triton_kernels.quant.dual_layout_mxfp4._dual_layout_quant_mxfp4_kernel",
        "aiter.ops.triton._triton_kernels.quant.fast_transpose._transpose_packed_fp4_kernel",
        "aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4._gemm_afp4wfp4_kernel",
        "aiter.ops.triton._triton_kernels.common.splitk_reduce._gemm_splitk_reduce_kernel",
    )
    compiled_records = []
    previous_id = 0
    keys = set()
    for record in records:
        if type(record) is not dict or set(record) != required:
            contract.fail(contract.Phase.PREPARE, 106)
        for name in required - {
            "profile_ref",
            "profile_sha256",
            "kernel_symbol",
            "kernel_source_sha256",
        }:
            contract.integer(
                record[name],
                -1 if name == "pass_id" else 0,
                phase=contract.Phase.PREPARE,
            )
        op = record["op_id"]
        if (
            op > 5
            or record["kernel_symbol"] != symbols[op]
            or record["config_id"] <= previous_id
        ):
            contract.fail(contract.Phase.PREPARE, 106)
        previous_id = record["config_id"]
        if record["source_dtype"] not in (0, 1) or record["output_dtype"] not in (0, 1):
            contract.fail(contract.Phase.PREPARE, 106)
        if record["b_scale_row_group"] not in (1, 32) or record["output_mode"] not in (
            0,
            1,
        ):
            contract.fail(contract.Phase.PREPARE, 106)
        if op == 4:
            if (
                record["pass_id"] not in (0, 1, 2)
                or record["variant"] != 0
                or record["source_dtype"] != 0
            ):
                contract.fail(contract.Phase.PREPARE, 106)
            if (record["linear_n"], record["linear_k"]) not in (
                (4096, 4096),
                (4096, 2048),
            ):
                contract.fail(contract.Phase.PREPARE, 106)
            if record["pass_id"] in (0, 1) and (
                record["output_dtype"],
                record["output_mode"],
            ) != (0, 0):
                contract.fail(contract.Phase.PREPARE, 106)
            if record["pass_id"] == 2 and (
                record["b_scale_row_group"] != 1
                or (record["output_mode"] == 1 and record["output_dtype"] != 1)
            ):
                contract.fail(contract.Phase.PREPARE, 106)
        else:
            if (
                record["pass_id"],
                record["linear_n"],
                record["linear_k"],
                record["output_mode"],
                record["b_scale_row_group"],
            ) != (-1, 0, 0, 0, 1):
                contract.fail(contract.Phase.PREPARE, 106)
            variants = ((0, 1), (0,), (2, 3, 5, 6), (0,), (), (4,))
            if record["variant"] not in variants[op]:
                contract.fail(contract.Phase.PREPARE, 106)
            if (op != 5 and record["output_dtype"] != 0) or (
                op in (3, 5) and record["source_dtype"] != 0
            ):
                contract.fail(contract.Phase.PREPARE, 106)
        if (record["workspace_bytes"], record["workspace_alignment"]) != (0, 1):
            contract.fail(contract.Phase.PREPARE, 107)
        ref = record["profile_ref"]
        if (
            type(ref) is not dict
            or set(ref) != {"family", "config_name", "suffix", "bucket"}
            or ref["bucket"] != "any"
        ):
            contract.fail(contract.Phase.PREPARE, 106)
        paths = (
            ("quant", "MXFP4-CONTRACT", "DEFAULT"),
            ("quant", "MXFP4-CONTRACT", "DEFAULT"),
            ("fusions", "DUAL-LAYOUT-MXFP4", "DEFAULT"),
            ("quant", "MXFP4-CONTRACT", "DEFAULT"),
            ("gemm", "GEMM-AFP4WFP4", "CONTRACT-V1"),
            ("gemm", "MXFP4-ZERO-OUTPUT", "DEFAULT"),
        )
        if (ref["family"], ref["config_name"], ref["suffix"]) != paths[op]:
            contract.fail(contract.Phase.PREPARE, 106)
        directory = resolve_config_dir(
            ref["family"], ref["config_name"], backend="triton"
        )
        filename = (
            "DEFAULT.json"
            if ref["suffix"] == "DEFAULT"
            else ref["config_name"] + "-" + ref["suffix"] + ".json"
        )
        profile_path = os.path.join(directory, filename)
        with open(profile_path, "rb") as stream:
            if hashlib.sha256(stream.read()).hexdigest() != record["profile_sha256"]:
                contract.fail(contract.Phase.PREPARE, 106)
        profile = load_config_json(profile_path)
        if op == 4:
            values, is_tuned = get_gemm_config(
                "GEMM-AFP4WFP4", 0, specialized_filename="CONTRACT-V1", backend="triton"
            )
            if is_tuned or values != profile["any"]:
                contract.fail(contract.Phase.PREPARE, 106)
            values = dict(values)
            values["cache_modifier"] = {None: 0, "": 0, ".cg": 1, ".ca": 2}[
                values["cache_modifier"]
            ]
        elif op == 5:
            values = dict(profile["any"]["contract_v1"])
        else:
            values = dict(profile[symbols[op].rsplit(".", 1)[1]]["contract_v1"])
        ready = _compile_contract_record(
            record, values, artifact["compile_options"], device, digest
        )
        if ready.semantic_key in keys:
            contract.fail(contract.Phase.PREPARE, 106)
        keys.add(ready.semantic_key)
        compiled_records.append(ready)
    table = contract.register_ready(compiled_records)
    return PreparedDenseV1(static_spec, cap, table, device, digest)
