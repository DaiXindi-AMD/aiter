# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _silu_exp2(x):
    """Compute SiLU with the exp2 formulation shared by Triton kernels."""
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))
