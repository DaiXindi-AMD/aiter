# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The public matched control preserves R75 FP32 main_grad semantics."""

import pytest
import torch

from aiter.ops.triton.gemm.grouped.bf16_wgrad import grouped_bf16_wgrad
from op_tests.triton_tests.utils.mxfp4_wgrad_reference import bf16_wgrad_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("transpose", [False, True])
def test_bf16_wgrad_two_mb_poison_and_zero(transpose):
    n, m = 65, 97
    shape = (2, n, m) if transpose else (2, m, n)
    out = torch.full(shape, float("nan"), dtype=torch.float32, device="cuda")
    expected = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for mb, counts in enumerate(((0, 33), (17, 0))):
        x = torch.randn((sum(counts), n), dtype=torch.bfloat16, device="cuda") * 0.05
        dy = torch.randn((sum(counts), m), dtype=torch.bfloat16, device="cuda") * 0.08
        expected += bf16_wgrad_reference(x, dy, counts, transpose_output=transpose)
        grouped_bf16_wgrad(
            x, dy, counts, out=out, accumulate=bool(mb), transpose_output=transpose
        )
        torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-5)
    out.zero_()
    grouped_bf16_wgrad(
        x[:0], dy[:0], (0, 0), out=out, accumulate=True, transpose_output=transpose
    )
    assert torch.count_nonzero(out) == 0
