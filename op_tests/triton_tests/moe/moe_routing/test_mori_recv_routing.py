# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.moe.mori_recv_routing import mori_recv_local_routing


def _reference(indices, num_valid, local_start, num_local):
    """Megatron MoriDispatch.forward receive-side post-processing."""
    rows = torch.arange(indices.shape[0], device=indices.device).unsqueeze(1)
    valid_rows = rows < num_valid.flatten()[0]
    is_local = (indices >= local_start) & (indices < local_start + num_local)
    local = (indices - local_start).to(torch.int64)
    local = torch.where(is_local & valid_rows, local, torch.full_like(local, -1))
    shifted = (local + 1).reshape(-1)
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=indices.device)
    counts.scatter_add_(0, shifted, torch.ones_like(shifted))
    return local, counts[1:]


@pytest.mark.parametrize("rows,topk", [(16384, 8), (1000, 8), (1, 1), (3, 5), (4097, 2)])
@pytest.mark.parametrize("num_local,rank", [(32, 0), (32, 3), (8, 1), (1, 0), (5, 2)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_matches_megatron_reference(rows, topk, num_local, rank, dtype):
    torch.manual_seed(rows * 31 + topk + num_local + rank)
    num_experts = num_local * 4
    indices = torch.randint(-1, num_experts, (rows, topk), dtype=dtype, device="cuda")
    for valid in (0, rows // 2, rows):
        num_valid = torch.tensor([valid], dtype=torch.int32, device="cuda")
        local_start = rank * num_local
        expected = _reference(indices, num_valid, local_start, num_local)
        actual = mori_recv_local_routing(indices, num_valid, local_start, num_local)
        for got, want in zip(actual, expected):
            assert got.dtype == torch.int64 and got.is_contiguous()
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_empty_rows():
    indices = torch.empty((0, 8), dtype=torch.int32, device="cuda")
    num_valid = torch.zeros(1, dtype=torch.int32, device="cuda")
    local, counts = mori_recv_local_routing(indices, num_valid, 0, 4)
    assert local.shape == (0, 8) and counts.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"indices": torch.zeros((4, 2), dtype=torch.float32)},
        {"indices": torch.zeros((4, 2, 1), dtype=torch.int32)},
        {"indices": torch.zeros((2, 4), dtype=torch.int32).t()},
        {"num_valid": torch.zeros(0, dtype=torch.int32)},
        {"local_start": -1},
        {"num_local_experts": 0},
    ],
)
def test_rejects_invalid_arguments(kwargs):
    args = {
        "indices": torch.zeros((4, 2), dtype=torch.int32),
        "num_valid": torch.ones(1, dtype=torch.int32),
        "local_start": 0,
        "num_local_experts": 4,
    }
    args.update(kwargs)
    for key in ("indices", "num_valid"):
        args[key] = args[key].cuda()
    with pytest.raises(ValueError):
        mori_recv_local_routing(**args)
