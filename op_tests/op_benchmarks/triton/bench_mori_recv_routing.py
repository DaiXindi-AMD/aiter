# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MORI receive routing vs the Megatron eager chain: GPU time and host submit time."""

import argparse
import json
import statistics
import time

import torch

from aiter.ops.triton.moe.mori_recv_routing import mori_recv_local_routing


def eager(indices, num_valid, local_start, num_local):
    rows = torch.arange(indices.shape[0], device=indices.device).unsqueeze(1)
    valid_rows = rows < num_valid.flatten()[0]
    is_local = (indices >= local_start) & (indices < local_start + num_local)
    local = (indices - local_start).to(torch.int64)
    local = torch.where(is_local & valid_rows, local, torch.full_like(local, -1))
    shifted = (local + 1).reshape(-1)
    counts = torch.zeros(num_local + 1, dtype=torch.long, device=indices.device)
    counts.scatter_add_(0, shifted, torch.ones_like(shifted))
    return local, counts[1:]


def measure(fn, calls=50, repeats=9):
    fn()
    torch.cuda.synchronize()
    host, gpu = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        host.append(1e6 * (t1 - t0) / calls)
        gpu.append(1e3 * start.elapsed_time(end) / calls)
    return {"host_us": round(statistics.median(host), 2), "gpu_us": round(statistics.median(gpu), 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-local", type=int, default=32)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--valid", type=int, default=12000)
    args = parser.parse_args()
    torch.manual_seed(0)
    indices = torch.randint(0, 4 * args.num_local, (args.rows, args.topk), dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([args.valid], dtype=torch.int32, device="cuda")
    start = args.rank * args.num_local
    exact = all(
        torch.equal(a, b)
        for a, b in zip(
            eager(indices, num_valid, start, args.num_local),
            mori_recv_local_routing(indices, num_valid, start, args.num_local),
        )
    )
    print(json.dumps({
        "shape": [args.rows, args.topk],
        "num_local": args.num_local,
        "exact": exact,
        "eager": measure(lambda: eager(indices, num_valid, start, args.num_local)),
        "fused": measure(lambda: mori_recv_local_routing(indices, num_valid, start, args.num_local)),
    }, indent=1))


if __name__ == "__main__":
    main()
