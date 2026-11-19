# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compact grouped MXFP4 routing metadata tests."""

import torch

from aiter.ops.triton.utils.mxfp4_wgrad import (
    build_mxfp4_wgrad_metadata,
    validate_mxfp4_wgrad_metadata,
)


def test_compact_blocks_encode_direct_tile_routes():
    counts = (0, 1, 31, 33, 513)
    metadata = build_mxfp4_wgrad_metadata(counts, "cpu", 256)

    assert metadata.compact_blocks.tolist() == [
        [0, 1, 0],
        [1, 32, 256],
        [32, 65, 512],
        [65, 578, 768],
        [321, 578, 1024],
        [577, 578, 1280],
    ]
    assert metadata.compact_blocks.dtype == torch.int32
    assert metadata.compact_blocks.is_contiguous()
    assert metadata.compact_blocks.untyped_storage().data_ptr() == metadata.storage_base
    validate_mxfp4_wgrad_metadata(metadata, counts, torch.device("cpu"), 32)
