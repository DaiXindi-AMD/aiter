# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import copy
import functools
import json
import os

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

_STANDARD_M_BOUNDS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


@functools.lru_cache(maxsize=128)
def _load_quant_config(config_name: str, arch: str) -> dict:
    path = os.path.join(
        AITER_TRITON_CONFIGS_PATH,
        "quant",
        f"{arch}-{config_name}.json",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required quantization config does not exist: {path}")
    with open(path, "r") as config_file:
        return json.load(config_file)


def get_quant_config(config_name: str, M: int) -> dict:
    """Load an architecture-specific quantization config for the M bucket."""
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")

    config_table = _load_quant_config(config_name, arch_info.get_arch())
    for bound in _STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in config_table:
            return copy.deepcopy(config_table[key])

    for bound in reversed(_STANDARD_M_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in config_table:
            return copy.deepcopy(config_table[key])

    if "any" in config_table:
        return copy.deepcopy(config_table["any"])
    raise KeyError(f"No matching M bucket for {M} in quant config {config_name}")
