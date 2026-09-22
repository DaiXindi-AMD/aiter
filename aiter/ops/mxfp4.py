# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports for :mod:`aiter.ops.triton.quant.mxfp4`.

New code should import MXFP4 helpers from the categorized Triton quant module.
This module remains as a stable compatibility surface for existing callers.
"""

from aiter.ops.triton.quant.mxfp4 import *  # noqa: F401,F403
from aiter.ops.triton.quant.mxfp4 import __all__
