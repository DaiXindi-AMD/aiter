// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host launchers for the fused pre-routed MXFP4 MoE chain. Both return 0 on success and a
// negative code for an unsupported shape (-1, -2), producer (-3) or launch failure (-4).
#pragma once

namespace aiter {

// routes: compact [nblocks, 3] int32 (real start, real end, padded start) per 256 padded tokens.
// Rowwise outputs are [rows, F/2] / [rows, F/32]; colwise outputs are [F, P/2] / [F, P/32].
int mxfp4_moe_fused_quant_launch(int prod,
                                 const void* s0,
                                 const void* s1,
                                 const void* score,
                                 void* rpay,
                                 void* rsc,
                                 void* cpay,
                                 void* csc,
                                 const void* routes,
                                 int F,
                                 int P,
                                 int I,
                                 int nblocks,
                                 void* stream);

// out[e] (+)= L_e @ R_e^T in FP32 [E, M, N] over the padded token range poff[e]..poff[e+1].
int opus_mxfp4_wgrad_launch(const void* lp,
                            const void* ls,
                            const void* rp,
                            const void* rs,
                            void* out,
                            const void* poff,
                            int pstride,
                            int sstride,
                            int M,
                            int N,
                            int E,
                            int accumulate,
                            void* stream);

} // namespace aiter
