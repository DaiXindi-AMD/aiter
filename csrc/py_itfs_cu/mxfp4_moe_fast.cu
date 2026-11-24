// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// torch-free ctypes entry points of module_mxfp4_moe_fast. Device buffers arrive as raw
// addresses so the per-call host cost stays at one foreign call per kernel.
#include "aiter_hip_common.h"
#include "aiter_ctypes_error.h"
#include "mxfp4_moe_fast.h"
#include <hip/hip_runtime.h>
#include <stdint.h>

AITER_CTYPES_ERROR_DEF

#define MXFP4_MOE_PTR(v) reinterpret_cast<void*>(static_cast<uintptr_t>(v))

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp4_moe_fused_quant,
    (int64_t prod,
     int64_t s0,
     int64_t s1,
     int64_t score,
     int64_t rpay,
     int64_t rsc,
     int64_t cpay,
     int64_t csc,
     int64_t routes,
     int64_t F,
     int64_t P,
     int64_t I,
     int64_t nblocks,
     hipStream_t stream),
    (prod, s0, s1, score, rpay, rsc, cpay, csc, routes, F, P, I, nblocks, stream))
{
    const int rc = aiter::mxfp4_moe_fused_quant_launch(int(prod),
                                                       MXFP4_MOE_PTR(s0),
                                                       MXFP4_MOE_PTR(s1),
                                                       MXFP4_MOE_PTR(score),
                                                       MXFP4_MOE_PTR(rpay),
                                                       MXFP4_MOE_PTR(rsc),
                                                       MXFP4_MOE_PTR(cpay),
                                                       MXFP4_MOE_PTR(csc),
                                                       MXFP4_MOE_PTR(routes),
                                                       int(F),
                                                       int(P),
                                                       int(I),
                                                       int(nblocks),
                                                       stream);
    AITER_CHECK(rc == 0, "mxfp4_moe_fused_quant(prod=", prod, ", F=", F, ", I=", I, ") failed: ", rc);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mxfp4_wgrad,
    (int64_t lp,
     int64_t ls,
     int64_t rp,
     int64_t rs,
     int64_t out,
     int64_t poff,
     int64_t pstride,
     int64_t sstride,
     int64_t M,
     int64_t N,
     int64_t E,
     int64_t accumulate,
     hipStream_t stream),
    (lp, ls, rp, rs, out, poff, pstride, sstride, M, N, E, accumulate, stream))
{
    const int rc = aiter::opus_mxfp4_wgrad_launch(MXFP4_MOE_PTR(lp),
                                                  MXFP4_MOE_PTR(ls),
                                                  MXFP4_MOE_PTR(rp),
                                                  MXFP4_MOE_PTR(rs),
                                                  MXFP4_MOE_PTR(out),
                                                  MXFP4_MOE_PTR(poff),
                                                  int(pstride),
                                                  int(sstride),
                                                  int(M),
                                                  int(N),
                                                  int(E),
                                                  int(accumulate),
                                                  stream);
    AITER_CHECK(rc == 0, "opus_mxfp4_wgrad(M=", M, ", N=", N, ") failed: ", rc);
}
