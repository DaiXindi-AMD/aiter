// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Grouped MXFP4 Wgrad on gfx950: out[e][m][n] (+)= sum_k L[m][k] * R[n][k] over the padded token
// range of expert e, with E2M1 colwise packs (even k in the low nibble) and E8M0 1x32 scales along k.
// Operands stream global->LDS by buffer_load ... lds. Each row chunk is XOR-swizzled in 16 B slots so the
// 16 lanes of one ds_read_b128 pass hit distinct bank groups (64 banks x 4 B).
#include "opus/hip_minimal.hpp"
#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#endif
#include "mxfp4_moe_fast.h"
#include <stdint.h>

#define WG_THREADS 256

namespace mxw {
constexpr int BM = 128, BN = 128; // workgroup tile (L rows x R rows); four waves of 64 x 64
constexpr int KC = 256;           // tokens per K chunk; the padding contract must be a multiple
constexpr int NS = 2;             // LDS stages; NS - 1 chunks stay in flight
constexpr int OCC = 2;            // workgroups per CU

constexpr int ROWB    = KC / 2;      // payload bytes per row per chunk
constexpr int SLOTS   = ROWB / 16;   // 16 B slots per row chunk
constexpr int GROUP   = 16 / SLOTS;  // rows sharing one 256 B bank cycle
constexpr int RPI     = 1024 / ROWB; // rows per wave-wide 16 B DMA
constexpr int PAY_DMA = 32 / RPI;    // DMAs per operand per wave (32 rows each)
constexpr int SB      = KC / 32;     // scale bytes per row per chunk
constexpr int STEPS   = KC / 64;     // MFMA k-steps per chunk (each lane half owns STEPS blocks)
constexpr int PAY_B   = BM * ROWB;
constexpr int SC_B    = BM * SB;
constexpr int STAGE_B = 2 * PAY_B + 2 * SC_B;
constexpr int LDS_B   = NS * STAGE_B;
constexpr int DMA_PER_CHUNK = 2 * PAY_DMA + 2;
static_assert(SB == 8, "the scale DMA below assumes 8 B scale rows per chunk");
__host__ __device__ constexpr int key(int row) { return (row / GROUP) & (SLOTS - 1); }
} // namespace mxw

#ifdef __HIP_DEVICE_COMPILE__
namespace mxw {
using i32x4  = int __attribute__((ext_vector_type(4)));
using i32x8  = int __attribute__((ext_vector_type(8)));
using f32x4  = float __attribute__((ext_vector_type(4)));
using mma_t  = opus::mfma_f32_32x32x64_fp4_fp4;
using acc_t  = typename mma_t::vtype_c;
using lds_ptr = __attribute__((address_space(3))) char*;

template <int N>
__device__ __forceinline__ void wait_vmcnt()
{
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}
} // namespace mxw
#endif

template <bool ACC>
__global__ __launch_bounds__(WG_THREADS, mxw::OCC) void opus_mxfp4_wgrad_kernel(const uint8_t* __restrict__ lp,
                                                                                const uint8_t* __restrict__ ls,
                                                                                const uint8_t* __restrict__ rp,
                                                                                const uint8_t* __restrict__ rs,
                                                                                float* __restrict__ out,
                                                                                const int* __restrict__ poff,
                                                                                int pstride,
                                                                                int sstride,
                                                                                int M,
                                                                                int N,
                                                                                int E,
                                                                                int xm)
{
#ifdef __HIP_DEVICE_COMPILE__
    using namespace mxw;
    __shared__ __attribute__((aligned(16))) char smem[LDS_B];
    const int tiles_m = M / BM, tiles_n = N / BN, tiles = tiles_m * tiles_n;
    const int bid = __builtin_amdgcn_workgroup_id_x();
    const int e   = bid / tiles;
    const int t   = bid - e * tiles;
    int tm, tn;
    if(xm > 0)
    {
        // Workgroups round-robin over 8 XCDs: XCD x = t % 8 owns an (xm x 8/xm) block of this expert's tiles,
        // so its L2 only fetches 1/xm of the L rows and xm/8 of the R rows.
        const int x = t & 7, l = t >> 3, xn = 8 / xm;
        const int gm = tiles_m / xm, gn = tiles_n / xn;
        tm = (x / xn) * gm + l / gn;
        tn = (x % xn) * gn + l % gn;
    }
    else
    {
        tm = t / tiles_n;
        tn = t - tm * tiles_n;
    }
    const int k0 = __builtin_amdgcn_readfirstlane(poff[e]);
    const int nk = (__builtin_amdgcn_readfirstlane(poff[e + 1]) - k0) / KC;
    if(ACC && nk == 0)
        return;

    const int tid  = __builtin_amdgcn_workitem_id_x();
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 6), lane = tid & 63;
    const int r32 = lane & 31, h = lane >> 5;
    const int wm = wave >> 1, wn = wave & 1;

    acc_t acc[2][2];
#pragma unroll
    for(int i = 0; i < 2; ++i)
#pragma unroll
        for(int j = 0; j < 2; ++j)
            acc[i][j] = acc_t{0};

    if(nk > 0)
    {
        // Wave w copies payload and scale rows [32w, 32w + 32) of both operands.
        const int crow = wave * 32;
        opus::gmem<uint8_t> lpay(lp + size_t(tm * BM + crow) * pstride + k0 / 2);
        opus::gmem<uint8_t> rpay(rp + size_t(tn * BN + crow) * pstride + k0 / 2);
        const int drow = lane / SLOTS, dslot = lane % SLOTS;
        int pv[PAY_DMA];
#pragma unroll
        for(int i = 0; i < PAY_DMA; ++i)
        {
            const int row = i * RPI + drow;
            pv[i]         = row * pstride + ((dslot ^ key(row)) << 4);
        }
        opus::gmem<uint8_t> lsc(ls + size_t(tm * BM + crow) * sstride + k0 / 32);
        opus::gmem<uint8_t> rsc(rs + size_t(tn * BN + crow) * sstride + k0 / 32);
        const int sv       = (lane >> 1) * sstride + (lane & 1) * 4;
        const lds_ptr base = (lds_ptr)smem;

        auto issue = [&](int c) {
            const lds_ptr st = base + (c % NS) * STAGE_B;
#pragma unroll
            for(int i = 0; i < PAY_DMA; ++i)
            {
                lpay._async_load<16>(st + (crow + i * RPI) * ROWB, pv[i], c * ROWB);
                rpay._async_load<16>(st + PAY_B + (crow + i * RPI) * ROWB, pv[i], c * ROWB);
            }
            lsc._async_load<4>(st + 2 * PAY_B + crow * 8, sv, c * 8);
            rsc._async_load<4>(st + 2 * PAY_B + SC_B + crow * 8, sv, c * 8);
        };

        // Fragment rows: L block j -> tile row 64wm + 32j + r32, R block i -> 64wn + 32i + r32.
        int lrow[2], rrow[2];
#pragma unroll
        for(int b = 0; b < 2; ++b)
        {
            lrow[b] = wm * 64 + 32 * b + r32;
            rrow[b] = wn * 64 + 32 * b + r32;
        }
        const int skey  = key(r32);
        const int s_off = h * 4;

#pragma unroll
        for(int p = 0; p < NS - 1; ++p)
            if(p < nk)
                issue(p);
        for(int c = 0; c < nk; ++c)
        {
            // In-order vmcnt: chunk c has landed once at most the younger in-flight chunks remain.
            if(c + NS - 2 < nk)
                wait_vmcnt<(NS - 2) * DMA_PER_CHUNK>();
            else
                wait_vmcnt<0>();
            __builtin_amdgcn_s_barrier();
            asm volatile("" ::: "memory");
            if(c + NS - 1 < nk)
                issue(c + NS - 1);
            const lds_ptr st = base + (c % NS) * STAGE_B;
            i32x4 lf[2][STEPS], rf[2][STEPS];
            int lsv[2], rsv[2];
#pragma unroll
            for(int b = 0; b < 2; ++b)
            {
#pragma unroll
                for(int s = 0; s < STEPS; ++s)
                {
                    const int slot = (STEPS * h + s) ^ skey;
                    lf[b][s] = *(__attribute__((address_space(3))) i32x4*)(st + lrow[b] * ROWB + slot * 16);
                    rf[b][s] =
                        *(__attribute__((address_space(3))) i32x4*)(st + PAY_B + rrow[b] * ROWB + slot * 16);
                }
                lsv[b] = *(__attribute__((address_space(3))) int*)(st + 2 * PAY_B + lrow[b] * SB + s_off);
                rsv[b] = *(__attribute__((address_space(3))) int*)(st + 2 * PAY_B + SC_B + rrow[b] * SB + s_off);
            }
            mma_t mma;
            opus::static_for<STEPS>([&](auto s) {
                constexpr int S = decltype(s)::value;
#pragma unroll
                for(int i = 0; i < 2; ++i)
#pragma unroll
                    for(int j = 0; j < 2; ++j)
                    {
                        const i32x4 a = rf[i][S], b = lf[j][S];
                        acc[i][j]     = mma(i32x8{a[0], a[1], a[2], a[3], 0, 0, 0, 0},
                                        i32x8{b[0], b[1], b[2], b[3], 0, 0, 0, 0},
                                        acc[i][j],
                                        rsv[i],
                                        lsv[j],
                                        opus::number<S>{},
                                        opus::number<S>{});
                    }
            });
        }
    }

    // Lane owns D[n = 8g + 4h + q][m = r32] per 32x32 block: four consecutive n per (g, m).
    const int m0 = tm * BM + wm * 64, n0 = tn * BN + wn * 64;
    opus::gmem<float> o(out + size_t(e) * M * N + size_t(m0) * N + n0);
    int ov[2][2][4];
#pragma unroll
    for(int i = 0; i < 2; ++i)
#pragma unroll
        for(int j = 0; j < 2; ++j)
#pragma unroll
            for(int g = 0; g < 4; ++g)
                ov[i][j][g] = ((32 * j + r32) * N + 32 * i + 8 * g + 4 * h) * 4;

    if constexpr(ACC)
    {
        f32x4 old[2][2][4];
#pragma unroll
        for(int i = 0; i < 2; ++i)
#pragma unroll
            for(int j = 0; j < 2; ++j)
#pragma unroll
                for(int g = 0; g < 4; ++g)
                    old[i][j][g] = __builtin_bit_cast(f32x4, o._load<4>(ov[i][j][g]));
#pragma unroll
        for(int i = 0; i < 2; ++i)
#pragma unroll
            for(int j = 0; j < 2; ++j)
#pragma unroll
                for(int g = 0; g < 4; ++g)
#pragma unroll
                    for(int q = 0; q < 4; ++q)
                        acc[i][j][4 * g + q] += old[i][j][g][q];
    }
#pragma unroll
    for(int i = 0; i < 2; ++i)
#pragma unroll
        for(int j = 0; j < 2; ++j)
#pragma unroll
            for(int g = 0; g < 4; ++g)
            {
                f32x4 v{acc[i][j][4 * g], acc[i][j][4 * g + 1], acc[i][j][4 * g + 2], acc[i][j][4 * g + 3]};
                o._store<4>(v, ov[i][j][g]);
            }
#endif
}

template __global__ void opus_mxfp4_wgrad_kernel<false>(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, float*, const int*, int, int, int, int, int, int);
template __global__ void opus_mxfp4_wgrad_kernel<true>(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, float*, const int*, int, int, int, int, int, int);

#ifndef __HIP_DEVICE_COMPILE__
// XCD split (xm row groups x 8/xm column groups) minimising total L2 operand fetch, 0 if tiles don't divide.
static int pick_xcd_split(int M, int N)
{
    const int tm = M / mxw::BM, tn = N / mxw::BN;
    int best       = 0;
    long best_cost = 0;
    for(int xm = 1; xm <= 8; xm *= 2)
    {
        const int xn = 8 / xm;
        if(tm % xm || tn % xn)
            continue;
        const long cost = long(xn) * M + long(xm) * N;
        if(!best || cost < best_cost)
            best = xm, best_cost = cost;
    }
    return best;
}

namespace aiter {

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
                            void* stream)
{
    if(M % mxw::BM || N % mxw::BN || pstride % 16 || sstride % 4)
        return -1;
    const int tiles = (M / mxw::BM) * (N / mxw::BN);
    if(E <= 0 || tiles == 0)
        return 0;
    int xm = pick_xcd_split(M, N);
    if(xm && tiles % 8)
        xm = 0;
    dim3 grid(E * tiles), block(WG_THREADS);
    auto s = (hipStream_t)stream;
#define MXW_LAUNCH(ACC)                                                                    \
    hipLaunchKernelGGL((opus_mxfp4_wgrad_kernel<ACC>), grid, block, 0, s, (const uint8_t*)lp, \
                       (const uint8_t*)ls, (const uint8_t*)rp, (const uint8_t*)rs, (float*)out, \
                       (const int*)poff, pstride, sstride, M, N, E, xm)
    if(accumulate)
        MXW_LAUNCH(true);
    else
        MXW_LAUNCH(false);
#undef MXW_LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

} // namespace aiter
#endif
