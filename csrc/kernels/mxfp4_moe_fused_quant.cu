// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Grouped MXFP4 producer + dual quant for the pre-routed SwiGLU MoE (gfx950).
//
// One workgroup = 128 padded tokens x 128 output features; wave w owns the 32-token
// colwise block w and each lane holds 8 consecutive tokens x 8 consecutive features, so
// both amax reductions are 4-lane DPP reductions. Bit-exact with AITER's _mxfp4_quant_op
// ("even" E8M0 scale, RNE E2M1) and with the Triton producers it replaces.
//
// prod 0: x                                 -> rowwise + colwise
// prod 1: a = swiglu(h1)                    -> rowwise + colwise
// prod 2: dh = swiglu'(h1, bf16(uda*score)) -> rowwise + colwise
// prod 3: dout -> rowwise, dy = bf16(dout*score) -> colwise
#include "mxfp4_moe_fast.h"
#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

namespace {

__device__ __forceinline__ float bf2f(uint32_t v) { return __uint_as_float(v << 16); }

__device__ __forceinline__ uint32_t f2bf(float f)
{
    uint32_t u = __float_as_uint(f);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

__device__ __forceinline__ void unpack8(const uint4& r, float* v)
{
    const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        v[2 * i]     = __uint_as_float(w[i] << 16);
        v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

// Triton: sigmoid(x) = 1 / (1 + exp(-x)), exp(y) = exp2(y * log2e), IEEE division.
__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + __builtin_exp2f(-x * 1.44269504088896340736f));
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }

// _mxfp4_scale_from_amax: E8M0 = clamp(exponent(amax rounded at 2^-2 ulp) - 2, 0, 254).
__device__ __forceinline__ void mx_scale(float amax, uint32_t& e8, float& qs)
{
    const uint32_t u = (__float_as_uint(amax) + 0x200000u) & 0xFF800000u;
    int e            = int(u >> 23) - 2;
    e                = e < 0 ? 0 : (e > 254 ? 254 : e);
    e8               = uint32_t(e);
    qs               = __uint_as_float(uint32_t(254 - e) << 23);
}

__device__ __forceinline__ uint32_t cvt_fp4x2(uint32_t old, float lo, float hi, int byte)
{
    switch(byte)
    {
    case 0: return __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(old, lo, hi, 1.0f, 0);
    case 1: return __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(old, lo, hi, 1.0f, 1);
    case 2: return __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(old, lo, hi, 1.0f, 2);
    default: return __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(old, lo, hi, 1.0f, 3);
    }
}

__device__ __forceinline__ float qmax(float a, float b) { return a > b ? a : b; }

template <int CTRL>
__device__ __forceinline__ float dpp(float v)
{
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}

// max over the 4 lanes {i, i^1, i^2, i^3} of a quad
__device__ __forceinline__ float quad_max(float v)
{
    v = qmax(v, dpp<0xB1>(v));
    return qmax(v, dpp<0x4E>(v));
}

// max over the 4 lanes {i, i+4, i+8, i+12} (mod 16) of a DPP row
__device__ __forceinline__ float row4_max(float v)
{
    v = qmax(v, dpp<0x124>(v));
    return qmax(v, dpp<0x128>(v));
}

template <int PROD>
__global__ __launch_bounds__(256, 2) void mxfp4_moe_fused_quant_kernel(const uint16_t* __restrict__ s0,
                                                                        const uint16_t* __restrict__ s1,
                                                                        const float* __restrict__ score,
                                                                        uint8_t* __restrict__ rpay,
                                                                        uint8_t* __restrict__ rsc,
                                                                        uint8_t* __restrict__ cpay,
                                                                        uint8_t* __restrict__ csc,
                                                                        const int* __restrict__ routes,
                                                                        int F,
                                                                        int P,
                                                                        int I,
                                                                        int tiles_f)
{
    const int bid  = blockIdx.x;
    const int tt   = bid / tiles_f;
    const int tf   = bid - tt * tiles_f;
    const int pb   = tt >> 1;
    const int tok0 = routes[3 * pb] + 128 * (tt & 1);
    const int end  = routes[3 * pb + 1];
    const int ptk0 = routes[3 * pb + 2] + 128 * (tt & 1);

    // lane = 4*fg + rq: the 4 rq lanes of a column group form a DPP quad (colwise amax) and
    // the 4 fg lanes of a 32-feature block share one 16-lane DPP row (rowwise amax).
    const int lane = threadIdx.x & 63;
    const int w    = threadIdx.x >> 6;
    const int fg   = lane >> 2;
    const int rq   = lane & 3;

    constexpr int LDS_STRIDE = 17; // words per staged row/column (16 + 1 pad against bank conflicts)
    __shared__ uint32_t lcol[128 * LDS_STRIDE];
    __shared__ uint32_t lrow[128 * LDS_STRIDE];
    __shared__ uint8_t lcs[128 * 4];
    __shared__ uint8_t lrs[128 * 4];
    const int r0 = tok0 + 32 * w + 8 * rq;

    int src;
    if constexpr(PROD == 2)
        src = tf * 64 + 8 * (fg & 7);
    else
        src = tf * 128 + 8 * fg;

    float v[8][8];
    float sc[8];
    if constexpr(PROD == 2)
    {
        // Lanes fg and fg^8 share columns src..src+7; lane half hi computes both d_gate and
        // d_up for rows 4hi..4hi+3 (one sigmoid per element pair), then they swap halves so
        // the gate lane holds 8 rows of d_gate and the up lane 8 rows of d_up.
        const int hi = fg >> 3;
        uint4 rg[4], ru[4], rd[4];
#pragma unroll
        for(int jj = 0; jj < 4; ++jj)
        {
            const long row = r0 + 4 * hi + jj;
            if(row < end)
            {
                rg[jj] = *reinterpret_cast<const uint4*>(s0 + row * 2 * I + src);
                ru[jj] = *reinterpret_cast<const uint4*>(s0 + row * 2 * I + I + src);
                rd[jj] = *reinterpret_cast<const uint4*>(s1 + row * I + src);
                sc[jj] = score[row];
            }
        }
        uint32_t keep[4][4], send[4][4];
#pragma unroll
        for(int jj = 0; jj < 4; ++jj)
        {
            float dg[8], du[8];
            const long row = r0 + 4 * hi + jj;
            if(row < end)
            {
                float g[8], u[8], d[8];
                unpack8(rg[jj], g);
                unpack8(ru[jj], u);
                unpack8(rd[jj], d);
#pragma unroll
                for(int k = 0; k < 8; ++k)
                {
                    const float da = rbf(d[k] * sc[jj]);
                    const float s  = sigmoid(g[k]);
                    du[k]          = da * (g[k] * s);
                    dg[k]          = da * u[k] * (s * __builtin_fmaf(g[k], 1.0f - s, 1.0f));
                }
            }
            else
            {
#pragma unroll
                for(int k = 0; k < 8; ++k)
                    dg[k] = du[k] = 0.0f;
            }
#pragma unroll
            for(int m = 0; m < 4; ++m)
            {
                const uint32_t pg = f2bf(dg[2 * m]) | (f2bf(dg[2 * m + 1]) << 16);
                const uint32_t pu = f2bf(du[2 * m]) | (f2bf(du[2 * m + 1]) << 16);
                keep[jj][m]       = hi ? pu : pg;
                send[jj][m]       = hi ? pg : pu;
            }
        }
#pragma unroll
        for(int jj = 0; jj < 4; ++jj)
        {
#pragma unroll
            for(int m = 0; m < 4; ++m)
            {
                const uint32_t got = uint32_t(__shfl_xor(int(send[jj][m]), 32, 64));
                const uint32_t lo4 = hi ? got : keep[jj][m];
                const uint32_t hi4 = hi ? keep[jj][m] : got;
                v[jj][2 * m]         = __uint_as_float(lo4 << 16);
                v[jj][2 * m + 1]     = __uint_as_float(lo4 & 0xFFFF0000u);
                v[4 + jj][2 * m]     = __uint_as_float(hi4 << 16);
                v[4 + jj][2 * m + 1] = __uint_as_float(hi4 & 0xFFFF0000u);
            }
        }
    }
    else
    {
        uint4 ra[8], rb[8];
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            const long row = r0 + j;
            if(row < end)
            {
                if constexpr(PROD == 1)
                {
                    ra[j] = *reinterpret_cast<const uint4*>(s0 + row * 2 * I + src);
                    rb[j] = *reinterpret_cast<const uint4*>(s0 + row * 2 * I + I + src);
                }
                else
                    ra[j] = *reinterpret_cast<const uint4*>(s0 + row * F + src);
                if constexpr(PROD == 3)
                    sc[j] = score[row];
            }
        }
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            const long row = r0 + j;
            if(row < end)
            {
                if constexpr(PROD == 1)
                {
                    float g[8], u[8];
                    unpack8(ra[j], g);
                    unpack8(rb[j], u);
#pragma unroll
                    for(int k = 0; k < 8; ++k)
                        v[j][k] = rbf(silu(g[k]) * u[k]);
                }
                else
                    unpack8(ra[j], v[j]);
            }
            else
            {
#pragma unroll
                for(int k = 0; k < 8; ++k)
                    v[j][k] = 0.0f;
            }
        }
    }

#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        float amax = 0.0f;
#pragma unroll
        for(int k = 0; k < 8; ++k)
            amax = qmax(amax, __builtin_fabsf(v[j][k]));
        amax = row4_max(amax);
        uint32_t e8;
        float qs;
        mx_scale(amax, e8, qs);
        uint32_t q = 0;
#pragma unroll
        for(int m = 0; m < 4; ++m)
            q = cvt_fp4x2(q, v[j][2 * m] * qs, v[j][2 * m + 1] * qs, m);
        const int lr               = 32 * w + 8 * rq + j;
        lrow[lr * LDS_STRIDE + fg] = q;
        if((fg & 3) == 0)
            lrs[lr * 4 + (fg >> 2)] = uint8_t(e8);
    }

    if constexpr(PROD == 3)
    {
#pragma unroll
        for(int j = 0; j < 8; ++j)
            if(r0 + j < end)
            {
#pragma unroll
                for(int k = 0; k < 8; ++k)
                    v[j][k] = rbf(v[j][k] * sc[j]);
            }
    }

#pragma unroll
    for(int k = 0; k < 8; ++k)
    {
        float amax = 0.0f;
#pragma unroll
        for(int j = 0; j < 8; ++j)
            amax = qmax(amax, __builtin_fabsf(v[j][k]));
        amax = quad_max(amax);
        uint32_t e8;
        float qs;
        mx_scale(amax, e8, qs);
        uint32_t q = 0;
#pragma unroll
        for(int m = 0; m < 4; ++m)
            q = cvt_fp4x2(q, v[2 * m][k] * qs, v[2 * m + 1][k] * qs, m);
        const int lc                       = 8 * fg + k;
        lcol[lc * LDS_STRIDE + 4 * w + rq] = q;
        if(rq == 0)
            lcs[lc * 4 + w] = uint8_t(e8);
    }

    __syncthreads();

    // Each output row / column of the tile is 64 contiguous payload bytes and 4 scale bytes;
    // write them as 16B / 4B stores from the LDS staging buffers.
    auto feat_of = [&](int g) {
        if constexpr(PROD == 2)
            return g < 8 ? tf * 64 + 8 * g : I + tf * 64 + 8 * (g - 8);
        else
            return tf * 128 + 8 * g;
    };
    const int t = threadIdx.x;
#pragma unroll
    for(int it = 0; it < 2; ++it)
    {
        const int idx      = it * 256 + t;
        const int l        = idx >> 2;
        const int p        = idx & 3;
        const uint32_t* cs = lcol + l * LDS_STRIDE + 4 * p;
        const long c       = feat_of(l >> 3) + (l & 7);
        *reinterpret_cast<uint4*>(cpay + c * (P / 2) + ptk0 / 2 + 16 * p) =
            make_uint4(cs[0], cs[1], cs[2], cs[3]);
        const long row = tok0 + l;
        if(row < end)
        {
            const uint32_t* rs = lrow + l * LDS_STRIDE + 4 * p;
            *reinterpret_cast<uint4*>(rpay + row * (F / 2) + feat_of(4 * p) / 2) =
                make_uint4(rs[0], rs[1], rs[2], rs[3]);
        }
    }
    if(t < 128)
    {
        const long c = feat_of(t >> 3) + (t & 7);
        *reinterpret_cast<uint32_t*>(csc + c * (P / 32) + ptk0 / 32) =
            *reinterpret_cast<const uint32_t*>(lcs + 4 * t);
    }
    else
    {
        const int l    = t - 128;
        const long row = tok0 + l;
        if(row < end)
        {
            const uint32_t s4 = *reinterpret_cast<const uint32_t*>(lrs + 4 * l);
            if constexpr(PROD == 2)
            {
                *reinterpret_cast<uint16_t*>(rsc + row * (F / 32) + feat_of(0) / 32) = uint16_t(s4);
                *reinterpret_cast<uint16_t*>(rsc + row * (F / 32) + feat_of(8) / 32) =
                    uint16_t(s4 >> 16);
            }
            else
                *reinterpret_cast<uint32_t*>(rsc + row * (F / 32) + feat_of(0) / 32) = s4;
        }
    }
}

} // namespace

namespace aiter {

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
                                 void* stream)
{
    if(nblocks <= 0)
        return 0;
    if(P % 256)
        return -1;
    int tiles_f;
    if(prod == 2)
    {
        if(I % 64 || F != 2 * I)
            return -2;
        tiles_f = I / 64;
    }
    else
    {
        if(F % 128 || (prod == 1 && F != I))
            return -2;
        tiles_f = F / 128;
    }
    const dim3 grid(nblocks * 2 * tiles_f), block(256);
    auto a0 = static_cast<const uint16_t*>(s0);
    auto a1 = static_cast<const uint16_t*>(s1);
    auto sc = static_cast<const float*>(score);
    auto rp = static_cast<uint8_t*>(rpay);
    auto rs = static_cast<uint8_t*>(rsc);
    auto cp = static_cast<uint8_t*>(cpay);
    auto cs = static_cast<uint8_t*>(csc);
    auto rt = static_cast<const int*>(routes);
#define FQ_LAUNCH(PROD)                                                                                  \
    hipLaunchKernelGGL((mxfp4_moe_fused_quant_kernel<PROD>), grid, block, 0, (hipStream_t)stream, a0, a1, \
                       sc, rp, rs, cp, cs, rt, F, P, I, tiles_f)
    switch(prod)
    {
    case 0: FQ_LAUNCH(0); break;
    case 1: FQ_LAUNCH(1); break;
    case 2: FQ_LAUNCH(2); break;
    case 3: FQ_LAUNCH(3); break;
    default: return -3;
    }
#undef FQ_LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

} // namespace aiter
