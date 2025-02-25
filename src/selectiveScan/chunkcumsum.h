
#pragma once

#include <cuda_fp8.h>
#include <mma.h>

#include "suggestify/common/cudaBf16Fallbacks.cuh"

#include "Common.h"
#include "CudaType.h"
#include "Poly.h"

namespace suggestify
{
namespace kernels
{

typedef void (*ChunkCumsumKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_,
    void* g_mxdc_,
    void* g_mxdA_,
    void const* g_mxdt_,
    void const* g_mxdb_,
    void const* g_mxA_,
    void const* g_mxZ_,
    bool removePadding_, int const* lastTokenIdsPtr_, bool dtSoftplus_);

template <int Q_, int tileH_, int warpH_, class Tp_, class Wt_>
__global__ std::enable_if_t<std::is_same_v<Tp_, half> || std::is_same_v<Tp_, __nv_bfloat16>> chunk_cumsum_kernel(int B_,
    int L_, int H_, int P_, int G_, int N_,
    void* g_mxdc_,
    void* g_mxdA_,
    void const* g_mxdt_,
    void const* g_mxdb_,
    void const* g_mxA_,
    void const* g_mxZ_,
    bool removePadding_, int const* lastTokenIdsPtr_, bool dtSoftplus_)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace suggestify::common;

    auto blockIdx_x = Rn<ID>{int(blockIdx.x)};
    auto blockIdx_y = Rn<ID>{int(blockIdx.y)};
    auto blockIdx_z = Rn<ID>{int(blockIdx.z)};

    auto threadIdx_x = Rn<ID, 32>{int(threadIdx.x)};
    auto threadIdx_y = Rn<ID, warpH_>{int(threadIdx.y)};

    auto L = Rn<ID>{L_};
    auto H = Rn<ID>{H_};
    auto Q = cn<Q_>;
    auto C = Rn<ID>{div_up(L.var, Q_)};

    auto Z_stride = Rn<ID>{(g_mxZ_ ? 2 : 1) * H_ * P_ + 2 * G_ * N_ + round_up(H_, 8)};

    auto aStart = blockIdx_z * L;
    auto cStart = blockIdx_z * C;

    if (removePadding_)
    {
        aStart = Rn<ID>{int(blockIdx.z ? lastTokenIdsPtr_[blockIdx.z - 1] : 0)};
        cStart = Rn<ID>{int(blockIdx.z ? div_up(aStart.var, Q_) + blockIdx.z - 1 : 0)};
        L = Rn<ID>{lastTokenIdsPtr_[blockIdx.z] - aStart.var};
        C = Rn<ID>{div_up(L.var, Q_)};
    }
    else
    {
        L = Rn<ID>{lastTokenIdsPtr_[blockIdx.z]};
        C = Rn<ID>{div_up(L.var, Q_)};
    }

    if (blockIdx_y * Q >= L)
        return;

    float* g_mxdc = (float*) g_mxdc_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    float* g_mxdA = (float*) g_mxdA_ + int64_t(get(cStart + blockIdx_y)) * get(H * Q);
    Tp_ const* g_mxdt = (Tp_ const*) g_mxdt_ + int64_t(get(aStart + blockIdx_y * Q)) * get(Z_stride);
    Wt_ const* g_mxdb = (Wt_ const*) g_mxdb_;
    Wt_ const* g_mxA = (Wt_ const*) g_mxA_;

    extern __shared__ float smem[];

    float* s_mxdc = smem;
    float* s_mxdb = smem + Q_ * tileH_;
    float* s_mxA = smem + Q_ * tileH_ + tileH_;

    auto thread = [=](auto iStep) { return iStep * cn<warpH_ * 32> + threadIdx_y * cn<32> + threadIdx_x; };

#pragma unroll
    for (Rn<UNROLL, div_up(tileH_, warpH_ * 32)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<tileH_>)
            if (blockIdx_x * cn<tileH_> + thread(iStep) < H)
            {
                s_mxdb[get(thread(iStep))] = g_mxdb ? float(g_mxdb[get(blockIdx_x * cn<tileH_> + thread(iStep))]) : 0.f;
                s_mxA[get(thread(iStep))] = float(g_mxA[get(blockIdx_x * cn<tileH_> + thread(iStep))]);
            }
            else
            {
                s_mxdb[get(thread(iStep))] = 0.f;
                s_mxA[get(thread(iStep))] = 0.f;
            }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(Q_ * tileH_, warpH_ * 256)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) * cn<8> < cn<Q_ * tileH_>)
        {
            if (blockIdx_y * Q + thread(iStep) * cn<8> / cn<tileH_> < L)
            {
                Tp_ tmp[8];

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                    if (blockIdx_x * cn<tileH_> + thread(iStep) * cn<8> % cn<tileH_> + Rn<UNROLL, 8>{i} < H)
                        *(int*) &tmp[i]
                            = *(int*) &g_mxdt[get(thread(iStep) * cn<8> / cn<tileH_> * Z_stride + Z_stride
                                                  + blockIdx_x * cn<tileH_> + thread(iStep) * cn<8> % cn<tileH_>)
                                - round_up(H_, 8) + i];
                    else
                        *(int*) &tmp[i] = 0;

#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    float2 tmp2 = std::is_same_v<Tp_, half> ? __half22float2(*(half2*) &tmp[i])
                                                            : bf1622float2(*(bf162*) &tmp[i]);

                    tmp2.x += s_mxdb[get(thread(iStep) * cn<8> % cn<tileH_> + Rn<UNROLL, 8>{i})];
                    tmp2.y += s_mxdb[get(thread(iStep) * cn<8> % cn<tileH_> + Rn<UNROLL, 8>{i + 1})];

                    if (dtSoftplus_)
                    {
                        float softplusx = log1p(expf(tmp2.x));
                        float softplusy = log1p(expf(tmp2.y));
                        tmp2.x = tmp2.x > 32.f ? tmp2.x : softplusx;
                        tmp2.y = tmp2.y > 32.f ? tmp2.y : softplusy;
                    }
                    else
                    {
                        tmp2.x = tmp2.x > 0.f ? tmp2.x : 0.f;
                        tmp2.y = tmp2.y > 0.f ? tmp2.y : 0.f;
                    }

                    s_mxdc[get((thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) % cn<tileH_> * Q
                        + (thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) / cn<tileH_>)]
                        = tmp2.x;
                    s_mxdc[get((thread(iStep) * cn<8> + Rn<UNROLL, 8>{i + 1}) % cn<tileH_> * Q
                        + (thread(iStep) * cn<8> + Rn<UNROLL, 8>{i + 1}) / cn<tileH_>)]
                        = tmp2.y;
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 8; i++)
                {
                    s_mxdc[get((thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) % cn<tileH_> * Q
                        + (thread(iStep) * cn<8> + Rn<UNROLL, 8>{i}) / cn<tileH_>)]
                        = 0.f;
                }
            }
        }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(Q_ * tileH_, warpH_ * 128)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) * cn<4> < cn<Q_ * tileH_>)
            if (blockIdx_x * cn<tileH_> + thread(iStep) * cn<4> / Q < H)
            {
                float4 tmp4 = *(float4*) &s_mxdc[get(thread(iStep) * cn<4>)];

                *(float4*) &g_mxdc[get(
                    (thread(iStep) * cn<4> / Q + blockIdx_x * cn<tileH_>) *Q + thread(iStep) * cn<4> % Q)]
                    = tmp4;
            }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(tileH_, warpH_ * 32)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) < cn<tileH_>)
        {
            float sum = 0.f;

#pragma unroll
            for (Rn<UNROLL, Q_> iQ; iQ.var < iQ.size; iQ.var++)
            {
                sum += s_mxdc[get(thread(iStep) * Q + iQ)];
                s_mxdc[get(thread(iStep) * Q + iQ)] = sum;
            }
        }

    __syncthreads();

#pragma unroll
    for (Rn<UNROLL, div_up(Q_ * tileH_, warpH_ * 128)> iStep; iStep.var < iStep.size; iStep.var++)
        if (thread(iStep) * cn<4> < cn<Q_ * tileH_>)
            if (blockIdx_x * cn<tileH_> + thread(iStep) * cn<4> / Q < H)
            {
                float r_A = s_mxA[get(thread(iStep) * cn<4> / Q)];

                float4 tmp4 = *(float4*) &s_mxdc[get(thread(iStep) * cn<4>)];

                tmp4.x *= r_A;
                tmp4.y *= r_A;
                tmp4.z *= r_A;
                tmp4.w *= r_A;

                *(float4*) &g_mxdA[get(
                    (thread(iStep) * cn<4> / Q + blockIdx_x * cn<tileH_>) *Q + thread(iStep) * cn<4> % Q)]
                    = tmp4;
            }
#endif
}

typedef ChunkCumsumKernelFunc (*GetChunkCumsumKernelFunc)(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_);

template <class Tp_, class Wt_>
ChunkCumsumKernelFunc getChunkCumsumKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_, int numTokens_,
    dim3* blockDims_, dim3* threadDims_, int* sharedMem_)
{
    int B = B_;
    int L = L_;
    int H = round_up(H_, 8);
    int Q = Q_;
    int C = div_up(L, Q);

    int64_t compute = int64_t(numTokens_) * H * Q;

    auto set = [&](int tileH, int warpH, ChunkCumsumKernelFunc func)
    {
        auto sharedMem = (Q + 2) * tileH * 4;

        *blockDims_ = dim3(div_up(H, tileH), C, B);
        *threadDims_ = dim3(32, warpH);
        *sharedMem_ = sharedMem;

        return func;
    };

    if (Q_ == 256 && H % 16 == 0)
    {
        if (compute >= (1LL << 29))
            return set(16, 8, chunk_cumsum_kernel<256, 16, 8, Tp_, Wt_>);
        else if (compute >= (1LL << 0))
            return set(8, 8, chunk_cumsum_kernel<256, 8, 8, Tp_, Wt_>);
    }

    if (Q_ == 256 && H % 8 == 0)
    {
        if (compute >= (1LL << 0))
            return set(8, 8, chunk_cumsum_kernel<256, 8, 8, Tp_, Wt_>);
    }

    if (Q_ == 128 && H % 16 == 0)
    {
        if (compute >= (1LL << 27))
            return set(16, 8, chunk_cumsum_kernel<128, 16, 8, Tp_, Wt_>);
        else if (compute >= (1LL << 26))
            return set(8, 8, chunk_cumsum_kernel<128, 8, 8, Tp_, Wt_>);
        else if (compute >= (1LL << 0))
            return set(8, 4, chunk_cumsum_kernel<128, 8, 4, Tp_, Wt_>);
    }

    if (Q_ == 128 && H % 8 == 0)
    {
        if (compute >= (1LL << 26))
            return set(8, 8, chunk_cumsum_kernel<128, 8, 8, Tp_, Wt_>);
        else if (compute >= (1LL << 0))
            return set(8, 4, chunk_cumsum_kernel<128, 8, 4, Tp_, Wt_>);
    }

    return nullptr;
}

extern GetChunkCumsumKernelFunc getChunkCumsumKernel_fp16_fp16;
extern GetChunkCumsumKernelFunc getChunkCumsumKernel_fp16_fp32;
extern GetChunkCumsumKernelFunc getChunkCumsumKernel_bf16_bf16;
extern GetChunkCumsumKernelFunc getChunkCumsumKernel_bf16_fp32;

static inline ChunkCumsumKernelFunc getChunkCumsumKernel(int B_, int L_, int H_, int P_, int G_, int N_, int Q_,
    int numTokens_, dim3* blockDims_, dim3* threadDims_, int* sharedMem_, CudaType tp_ = CT_FP16,
    CudaType wt_ = CT_FP32)
{
    if (tp_ == CT_FP16 && wt_ == CT_FP16)
        return getChunkCumsumKernel_fp16_fp16(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);
    else if (tp_ == CT_FP16 && wt_ == CT_FP32)
        return getChunkCumsumKernel_fp16_fp32(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);
    else if (tp_ == CT_BF16 && wt_ == CT_BF16)
        return getChunkCumsumKernel_bf16_bf16(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);
    else if (tp_ == CT_BF16 && wt_ == CT_FP32)
        return getChunkCumsumKernel_bf16_fp32(
            B_, L_, H_, P_, G_, N_, Q_, numTokens_, blockDims_, threadDims_, sharedMem_);

    return nullptr;
}

}
}

