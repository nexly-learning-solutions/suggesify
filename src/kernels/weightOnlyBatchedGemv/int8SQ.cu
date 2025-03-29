#include "../src/weightOnlyBatchedGemv/int8SQ.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <cassert>
#include <type_traits>

namespace sugesstify {
namespace kernels {
namespace smooth_quant {


constexpr int kWarpSize = 32;

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t status = (call);                                     \
        if (status != cudaSuccess) {                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)     \
                      << " at " << __FILE__ << ":" << __LINE__           \
                      << " - error code: " << static_cast<int>(status)   \
                      << std::endl;                                      \
            std::cerr << "Halting execution due to CUDA error." << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#ifdef DEBUG
#define DEBUG_PRINT(...)                                                        \
    do {                                                                        \
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << __VA_ARGS__ << std::endl; \
    } while (0)
#else
#define DEBUG_PRINT(...)
#endif

template <typename Type, int CtaM, int CtaN, int Threads, bool PerChannel, bool PerToken>
__global__ void int8_sq(int8_t const* act, int8_t const* weight, float const* scale_channels, float const* scale_tokens,
    Type* output, int m, int n, int k) {

    using VecType = int4;
    static constexpr int kStepK = 128 / (8 * sizeof(int8_t));
    static constexpr int CtaK = kStepK * Threads;

    int tile_id_m = blockIdx.x * CtaM;
    int tile_id_n = blockIdx.y * CtaN;
    int tid = threadIdx.x;

    int8_t tile_a[kStepK];
    int8_t tile_w[CtaN * kStepK];
    int acc[CtaM * CtaN];

#pragma unroll
    for (int i = 0; i < CtaM * CtaN; ++i) {
        acc[i] = 0;
    }

    act += tile_id_m * k;
    weight += tile_id_n * k;
    output += tile_id_m * n + tile_id_n;

    for (int idx_k = tid * kStepK; idx_k < k; idx_k += CtaK) {

#pragma unroll
        for (int i = 0; i < CtaN; ++i) {
            reinterpret_cast<VecType*>(tile_w)[i] = reinterpret_cast<VecType const*>(weight + i * k + idx_k)[0];
        }

#pragma unroll
        for (int i = 0; i < CtaM; ++i) {
            reinterpret_cast<VecType*>(tile_a)[0] = reinterpret_cast<VecType const*>(act + i * k + idx_k)[0];
#pragma unroll
            for (int j = 0; j < CtaN; ++j) {
#pragma unroll
                for (int l = 0; l < kStepK; l += 4) {
                    acc[i * CtaN + j] = __dp4a(reinterpret_cast<int*>(tile_a + l)[0],
                        reinterpret_cast<int*>(tile_w + j * kStepK + l)[0], acc[i * CtaN + j]);
                }
            }
        }
    }

    static constexpr int kWarpNum = Threads / kWarpSize;
    int warp_id = tid / kWarpSize, lane_id = tid % kWarpSize;

    __shared__ int shmem[CtaM * CtaN * kWarpNum];

#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
#pragma unroll
        for (int j = 0; j < CtaN; ++j) {
            int val = acc[i * CtaN + j];
            val += __shfl_xor_sync(0xffffffff, val, 16);
            val += __shfl_xor_sync(0xffffffff, val, 8);
            val += __shfl_xor_sync(0xffffffff, val, 4);
            val += __shfl_xor_sync(0xffffffff, val, 2);
            val += __shfl_xor_sync(0xffffffff, val, 1);
            if (lane_id == 0) {
                shmem[i * CtaN + j + warp_id * CtaM * CtaN] = val;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int ii = tid; ii < CtaM * CtaN; ii += Threads) {
        int mid = ii / CtaN, nid = ii % CtaN;
        float scale_channel, scale_token;

        if constexpr (PerChannel) {
            scale_channel = scale_channels[tile_id_n + nid];
        } else {
            scale_channel = scale_channels[0];
        }

        if constexpr (PerToken) {
            scale_token = scale_tokens[tile_id_m + mid];
        } else {
            scale_token = scale_tokens[0];
        }

        int val = 0;
#pragma unroll
        for (int jj = 0; jj < kWarpNum; ++jj) {
            val += shmem[jj * CtaM * CtaN + ii];
        }
        output[mid * n + nid] = static_cast<Type>(static_cast<float>(val) * scale_channel * scale_token);
    }
}

template <typename Type, int CtaM, int CtaN, int Threads, bool PerChannel, bool PerToken>
void int8_sq_kernel(Params& params, cudaStream_t s) {
    dim3 block(Threads);
    dim3 grid(params.m / CtaM, params.n / CtaN);
    int8_sq<Type, CtaM, CtaN, Threads, PerChannel, PerToken><<<grid, block, 0, s>>>(params.act, params.weight,
        params.scale_channels, params.scale_tokens, reinterpret_cast<Type*>(params.output), params.m, params.n,
        params.k);
}

template <typename Type, bool PerChannel, bool PerToken>
void algo_tactic_dispatcher(Params& params, cudaStream_t s) {
#define DISPATCH(TargetM, CtaM, CtaN, Threads)                                                                         \
    if (params.m == TargetM) {                                                                                       \
        DEBUG_PRINT("Dispatching: Type=%s, PerChannel=%s, PerToken=%s, TargetM=%d, CtaM=%d, CtaN=%d, Threads=%d",    \
            typeid(Type).name(), PerChannel ? "true" : "false", PerToken ? "true" : "false", TargetM, CtaM, CtaN,    \
            Threads);                                                                                                \
        int8_sq_kernel<Type, CtaM, CtaN, Threads, PerChannel, PerToken>(params, s);                                    \
        return;                                                                                                        \
    }
    DISPATCH(1, 1, 2, 128);
    DISPATCH(2, 2, 2, 128);
    DISPATCH(3, 3, 2, 128);
    DISPATCH(4, 4, 2, 128);
#undef DISPATCH
    DEBUG_PRINT(
        "Error: No matching tactic found for: Type=%s, PerChannel=%s, PerToken=%s, m=%d", typeid(Type).name(),
        PerChannel ? "true" : "false", PerToken ? "true" : "false", params.m);
}

template <typename Type>
void int8_sq_launcher(Params& params, cudaStream_t s) {
#define DISPATCH(PerChannel, PerToken)                                                                                 \
    if (params.quant_mode.hasPerChannelScaling() == PerChannel && params.quant_mode.hasPerTokenScaling() == PerToken) \
    {                                                                                                                  \
        DEBUG_PRINT("Launching: Type=%s, PerChannel=%s, PerToken=%s", typeid(Type).name(),                          \
            PerChannel ? "true" : "false", PerToken ? "true" : "false");                                              \
        algo_tactic_dispatcher<Type, PerChannel, PerToken>(params, s);                                                 \
        return;                                                                                                        \
    }
    DISPATCH(false, false);
    DISPATCH(false, true);
    DISPATCH(true, false);
    DISPATCH(true, true);
#undef DISPATCH
    DEBUG_PRINT(
        "Error: No matching launcher found for: Type=%s, PerChannel=%s, PerToken=%s", typeid(Type).name(),
        params.quant_mode.hasPerChannelScaling() ? "true" : "false",
        params.quant_mode.hasPerTokenScaling() ? "true" : "false");
}

template void int8_sq_launcher<float>(Params& params, cudaStream_t s);
template void int8_sq_launcher<half>(Params& params, cudaStream_t s);
template void int8_sq_launcher<int>(Params& params, cudaStream_t s);
#ifdef ENABLE_BF16
template void int8_sq_launcher<__nv_bfloat16>(Params& params, cudaStream_t s);
#endif

}
}
}