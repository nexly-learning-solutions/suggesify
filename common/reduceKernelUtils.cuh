#pragma once
#include <array>
#include <assert.h>
#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
#include <cooperative_groups/reduce.h>
#else
#include <cooperative_groups.h>
#endif
#include "cudaTypeUtils.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <type_traits>

namespace cg = cooperative_groups;

namespace suggestify
{
namespace common
{

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<1>
{
    using type = uint8_t;
};

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};

template <>
struct BytesToType<4>
{
    using type = uint32_t;
};

template <>
struct BytesToType<8>
{
    using type = uint64_t;
};

template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(void const* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    T const* in = static_cast<T const*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

static float constexpr HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T) (0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

template <typename T>
__inline__ __device__ T blockAllReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T) (0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[i][lane] : (T) (0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T) 0.0f;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceMaxV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] = max(val[i], __shfl_xor_sync(FINAL_MASK, val[i], mask, 32));
    }
    return (T) (0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceMaxV2(T* val)
{
    static __shared__ T shared[32][NUM];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceMaxV2<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[wid][i] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[lane][i] : (T) -1e20f;
    }
    warpReduceMaxV2<T, NUM>(val);

    return (T) 0.0f;
}

template <int NUM>
__inline__ __device__ void cgBlockReduceSumElements(float* element_list, float* cgBlockReduceSumElements_shm)
{
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    int const tid = cta.thread_rank();
    int const blockz = blockDim.x;
    for (int i = 0; i < NUM; i++)
    {
#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
        cgBlockReduceSumElements_shm[i * blockz + tid] = cg::reduce(tile, element_list[i], cg::plus<float>());
#else
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            printf("[ERROR] Not support cgBlockReduceSumElements when CUDA < 11 \n");
            assert(false);
        }
#endif
    }
    cg::sync(cta);
    if (tid == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            float beta = 0.0f;
            for (int j = 0; j < blockz; j += 32)
            {
                beta += cgBlockReduceSumElements_shm[i * blockz + j];
            }
            element_list[i] = beta;
        }
    }
}

template <typename T, int MAX_K>
struct TopK
{
    int p[MAX_K];
    T u[MAX_K];

    __device__ __forceinline__ void insert(T const elem, int const elem_id)
    {
        if (elem_id < 0)
        {
            return;
        }
        bool const need_update
            = (p[MAX_K - 1] == -1 || elem > u[MAX_K - 1] || elem == u[MAX_K - 1] && elem_id < p[MAX_K - 1]);
        if (!need_update)
        {
            return;
        }
        int i;
        for (i = MAX_K - 2; i >= 0; --i)
        {
            bool const need_decrease = (p[i] == -1 || elem > u[i] || elem == u[i] && elem_id < p[i]);
            if (!need_decrease)
                break;
        }
        for (int k = MAX_K - 2; k >= i; --k)
        {
            p[k + 1] = p[k];
            u[k + 1] = u[k];
        }
        p[i] = elem_id;
        u[i] = elem;
    }

    __device__ __forceinline__ void init()
    {
        T const MAX_T_VAL = (std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX;
        for (int i = 0; i < MAX_K; i++)
        {
            p[i] = -1;
            u[i] = -MAX_T_VAL;
        }
    }
};

template <typename T, int MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(TopK<T, MAX_K> const& a, TopK<T, MAX_K> const& b)
{
    TopK<T, MAX_K> res = a;
    for (int i = 0; i < MAX_K; ++i)
        res.insert(b.u[i], b.p[i]);
    return res;
}

template <typename T>
struct TopK_2
{
    int p = -1;
    T u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);

    __device__ __forceinline__ void insert(T elem, int elem_id)
    {
        if (elem > u)
        {
            u = elem;
            p = elem_id;
        }
    }

    __device__ __forceinline__ void init()
    {
        u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);
        p = -1;
    }
};

template <typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op_2(TopK_2<T> const& a, TopK_2<T> const& b)
{
    return a.u > b.u ? a : b;
}

template <typename T>
__device__ __forceinline__ T clamp_inf_for_half(float const input)
{
    return input;
}

template <>
__device__ __forceinline__ half clamp_inf_for_half(float const input)
{
    return input > 0.0f ? (half) min(input, HALF_FLT_MAX - 1000) : (half) max(input, -HALF_FLT_MAX + 1000);
}

}
}
