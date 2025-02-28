

#include "../common/assert.h"
#include "../common/cudaTypeUtils.cuh"
#include "../common/cudaUtils.h"
#include "../common/quantTypeUtils.cuh"
#include "../common/reduceKernelUtils.cuh"
#include "quantization.h"
#include <float.h>

using namespace suggestify::common;

namespace suggestify
{
namespace kernels
{

__global__ void quantizedKernel(char4* dst, float4 const* src, int64_t const sizeDiv4, float const* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        float const scale = __ldg(scalePtr);
        char4 tmp;
        float4 const floatTmp = __ldg(src + idx);
        tmp.x = cuda_cast<int8_t>(floatTmp.x * scale);
        tmp.y = cuda_cast<int8_t>(floatTmp.y * scale);
        tmp.z = cuda_cast<int8_t>(floatTmp.z * scale);
        tmp.w = cuda_cast<int8_t>(floatTmp.w * scale);
        dst[idx] = tmp;
    }
}

__global__ void quantizedKernel(char4* dst, half2 const* src, int64_t const sizeDiv4, float const* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        float const scale = __ldg(scalePtr);
        char4 tmp;
        int srcId = idx << 1;

        uint2 const h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

        half2 const half2Tmp = reinterpret_cast<half2 const&>(h2.x);
        half2 const half2Tmp2 = reinterpret_cast<half2 const&>(h2.y);

        tmp.x = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
        tmp.y = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
        tmp.z = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
        tmp.w = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
        dst[idx] = tmp;
    }
}

#ifdef ENABLE_BF16
__global__ void quantizedKernel(char4* dst, __nv_bfloat162 const* src, int64_t const sizeDiv4, float const* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        float const scale = __ldg(scalePtr);
        char4 tmp;
        int srcId = idx << 1;

        uint2 const h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

        __nv_bfloat162 const bfloat162Tmp = reinterpret_cast<__nv_bfloat162 const&>(h2.x);
        __nv_bfloat162 const bfloat162Tmp2 = reinterpret_cast<__nv_bfloat162 const&>(h2.y);

        tmp.x = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.x) * scale);
        tmp.y = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.y) * scale);
        tmp.z = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.x) * scale);
        tmp.w = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.y) * scale);

        dst[idx] = tmp;
    }
}
#endif

template <typename T>
void invokeQuantization(
    int8_t* dst, T const* src, int64_t const size, float const* scalePtr, cudaStream_t stream, int maxGridSize)
{
    CHECK_WITH_INFO(size % 4 == 0, "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

    int numBlocks{static_cast<int>((size + 255) / 256)};
    dim3 grid(std::min(numBlocks, maxGridSize));
    CHECK_WITH_INFO(grid.x <= maxGridSize, "[ERROR][invokeQuantization] grid max size is exceeded\n");
    dim3 block(64);
    if (std::is_same_v<T, float>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (float4 const*) src, size / 4, scalePtr);
    }
    else if (std::is_same_v<T, half>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (half2 const*) src, size / 4, scalePtr);
    }
#ifdef ENABLE_BF16
    else if (std::is_same_v<T, __nv_bfloat16>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (__nv_bfloat162 const*) src, size / 4, scalePtr);
    }
#endif
}

template void invokeQuantization<float>(
    int8_t* dst, float const* src, int64_t const size, float const* scalePtr, cudaStream_t stream, int maxGridSize);

template void invokeQuantization<half>(
    int8_t* dst, half const* src, int64_t const size, float const* scalePtr, cudaStream_t stream, int maxGridSize);

#ifdef ENABLE_BF16
template void invokeQuantization<__nv_bfloat16>(int8_t* dst, __nv_bfloat16 const* src, int64_t const size,
    float const* scalePtr, cudaStream_t stream, int maxGridSize);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int NUM_ELTS>
struct DstVec
{
    static_assert("not implemented.");
};

template <>
struct DstVec<float2, 2>
{
    using Type = uint32_t;
};

template <>
struct DstVec<half2, 4>
{
    using Type = uint2;
};

#ifdef ENABLE_BF16

template <>
struct DstVec<__nv_bfloat162, 4>
{
    using Type = uint2;
};

#endif // ENABLE_BF16

template <typename T>
struct DstVec<T, 4>
{
    static_assert(sizeof(T) == 4, "not implemented.");
    using Type = uint32_t;
};

template <typename T>
struct DstVec<T, 8>
{
    static_assert(sizeof(T) == 2, "not implemented.");
    using Type = uint2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function of getting the absMax of all elements in the vector after clamping.
// Pack two elements in order to use possible hmax2 instructions.
template <typename T>
inline __device__ void clampAndAbsMax(T& localMax, uint4& vec, T const clampMin, T const clampMax)
{
    static constexpr int NUM_ELTS = sizeof(uint4) / sizeof(T);

#pragma unroll
    for (int i = 0; i < NUM_ELTS; ++i)
    {
        T& val = reinterpret_cast<T*>(&vec)[i];
        val = cuda_clamp(val, clampMin, clampMax);
        localMax = cuda_max(localMax, cuda_abs(val));
    }
}

// Helper function of quantizing the vector and storing it to global memory.
// Pack two elements in order to use fast convert instructions.
template <typename T, typename QuantT, bool USE_SMEM>
inline __device__ void quantizeAndStore(
    QuantT* dstPtr, uint4 vec, T const clampMin, T const clampMax, float const scaleOrigQuant)
{
    static constexpr int NUM_ELTS = sizeof(uint4) / sizeof(T);

    using DstVecType = typename DstVec<T, NUM_ELTS>::Type;
    DstVecType dstVec;
#pragma unroll
    for (int i = 0; i < NUM_ELTS; ++i)
    {
        T val = reinterpret_cast<T*>(&vec)[i];
        // Values loaded from smem has already been clamped.
        if constexpr (!USE_SMEM)
        {
            val = cuda_clamp(val, clampMin, clampMax);
        }
        float2 val2 = cuda_cast<float2>(val);
        val2.x *= scaleOrigQuant;
        val2.y *= scaleOrigQuant;
        QuantT quantVal = cuda_cast<QuantT>(val2);
        reinterpret_cast<QuantT*>(&dstVec)[i] = quantVal;
    }
    // Store to destination buffer.
    *reinterpret_cast<DstVecType*>(dstPtr) = dstVec;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename QuantT, bool USE_SMEM>
__global__ void perTokenQuantization(QuantT* dst, T const* src, int64_t const numRows, int64_t const numCols,
    float const* clampPtr, float* scalePtr, float* sumPtr, bool hasFp8MinScaling)
{
    // Smem buffer.
    extern __shared__ uint4 smemBuffer[];

    // The clamping minimum / maximum values.
    T const clampMin = cuda_cast<T>(clampPtr ? clampPtr[0] : -FLT_MAX);
    T const clampMax = cuda_cast<T>(clampPtr ? clampPtr[1] : FLT_MAX);

    // Pack two elements in order to use higher through instructions.
    using T2 = typename packed_as<T, 2>::type;
    using QuantT2 = typename packed_as<QuantT, 2>::type;
    T2 const clampMin2 = cuda_cast<T2, T>(clampMin);
    T2 const clampMax2 = cuda_cast<T2, T>(clampMax);

    // The quantized data type's maximum value (upper-bound).
    static constexpr float MAX_QUANT_VAL = QuantTypeStaticVals<QuantT>::MAX_VAL;
    // The minimum scaling factor (lower-bound).
    static constexpr float MIN_SCALING_FACTOR = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR;
    static constexpr float MIN_SCALING_FACTOR_RCP = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR_RCP;

    // The number of elements in the packed uint4 vec.
    static constexpr int NUM_ELTS_PER_VEC = sizeof(uint4) / sizeof(T);
    static constexpr int NUM_ELTS2_PER_VEC = sizeof(uint4) / sizeof(T2);

    // The number of vectors in the column.
    int const numColVecs = numCols / NUM_ELTS_PER_VEC;
    // The vector pointers for src.
    uint4 const* srcVec = reinterpret_cast<uint4 const*>(src) + blockIdx.x * numColVecs;
    // The pointer for dst.
    QuantT* dstRow = dst + blockIdx.x * numCols;
    // T const* srcRow = src + blockIdx.x * numCols;

    T2 localMax2 = cuda_cast<T2, T>(T(1e-6f));
    float2 localSum2 = {0.f, 0.f};

    for (int i = threadIdx.x; i < numColVecs; i += blockDim.x)
    {
        uint4 vec = srcVec[i];

#pragma unroll
        for (int j = 0; j < NUM_ELTS2_PER_VEC; ++j)
        {
            T2& val2 = reinterpret_cast<T2*>(&vec)[j];
            val2 = cuda_clamp(val2, clampMin2, clampMax2);
            localMax2 = cuda_max(localMax2, cuda_abs(val2));
            // TODO: template the version that requires sum to avoid dynamic branching.
            if (sumPtr != nullptr)
            {
                localSum2.x += cuda_cast<float>(val2.x);
                localSum2.y += cuda_cast<float>(val2.y);
            }
        }
        // Avoid reloading from global memory.
        if constexpr (USE_SMEM)
        {
            smemBuffer[i] = vec;
        }
    }
    float const rowMax = blockAllReduceMax(cuda_cast<float>(cuda_max<T, T2>(localMax2)));
    if (threadIdx.x == 0)
    {
        scalePtr[blockIdx.x]
            = hasFp8MinScaling ? cuda_max(rowMax / MAX_QUANT_VAL, MIN_SCALING_FACTOR) : (rowMax / MAX_QUANT_VAL);
    }

    if (sumPtr != nullptr)
    {
        float rowSum[1] = {cuda_sum<float>(localSum2)};
        blockReduceSumV2<float, 1>(rowSum);
        if (threadIdx.x == 0)
        {
            sumPtr[blockIdx.x] = rowSum[0];
        }
    }

    float const scaleOrigQuant
        = hasFp8MinScaling ? fminf(MAX_QUANT_VAL / rowMax, MIN_SCALING_FACTOR_RCP) : MAX_QUANT_VAL / rowMax;
    for (int i = threadIdx.x; i < numColVecs; i += blockDim.x)
    {
        uint4 vec = USE_SMEM ? smemBuffer[i] : srcVec[i];
        QuantT2* dstPtr = reinterpret_cast<QuantT2*>(dstRow + i * NUM_ELTS_PER_VEC);
        quantizeAndStore<T2, QuantT2, USE_SMEM>(dstPtr, vec, clampMin2, clampMax2, scaleOrigQuant);
    }
}

// Do per-token (row) quantization from fp16/bf16/fp32 to int8/fp8_e4m3.
template <typename T, typename QuantT>
void invokePerTokenQuantization(QuantT* dst, T const* src, int64_t const numRows, int64_t const numCols,
    float const* clampPtr, float* scalePtr, float* sumPtr, QuantMode quantMode, cudaStream_t stream)
{
    // each block is responsible for a single row
    dim3 const block(512);
    dim3 const grid(numRows);

    // The number of elements in the packed uint4 vec.
    static constexpr int NUM_ELTS_PER_VEC = sizeof(uint4) / sizeof(T);
    CHECK_WITH_INFO(numCols % NUM_ELTS_PER_VEC == 0, "Not supported.");

    // Cache vectors to smem to avoid reloading.
    size_t const dynamicSmemSz = numCols * sizeof(T);
    // Need to check if smem capacity is enough.
    bool useSmem = true;
    if (dynamicSmemSz >= 48 * 1024)
    {
        cudaError_t res = cudaFuncSetAttribute(
            perTokenQuantization<T, QuantT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicSmemSz);
        // Fall back to reloading-reversion if smem is not enough.
        useSmem = (res == cudaSuccess);
    }

    // Enable min_scaling_factor if it is fp8 rowwise per-token quantization.
    bool hasFp8MinScaling = quantMode.hasFp8RowWise();
    // Do we use smem ?
    if (useSmem)
    {
        perTokenQuantization<T, QuantT, true><<<grid, block, dynamicSmemSz, stream>>>(
            dst, src, numRows, numCols, clampPtr, scalePtr, sumPtr, hasFp8MinScaling);
    }
    else
    {
        perTokenQuantization<T, QuantT, false>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, clampPtr, scalePtr, sumPtr, hasFp8MinScaling);
    }
}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(T, QuantT)                                                           \
    template void invokePerTokenQuantization(QuantT* dst, const T* src, const int64_t numRows, const int64_t numCols,  \
        float const* clampPtr, float* scalePtr, float* sumPtr, QuantMode quantMode, cudaStream_t stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float, int8_t);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half, int8_t);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16, int8_t);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float, __nv_fp8_e4m3);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

} // namespace kernels
} // namespace suggestify
