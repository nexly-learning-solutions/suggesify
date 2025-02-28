

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/memoryUtils.h"
#include "../src/samplingTopPKernels.h"
#include <cuda/atomic>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda/std/limits>
#include <cuda_fp16.h>

using namespace sugesstify::common;

namespace sugesstify
{
namespace kernels
{

using IdxT = int;
using AccT = float;

template <typename T, typename IdxT, typename AccT>
struct alignas(128) Counter
{
    T const* in;
    IdxT const* inIdx;

    IdxT oriLen;

    AccT sum;
    IdxT len;
    float p;

    IdxT previousLen;

    typename cub::Traits<T>::UnsignedBits kthValueBits;

    alignas(128) IdxT filterCnt;

    alignas(128) uint32_t finishedBlockCnt;
};

using WideT = float4;

template <typename IntType>
constexpr __host__ __device__ IntType ceilDiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}

template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b)
{
    return ceilDiv(a, b) * b;
}

template <int BitsPerPass>
__host__ __device__ int constexpr calcNumBuckets()
{
    return 1 << BitsPerPass;
}

template <typename T, int BitsPerPass>
__host__ __device__ int constexpr calcNumPasses()
{
    return ceilDiv<int>(sizeof(T) * 8, BitsPerPass);
}

template <typename T, int BitsPerPass>
__device__ int constexpr calcStartBit(int pass)
{
    int startBit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    if (startBit < 0)
    {
        startBit = 0;
    }
    return startBit;
}

template <typename T, int BitsPerPass>
__device__ uint32_t constexpr calcMask(int pass)
{
    static_assert(BitsPerPass <= 31);
    int numBits = calcStartBit<T, BitsPerPass>(pass - 1) - calcStartBit<T, BitsPerPass>(pass);
    return (1 << numBits) - 1;
}

template <typename T>
__device__ constexpr uint32_t getNumTotalMantissa()
{
    if constexpr (std::is_same_v<T, half>)
    {
        return 10;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return 23;
    }
}

template <typename T>
__device__ uint32_t calcMantissa(T value);

template <>
__device__ uint32_t calcMantissa(float value)
{
    union
    {
        uint32_t bits;
        float value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<float>();
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return input.bits & mask;
}

__device__ uint32_t calcMantissa(half value)
{
    union
    {
        uint16_t bits;
        half value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<half>();
    uint32_t t = 0u | input.bits;
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return t & mask;
}

template <typename T>
__device__ uint32_t calcExponent(T value);

template <>
__device__ uint32_t calcExponent(float value)
{
    union
    {
        uint32_t bits;
        float value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<float>();
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return input.bits & ~mask;
}

template <>
__device__ uint32_t calcExponent(half value)
{
    union
    {
        uint16_t bits;
        half value;
    } input;

    input.value = value;

    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<half>();
    uint32_t t = 0u | input.bits;
    uint32_t mask = (1u << numTotalMantissa) - 1;
    return t & ~mask;
}

__device__ float calcHalfValue(uint32_t count, uint32_t exponent, uint32_t sign, uint64_t bitSum)
{
    constexpr uint32_t numTotalBits = 64;
    constexpr uint32_t numOffset = 16;
    constexpr uint32_t numTotalMantissaHalf
        = getNumTotalMantissa<half>();
    constexpr uint32_t numTotalMantissaFloat
        = getNumTotalMantissa<float>();

    uint64_t extraInMatissa = (bitSum >> numTotalMantissaHalf);

    uint32_t numExtra = 0;
    uint32_t numDeNorm = 0;
    int numNorm = 0;
    uint32_t mask = 0;
    extraInMatissa = (exponent == 0) ? extraInMatissa : extraInMatissa + count;
    numExtra = numTotalBits - __clzll(extraInMatissa);
    numNorm = (exponent == 0) ? 0 : -1;
    if (extraInMatissa == 0)
    {
        numDeNorm = numTotalMantissaHalf - (numTotalBits - __clzll(bitSum));
    }
    exponent = exponent + ((numExtra + numNorm + 127 - 15 - numDeNorm) << numTotalMantissaHalf);
    uint32_t mantissa;
    if (extraInMatissa != 0)
    {
        int numMove = numTotalMantissaFloat - (numExtra - 1);
        mask = (1u << (numExtra - 1)) - 1;
        extraInMatissa = extraInMatissa & mask;
        if (numMove > 0)
        {
            extraInMatissa = extraInMatissa << numMove;
            mask = (1u << numTotalMantissaHalf) - 1;
            mantissa = (((bitSum & mask) << (numTotalMantissaFloat - numTotalMantissaHalf)) >> (numExtra - 1))
                | extraInMatissa;
        }
        else
        {
            mantissa = extraInMatissa >> (-1 * numMove);
        }
    }
    else
    {
        mask = (1u << numTotalMantissaHalf) - 1;
        mantissa = bitSum << (numDeNorm + 1);
        mantissa = mantissa & mask;
        mantissa = mantissa << (numTotalMantissaFloat - numTotalMantissaHalf);
    }

    uint32_t bitFloat = (sign << numOffset) | (exponent << (numTotalMantissaFloat - numTotalMantissaHalf)) | mantissa;
    return reinterpret_cast<float&>(bitFloat);
}

__device__ float calcFloatValue(uint32_t count, uint32_t exponent, uint64_t bitSum)
{
    constexpr uint32_t numTotalBits = 64;
    constexpr uint32_t numTotalMantissa = getNumTotalMantissa<float>();
    uint64_t extraInMatissa = (bitSum >> numTotalMantissa);
    uint32_t numExtra;
    int numNorm = 0;
    uint32_t mask = 0;
    extraInMatissa = (exponent == 0) ? extraInMatissa : extraInMatissa + count;
    numExtra = numTotalBits - __clzll(extraInMatissa);
    numNorm = (exponent == 0) ? 0 : -1;
    exponent = exponent + ((numExtra + numNorm) << numTotalMantissa);
    uint32_t mantissa;
    if (extraInMatissa != 0)
    {
        int numMove = numTotalMantissa - (numExtra - 1);
        mask = (1u << (numExtra - 1)) - 1;
        extraInMatissa = extraInMatissa & mask;
        if (numMove > 0)
        {
            extraInMatissa = extraInMatissa << numMove;
            mask = (1u << numTotalMantissa) - 1;
            mantissa = ((bitSum & mask) >> (numExtra - 1)) | extraInMatissa;
        }
        else
        {
            mantissa = extraInMatissa >> (-1 * numMove);
        }
    }
    else
    {
        mantissa = bitSum;
    }
    uint32_t bitFloat = exponent | mantissa;
    return reinterpret_cast<float&>(bitFloat);
}

template <typename T, typename HisT, bool isDeterministic = false>
__device__ constexpr void calcAtomicAdd(HisT* dst, T value)
{
    if constexpr (isDeterministic)
    {
        uint32_t mantissa = calcMantissa(value);
        if constexpr (std::is_same_v<T, half>)
        {
            atomicAdd(dst, mantissa);
        }
        else
        {
            atomicAdd(reinterpret_cast<unsigned long long*>(dst), static_cast<HisT>(mantissa));
        }
    }
    else
    {
        if constexpr (std::is_same_v<T, half>)
        {
            atomicAdd(dst, __half2float(value));
        }
        else
        {
            atomicAdd(dst, value);
        }
    }
}

template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddleIn(T key, bool selectMin)
{
    auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
    bits = cub::Traits<T>::TwiddleIn(bits);
    if (!selectMin)
    {
        bits = ~bits;
    }
    return bits;
}

template <typename T>
__device__ T twiddleOut(typename cub::Traits<T>::UnsignedBits bits, bool selectMin)
{
    if (!selectMin)
    {
        bits = ~bits;
    }
    bits = cub::Traits<T>::TwiddleOut(bits);
    return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
__device__ int calcBucket(T x, int startBit, uint32_t mask, bool selectMin)
{
    static_assert(BitsPerPass <= sizeof(int) * 8 - 1, "BitsPerPass is too large that the result type could not be int");
    return (twiddleIn(x, selectMin) >> startBit) & mask;
}

template <typename T, typename IdxT>
__host__ __device__ IdxT calcBufLen(IdxT len)
{
    IdxT constexpr ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
    IdxT bufLen = len / (ratio * 8);

    bufLen = alignTo(bufLen, 256);
    return bufLen;
}

template <typename T, typename IdxT>
__host__ __device__ void setBufPointers(T const* in, IdxT const* inIdx, T* buf1, IdxT* idxBuf1, T* buf2, IdxT* idxBuf2,
    int pass, T const*& inBuf, IdxT const*& inIdxBuf, T*& outBuf, IdxT*& outIdxBuf)
{
    if (pass == 0)
    {
        inBuf = in;
        inIdxBuf = nullptr;
        outBuf = nullptr;
        outIdxBuf = nullptr;
    }
    else if (pass == 1)
    {
        inBuf = in;
        inIdxBuf = inIdx;
        outBuf = buf1;
        outIdxBuf = idxBuf1;
    }
    else if (pass % 2 == 0)
    {
        inBuf = buf1;
        inIdxBuf = idxBuf1;
        outBuf = buf2;
        outIdxBuf = idxBuf2;
    }
    else
    {
        inBuf = buf2;
        inIdxBuf = idxBuf2;
        outBuf = buf1;
        outIdxBuf = idxBuf1;
    }
}

template <typename T, typename IdxT, typename Func>
__device__ void vectorizedProcess(size_t threadRank, size_t numThreads, T const* in, IdxT len, Func f)
{
    int constexpr WARP_SIZE = 32;
    if constexpr (sizeof(T) >= sizeof(WideT))
    {
        for (IdxT i = threadRank; i < len; i += numThreads)
        {
            f(in[i], i);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        int constexpr itemsPerScalar = sizeof(WideT) / sizeof(T);

        union
        {
            WideT scalar;
            T array[itemsPerScalar];
        } wide;

        int skipCnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
        if (skipCnt > len)
        {
            skipCnt = len;
        }
        WideT const* inCast = reinterpret_cast<decltype(inCast)>(in + skipCnt);
        IdxT const lenCast = (len - skipCnt) / itemsPerScalar;

        for (IdxT i = threadRank; i < lenCast; i += numThreads)
        {
            wide.scalar = inCast[i];
            IdxT const real_i = skipCnt + i * itemsPerScalar;
#pragma unroll
            for (int j = 0; j < itemsPerScalar; ++j)
            {
                f(wide.array[j], real_i + j);
            }
        }

        static_assert(WARP_SIZE >= itemsPerScalar);
        if (threadRank < skipCnt)
        {
            f(in[threadRank], threadRank);
        }
        IdxT const remain_i = skipCnt + lenCast * itemsPerScalar + threadRank;
        if (remain_i < len)
        {
            f(in[remain_i], remain_i);
        }
    }
}

template <typename T, typename IdxT, typename AccT, typename HisT, int BitsPerPass, bool isDeterministic = false>
__device__ __forceinline__ void filterAndHistogram(T const* inBuf, IdxT const* inIdxBuf, T* outBuf, IdxT* outIdxBuf,
    int previousLen, Counter<T, IdxT, AccT>* counter, HisT* histogram, IdxT* countHistogram, HisT* histogramSmem,
    IdxT* countHistogramSmem, int pass, float* outputLogProbs, float* cumLogProbs, IdxT** ids, IdxT const* endIds,
    IdxT* sequenceLengths, FinishedState* finishedOutput, int const batchId, int maxBatchSize, bool earlyStop)
{
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");

    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    bool constexpr selectMin = false;

    for (IdxT i = threadIdx.x; i < numBuckets; i += blockDim.x)
    {
        histogramSmem[i] = 0;
        countHistogramSmem[i] = 0;
    }
    __syncthreads();

    int const startBit = calcStartBit<T, BitsPerPass>(pass);
    uint32_t const mask = calcMask<T, BitsPerPass>(pass);

    if (pass == 0)
    {
        auto f = [selectMin, startBit, mask, histogramSmem, countHistogramSmem](T value, IdxT)
        {
            int bucket = calcBucket<T, BitsPerPass>(value, startBit, mask, selectMin);
            calcAtomicAdd<T, HisT, isDeterministic>(histogramSmem + bucket, value);
            atomicAdd(countHistogramSmem + bucket, static_cast<IdxT>(1));
        };
        vectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(blockDim.x) * gridDim.x, inBuf, previousLen, f);
    }
    else
    {
        IdxT* pFilterCnt = &counter->filterCnt;
        auto const kthValueBits = counter->kthValueBits;
        int const previousStartBit = calcStartBit<T, BitsPerPass>(pass - 1);

        auto f = [inIdxBuf, outBuf, outIdxBuf, selectMin, startBit, mask, previousStartBit, kthValueBits, pFilterCnt,
                     histogramSmem, countHistogramSmem, outputLogProbs, cumLogProbs, ids, endIds, sequenceLengths,
                     finishedOutput, batchId, maxBatchSize, earlyStop](T value, IdxT i)
        {
            auto const previousBits = (twiddleIn(value, selectMin) >> previousStartBit) << previousStartBit;
            if (previousBits == kthValueBits)
            {
                if (earlyStop)
                {

                    int const currentStep = sequenceLengths ? sequenceLengths[batchId] : 0;
                    IdxT index = inIdxBuf ? inIdxBuf[i] : i;
                    ids[batchId][currentStep] = index;
                    float valueFloat;
                    if constexpr (std::is_same_v<T, half>)
                    {
                        valueFloat = __half2float(value);
                    }
                    else
                    {
                        valueFloat = value;
                    }
                    epilogue(valueFloat, index, outputLogProbs, cumLogProbs, endIds, sequenceLengths, finishedOutput,
                        batchId, maxBatchSize);
                }
                if (outBuf)
                {
                    IdxT pos = atomicAdd(pFilterCnt, static_cast<IdxT>(1));
                    outBuf[pos] = value;
                    outIdxBuf[pos] = inIdxBuf ? inIdxBuf[i] : i;
                }

                int bucket = calcBucket<T, BitsPerPass>(value, startBit, mask, selectMin);
                calcAtomicAdd<T, HisT, isDeterministic>(histogramSmem + bucket, value);
                atomicAdd(countHistogramSmem + bucket, static_cast<IdxT>(1));
            }
        };
        vectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(blockDim.x) * gridDim.x, inBuf, previousLen, f);
    }

    __syncthreads();
    if (earlyStop)
    {
        return;
    }

    for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
    {
        if (histogramSmem[i] != 0)
        {
            if constexpr ((isDeterministic) && (std::is_same_v<T, float>) )
            {
                atomicAdd(reinterpret_cast<unsigned long long*>(histogram + i), histogramSmem[i]);
            }
            else
            {
                atomicAdd(histogram + i, histogramSmem[i]);
            }
        }
        if (countHistogramSmem[i] != 0)
        {
            atomicAdd(countHistogram + i, countHistogramSmem[i]);
        }
    }
}

template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(IdxT volatile* histogram, IdxT* histogramOut)
{
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    if constexpr (numBuckets >= BlockSize)
    {
        static_assert(numBuckets % BlockSize == 0);
        int constexpr itemsPerThread = numBuckets / BlockSize;
        typedef cub::BlockLoad<IdxT, BlockSize, itemsPerThread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
        typedef cub::BlockStore<IdxT, BlockSize, itemsPerThread, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
        typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

        __shared__ union
        {
            typename BlockLoad::TempStorage load;
            typename BlockScan::TempStorage scan;
            typename BlockStore::TempStorage store;
        } tempStorage;

        IdxT threadData[itemsPerThread];

        BlockLoad(tempStorage.load).Load(histogram, threadData);
        __syncthreads();

        BlockScan(tempStorage.scan).InclusiveSum(threadData, threadData);
        __syncthreads();

        BlockStore(tempStorage.store).Store(histogramOut, threadData);
    }
    else
    {
        typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
        __shared__ typename BlockScan::TempStorage tempStorage;

        IdxT threadData = 0;
        if (threadIdx.x < numBuckets)
        {
            threadData = histogram[threadIdx.x];
        }

        BlockScan(tempStorage).InclusiveSum(threadData, threadData);
        __syncthreads();

        if (threadIdx.x < numBuckets)
        {
            histogramOut[threadIdx.x] = threadData;
        }
    }
}

template <typename T, typename IdxT>
__device__ void epilogue(T const value, IdxT const index, float* outputLogProbs, float* cumLogProbs, IdxT const* endIds,
    IdxT* sequenceLengths, FinishedState* finishedOutput, int const batchId, int maxBatchSize)
{
    if (outputLogProbs != nullptr || cumLogProbs != nullptr)
    {
        float res = logf(value);
        if (outputLogProbs)
        {
            auto const curLen = sequenceLengths ? sequenceLengths[batchId] : 0;
            outputLogProbs[curLen * maxBatchSize + batchId] = res;
        }
        if (cumLogProbs)
        {
            cumLogProbs[batchId] += res;
        }
    }
    if (endIds && index == endIds[batchId])
    {
        if (finishedOutput != nullptr)
        {
            finishedOutput[batchId].setFinishedEOS();
        }
    }
    else if (sequenceLengths != nullptr)
    {
        sequenceLengths[batchId] += 1;
    }
}

template <typename T, typename IdxT, typename AccT, int BitsPerPass, int BlockSize, bool isDeterministic = false>
__device__ void lastFilter(T const* inBuf, IdxT const* inIdxBuf, IdxT currentLen, Counter<T, IdxT, AccT>* counter,
    float* outputLogProbs, float* cumLogProbs, IdxT** ids, IdxT const* endIds, IdxT* sequenceLengths,
    FinishedState* finishedOutput, int const batchId, int maxBatchSize, IdxT* lastIdxBuf, IdxT* countHistogram)
{
    auto const kthValueBits = counter->kthValueBits;
    auto const equalValue = twiddleOut<T>(kthValueBits, false);
    int const currentStep = sequenceLengths ? sequenceLengths[batchId] : 0;
    IdxT* outIdx = &ids[batchId][currentStep];

    float equalValueFloat;
    if constexpr (std::is_same_v<T, half>)
    {
        equalValueFloat = __half2float(equalValue);
    }
    else
    {
        equalValueFloat = equalValue;
    }
    if constexpr (!isDeterministic)
    {

        for (IdxT i = threadIdx.x; i < currentLen; i += blockDim.x)
        {
            if (inBuf[i] == equalValue)
            {
                *outIdx = inIdxBuf ? inIdxBuf[i] : i;
                break;
            }
        }
    }
    else
    {
        IdxT const bufLen = calcBufLen<T>(counter->oriLen);
        IdxT neededNumOfKth = counter->sum > 0 ? ceil(counter->sum / equalValueFloat) : 1;

        if (counter->len < neededNumOfKth)
        {
            neededNumOfKth = counter->len;
        }

        if (neededNumOfKth < bufLen)
        {
            for (int i = threadIdx.x; i < neededNumOfKth; i += blockDim.x)
            {
                lastIdxBuf[i] = cuda::std::numeric_limits<IdxT>::max();
            }
            __threadfence_block();
            __syncthreads();

            cuda::atomic_ref<IdxT, cuda::thread_scope_block> refLast(lastIdxBuf[neededNumOfKth - 1]);

            for (IdxT i = threadIdx.x; i < currentLen; i += blockDim.x)
            {
                if (inBuf[i] == equalValue)
                {
                    IdxT newIdx = inIdxBuf ? inIdxBuf[i] : i;
                    if (newIdx < refLast.load(cuda::memory_order_relaxed))
                    {
                        for (int j = 0; j < neededNumOfKth; j++)
                        {
                            IdxT preIdx = atomicMin_block(&lastIdxBuf[j], newIdx);
                            if (preIdx > newIdx)
                            {
                                newIdx = preIdx;
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                *outIdx = refLast.load(cuda::memory_order_relaxed);
            }
        }
        else
        {
            int numPass = calcNumPasses<IdxT, BitsPerPass>();
            int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
            __shared__ typename cub::Traits<IdxT>::UnsignedBits kthValueBitsIdx;
            __shared__ IdxT neededNumOfKthSmem;
            if (threadIdx.x == 0)
            {
                kthValueBitsIdx = 0;
                neededNumOfKthSmem = neededNumOfKth;
            }
            __syncthreads();
            for (int pass = 0; pass < numPass; pass++)
            {
                for (IdxT i = threadIdx.x; i < numBuckets; i += blockDim.x)
                {
                    countHistogram[i] = 0;
                }
                __syncthreads();

                int preNeededNumOfKth = neededNumOfKthSmem;
                int const startBit = calcStartBit<IdxT, BitsPerPass>(pass);
                uint32_t const mask = calcMask<IdxT, BitsPerPass>(pass);
                for (IdxT j = threadIdx.x; j < currentLen; j += blockDim.x)
                {
                    if (inBuf[j] == equalValue)
                    {
                        IdxT newIdx = inIdxBuf ? inIdxBuf[j] : j;
                        bool isQualified = (pass == 0) ? true : false;
                        if (pass > 0)
                        {
                            int const previousStartBit = calcStartBit<IdxT, BitsPerPass>(pass - 1);
                            auto const previousBits = (twiddleIn(newIdx, true) >> previousStartBit) << previousStartBit;
                            if (previousBits == kthValueBitsIdx)
                            {
                                isQualified = true;
                            }
                        }
                        if (isQualified)
                        {
                            int bucket = calcBucket<IdxT, BitsPerPass>(newIdx, startBit, mask, true);
                            atomicAdd(countHistogram + bucket, static_cast<IdxT>(1));
                        }
                    }
                }
                __syncthreads();

                scan<IdxT, BitsPerPass, BlockSize>(countHistogram, countHistogram);
                __syncthreads();
                for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
                {
                    IdxT prev = (i == 0) ? 0 : countHistogram[i - 1];
                    IdxT cur = countHistogram[i];
                    if (prev < preNeededNumOfKth && preNeededNumOfKth <= cur)
                    {
                        neededNumOfKthSmem = neededNumOfKthSmem - prev;
                        typename cub::Traits<IdxT>::UnsignedBits bucket = i;
                        kthValueBitsIdx |= bucket << startBit;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                *outIdx = twiddleOut<IdxT>(kthValueBitsIdx, true);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        epilogue(equalValueFloat, *outIdx, outputLogProbs, cumLogProbs, endIds, sequenceLengths, finishedOutput,
            batchId, maxBatchSize);
    }
}

template <typename T, typename IdxT, typename AccT, typename HisT, int BitsPerPass, int BlockSize,
    bool isFusedFilter = false, bool isDeterministic = false>
__global__ void airTopPSampling(Counter<T, IdxT, AccT>* counters, HisT* histograms, IdxT* countHistograms, IdxT** ids,
    int* sequenceLengths, FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, IdxT const* endIds, int const maxBatchSize, bool const* skipDecode, int const pass, T* buf1,
    IdxT* idxBuf1, T* buf2, IdxT* idxBuf2, int32_t const* batchSlots)
{
    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");

    int const tid = threadIdx.x;
    int const batchId = blockIdx.y;
    auto const batchSlot = batchSlots ? batchSlots[batchId] : batchId;
    auto counter = counters + batchId;

    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchSlot] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchSlot]) || (finishState.isSkipDecoding()))
    {
        return;
    }

    if (finishState.isFinished())
    {
        if (pass == 0 && tid == 0)
        {
            if (finishedOutput != nullptr)
            {
                finishedOutput[batchSlot] = finishState;
            }
        }
        return;
    }

    AccT currentSum;
    IdxT previousLen;
    IdxT currentLen;

    if (pass == 0)
    {
        currentSum = 0;
        previousLen = counter->len;
        currentLen = counter->len;
    }
    else
    {
        currentSum = counter->sum;
        currentLen = counter->len;
        previousLen = counter->previousLen;
    }
    if (currentLen == 0)
    {
        return;
    }
    bool const earlyStop = (currentLen == 1);
    IdxT const bufLen = calcBufLen<T>(counter->oriLen);

    T const* inBuf = nullptr;
    IdxT const* inIdxBuf = nullptr;
    T* outBuf = nullptr;
    IdxT* outIdxBuf = nullptr;

    setBufPointers(counter->in, counter->inIdx, buf1 + bufLen * batchId, idxBuf1 + bufLen * batchId,
        buf2 + bufLen * batchId, idxBuf2 + bufLen * batchId, pass, inBuf, inIdxBuf, outBuf, outIdxBuf);

    if (pass == 0 || pass == 1 || previousLen > bufLen)
    {
        inBuf = counter->in;
        inIdxBuf = counter->inIdx;
        previousLen = counter->oriLen;
    }
    if (pass == 0 || currentLen > bufLen)
    {
        outBuf = nullptr;
        outIdxBuf = nullptr;
    }
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    auto histogram = histograms + batchId * numBuckets;
    auto countHistogram = countHistograms + batchId * numBuckets;
    __shared__ HisT histogramSmem[numBuckets];
    __shared__ IdxT countHistogramSmem[numBuckets];
    AccT* histValueSmem = reinterpret_cast<AccT*>(histogramSmem);

    filterAndHistogram<T, IdxT, AccT, HisT, BitsPerPass, isDeterministic>(inBuf, inIdxBuf, outBuf, outIdxBuf,
        previousLen, counter, histogram, countHistogram, histogramSmem, countHistogramSmem, pass, outputLogProbs,
        cumLogProbs, ids, endIds, sequenceLengths, finishedOutput, batchSlot, maxBatchSize, earlyStop);

    __syncthreads();
    __threadfence();

    bool isLastBlock = false;
    if (threadIdx.x == 0)
    {
        uint32_t finished = atomicInc(&counter->finishedBlockCnt, gridDim.x - 1);
        isLastBlock = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(isLastBlock))
    {
        if (earlyStop)
        {
            if (threadIdx.x == 0)
            {
                counter->previousLen = 0;
                counter->len = 0;
            }
            return;
        }

        if constexpr (isDeterministic)
        {
            for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
            {
                uint64_t value = (uint64_t) histogram[i];
                IdxT count = countHistogram[i];

                if (count != 0)
                {
                    uint32_t startBit = calcStartBit<T, BitsPerPass>(pass);
                    [[maybe_unused]] float bucketValueFloat;
                    if constexpr (std::is_same_v<T, half>)
                    {
                        uint16_t bucketValue = counter->kthValueBits;

                        if (pass == 0)
                        {
                            bucketValue = i << startBit;
                        }
                        uint32_t exponent = calcExponent(twiddleOut<T>(bucketValue, false));
                        uint32_t mask = (1u << (sizeof(half) * CHAR_BIT - 1)) - 1;
                        uint32_t sign = exponent & (~mask);
                        exponent = exponent & mask;
                        float tmp = calcHalfValue((uint32_t) count, exponent, sign, value);
                        histValueSmem[i] = tmp;
                    }
                    else
                    {
                        uint32_t bucketValue = counter->kthValueBits;
                        if (pass == 0)
                        {
                            bucketValue = i << startBit;
                        }
                        bucketValueFloat = twiddleOut<T>(bucketValue, false);
                        uint32_t exponent = calcExponent(bucketValueFloat);
                        histValueSmem[i] = calcFloatValue((uint32_t) count, exponent, value);
                    }
                }
                else
                {
                    histValueSmem[i] = 0.0f;
                }
            }
        }

        int constexpr WARP_SIZE = 32;
        int constexpr WARP_COUNT = numBuckets / WARP_SIZE;
        namespace cg = cooperative_groups;
        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        AccT* histPtr = isDeterministic ? histValueSmem : reinterpret_cast<AccT*>(histogram);
        __shared__ AccT warpSum[WARP_COUNT];
        __shared__ cuda::atomic<AccT, cuda::thread_scope_block> blockSum;
        if constexpr (BitsPerPass != 11)
        {
            for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
            {
                warpSum[i] = 0;
            }
            __syncthreads();
        }

        for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
        {
            reduce_store_async(warp, warpSum + i / WARP_SIZE, histPtr[i], cg::plus<float>{});
        }
        __syncthreads();

        if (threadIdx.x < WARP_SIZE)
        {
            reduce_store_async(warp, blockSum, warpSum[threadIdx.x], cg::plus<float>{});
            if constexpr (BitsPerPass == 11)
            {
                reduce_update_async(warp, blockSum, warpSum[threadIdx.x + WARP_SIZE], cg::plus<float>{});
            }
        }
        __syncthreads();

        if (pass == 0)
        {
            currentSum = blockSum * counter->p;
        }

        if (threadIdx.x == 0)
        {
            AccT prev = 0;

            int iStep = 0;
            int targetStep = 0;
            for (; iStep < WARP_COUNT; iStep++)
            {
                if (warpSum[iStep])
                {
                    targetStep = iStep;
                    if ((prev + warpSum[iStep]) >= currentSum)
                    {
                        break;
                    }
                    prev += warpSum[iStep];
                }
            }

            int targetIdx = 0;
            for (int i = targetStep * WARP_SIZE; i < numBuckets; i++)
            {
                if (countHistogram[i])
                {
                    targetIdx = i;
                    if ((prev + histPtr[i]) >= currentSum)
                    {
                        break;
                    }
                    prev += histPtr[i];
                }
            }

            counter->sum = currentSum - prev;
            counter->len = countHistogram[targetIdx];
            typename cub::Traits<T>::UnsignedBits bucket = targetIdx;
            int startBit = calcStartBit<T, BitsPerPass>(pass);
            counter->kthValueBits |= bucket << startBit;
        }
        __syncthreads();

        int constexpr numPasses = calcNumPasses<T, BitsPerPass>();
        if (pass != numPasses - 1)
        {
            for (int i = threadIdx.x; i < numBuckets; i += blockDim.x)
            {
                histogram[i] = 0;
                countHistogram[i] = 0;
            }
        }
        if (threadIdx.x == 0)
        {
            counter->previousLen = currentLen;
            counter->filterCnt = 0;
        }

        if (pass == numPasses - 1)
        {
            [[maybe_unused]] IdxT* lastIdxBuf
                = (pass % 2 == 0) ? idxBuf1 + bufLen * batchId : idxBuf2 + bufLen * batchId;
            if constexpr (isFusedFilter)
            {
                lastFilter<T, IdxT, AccT, BitsPerPass, BlockSize, isDeterministic>(outBuf ? outBuf : inBuf,
                    outIdxBuf ? outIdxBuf : inIdxBuf, outBuf ? currentLen : counter->oriLen, counter, outputLogProbs,
                    cumLogProbs, ids, endIds, sequenceLengths, finishedOutput, batchSlot, maxBatchSize, lastIdxBuf,
                    countHistogramSmem);
                __syncthreads();
            }
        }
    }
}

template <typename T, typename IdxT, typename AccT, typename HisT, int BitsPerPass, int BlockSize>
__global__ void airTopPInitialize(Counter<T, IdxT, AccT>* counters, int const batchSize, int const len, T const* in,
    IdxT const* inIdx, float const* topPs, curandState_t* curandState, float const* randomVals, HisT* histograms,
    IdxT* countHistograms, int32_t const* batchSlots)
{
    auto const batchIdx = blockIdx.x;
    auto const batchSlot = batchSlots ? batchSlots[batchIdx] : batchIdx;
    Counter<T, IdxT, AccT>* counter = counters + batchIdx;
    IdxT offset = batchIdx * len;
    IdxT bufOffset = batchIdx * calcBufLen<T>(len);
    if (threadIdx.x == 0)
    {
        counter->in = in + offset;
        counter->inIdx = nullptr;
        if (inIdx)
        {
            counter->inIdx = inIdx + offset;
        }

        counter->len = len;
        counter->oriLen = len;
        counter->previousLen = len;

        float const probThreshold = topPs[batchSlot];
        auto const randomNumber = randomVals ? randomVals[batchSlot] : curand_uniform(curandState + batchSlot);
        float const randP = randomNumber * probThreshold;
        counter->p = randP;
        counter->sum = 0;

        counter->kthValueBits = 0;
        counter->finishedBlockCnt = 0;
        counter->filterCnt = 0;
    }

    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    HisT* histogram = histograms + batchIdx * numBuckets;
    for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
    {
        histogram[i] = 0;
    }

    IdxT* countHistogram = nullptr;
    if (countHistograms)
    {
        countHistogram = countHistograms + batchIdx * numBuckets;
        for (int i = threadIdx.x; i < numBuckets; i += BlockSize)
        {
            countHistogram[i] = 0;
        }
    }
}

template <typename T>
uint32_t calcAirTopPBlockNum(int batchSize, int len, int smCnt, bool isDeterministic)
{
    int constexpr BitsPerPass = 11;
    int constexpr BlockSize = 512;
    int constexpr VECTORIZED_READ_SIZE = 16;
    static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);
    CHECK_WITH_INFO(
        smCnt > 0, "AIR Top-P needs the count of multiprocessor to calculate the proper block dimension settings");

    int activeBlocks;
    if (isDeterministic)
    {
        using HisT = std::conditional_t<std::is_same_v<T, float>, uint64_t, uint32_t>;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, airTopPSampling<T, IdxT, AccT, HisT, BitsPerPass, BlockSize, false, true>, BlockSize, 0);
    }
    else
    {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, airTopPSampling<T, IdxT, AccT, float, BitsPerPass, BlockSize, false, false>, BlockSize, 0);
    }
    activeBlocks *= smCnt;

    IdxT bestNumBlocks = 0;
    float bestTailWavePenalty = 1.0f;
    IdxT const maxNumBlocks = ceilDiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize);
    for (int numWaves = 1;; ++numWaves)
    {
        IdxT numBlocks = std::min(maxNumBlocks, static_cast<IdxT>(std::max(numWaves * activeBlocks / batchSize, 1)));
        IdxT itemsPerThread = ceilDiv<IdxT>(len, numBlocks * BlockSize);
        itemsPerThread = alignTo<IdxT>(itemsPerThread, VECTORIZED_READ_SIZE / sizeof(T));
        numBlocks = ceilDiv<IdxT>(len, itemsPerThread * BlockSize);
        float actualNumWaves = static_cast<float>(numBlocks) * batchSize / activeBlocks;
        float tailWavePenalty = (ceilf(actualNumWaves) - actualNumWaves) / ceilf(actualNumWaves);

        if (tailWavePenalty < 0.15)
        {
            bestNumBlocks = numBlocks;
            break;
        }
        else if (tailWavePenalty < bestTailWavePenalty)
        {
            bestNumBlocks = numBlocks;
            bestTailWavePenalty = tailWavePenalty;
        }

        if (numBlocks == maxNumBlocks)
        {
            break;
        }
    }
    return bestNumBlocks;
}

template <typename T, bool isDeterministic = false>
[[nodiscard]] std::vector<size_t> getAirTopPWorkspaceSizes(int32_t batchSize, int32_t vocabSize)
{
    using HisT
        = std::conditional_t<isDeterministic, std::conditional_t<std::is_same_v<T, float>, uint64_t, uint32_t>, float>;
    int constexpr BitsPerPass = 11;
    int constexpr numBuckets = calcNumBuckets<BitsPerPass>();
    IdxT const bufLen = calcBufLen<T>(vocabSize);

    size_t countersSize = sizeof(Counter<T, IdxT, AccT>) * batchSize;
    size_t histogramsSize = sizeof(HisT) * numBuckets * batchSize;
    size_t countHistogramsSize = sizeof(IdxT) * numBuckets * batchSize;
    size_t buf1Size = sizeof(T) * bufLen * batchSize;
    size_t idxBuf1Size = sizeof(IdxT) * bufLen * batchSize;
    size_t buf2Size = sizeof(T) * bufLen * batchSize;
    size_t idxBuf2Size = sizeof(IdxT) * bufLen * batchSize;

    std::vector<size_t> sizes
        = {countersSize, histogramsSize, countHistogramsSize, buf1Size, idxBuf1Size, buf2Size, idxBuf2Size};

    return sizes;
}

template std::vector<size_t> getAirTopPWorkspaceSizes<float, true>(int32_t batchSize, int32_t vocabSize);
template std::vector<size_t> getAirTopPWorkspaceSizes<float, false>(int32_t batchSize, int32_t vocabSize);
template std::vector<size_t> getAirTopPWorkspaceSizes<half, true>(int32_t batchSize, int32_t vocabSize);
template std::vector<size_t> getAirTopPWorkspaceSizes<half, false>(int32_t batchSize, int32_t vocabSize);

template <typename T, bool isDeterministic = false>
void invokeAirTopPSamplingWithDeterministicPara(TopPSamplingKernelParams<T> const& params, cudaStream_t stream)
{
    using HisT
        = std::conditional_t<isDeterministic, std::conditional_t<std::is_same_v<T, float>, uint64_t, uint32_t>, float>;

    static_assert(std::is_same_v<T, half> | std::is_same_v<T, float>, "T needs to be either half or float");
    static_assert(std::is_same_v<AccT, float>, "AccT needs to be float");
    CHECK_WITH_INFO(((std::is_same_v<T, half>) &&(params.vocabSizePadded < pow(2, 22)) && isDeterministic)
            || ((std::is_same_v<T, float>) &&(params.vocabSizePadded < pow(2, 41)) && isDeterministic)
            || (~isDeterministic),
        "For Deterministic AIR Top-P, the maximum vocab_size we support is pow(2,22) for half-precision and pow(2,41) "
        "for single-precision");

    IdxT const vocabSize = params.vocabSizePadded;
    int constexpr BitsPerPass = 11;

    int constexpr SAMPLING_BLOCK_SIZE = 512;
    int constexpr THREADS_PER_CTA_TOP_P_INIT = 1024;

    Counter<T, IdxT, AccT>* counters = nullptr;
    HisT* histograms = nullptr;
    IdxT* countHistograms = nullptr;
    T* buf1 = nullptr;
    IdxT* idxBuf1 = nullptr;
    T* buf2 = nullptr;
    IdxT* idxBuf2 = nullptr;

    auto const workspaceSizes = getAirTopPWorkspaceSizes<T, isDeterministic>(params.batchSize, vocabSize);
    calcAlignedPointers(params.workspace, workspaceSizes)(
        counters, histograms, countHistograms, buf1, idxBuf1, buf2, idxBuf2);

    airTopPInitialize<T, IdxT, AccT, HisT, BitsPerPass, THREADS_PER_CTA_TOP_P_INIT>
        <<<params.batchSize, THREADS_PER_CTA_TOP_P_INIT, 0, stream>>>(counters, params.batchSize, vocabSize,
            params.probs, nullptr, params.topPs, params.curandState, params.randomVals, histograms, countHistograms,
            params.batchSlots);

    dim3 grid(params.blockNum, params.batchSize);
    int constexpr numPasses = calcNumPasses<T, BitsPerPass>();
    auto kernel = airTopPSampling<T, IdxT, AccT, HisT, BitsPerPass, SAMPLING_BLOCK_SIZE, false, isDeterministic>;

    for (int pass = 0; pass < numPasses; ++pass)
    {
        if (pass == numPasses - 1)
        {
            kernel = airTopPSampling<T, IdxT, AccT, HisT, BitsPerPass, SAMPLING_BLOCK_SIZE, true, isDeterministic>;
        }

        kernel<<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(counters, histograms, countHistograms, params.outputIdsPtrs,
            params.sequenceLength, params.finishedInput, params.finishedOutput, params.cumLogProbs,
            params.outputLogProbs, params.endIds, params.maxBatchSize, params.skipDecode, pass, buf1, idxBuf1, buf2,
            idxBuf2, params.batchSlots);
    }
}

template <typename T>
void invokeBatchAirTopPSampling(TopPSamplingKernelParams<T> const& params, cudaStream_t stream)
{
    if (params.isDeterministic)
    {
        invokeAirTopPSamplingWithDeterministicPara<T, true>(params, stream);
    }
    else
    {
        invokeAirTopPSamplingWithDeterministicPara<T, false>(params, stream);
    }
}

template void invokeBatchAirTopPSampling(TopPSamplingKernelParams<float> const& params, cudaStream_t stream);

template void invokeBatchAirTopPSampling(TopPSamplingKernelParams<half> const& params, cudaStream_t stream);

template <typename T>
size_t getAirTopPWorkspaceSize(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic)
{
    std::vector<size_t> workspaceSizes;
    if (isDeterministic == true)
    {
        workspaceSizes = getAirTopPWorkspaceSizes<T, true>(batchSize, vocabSizePadded);
    }
    else
    {
        workspaceSizes = getAirTopPWorkspaceSizes<T, false>(batchSize, vocabSizePadded);
    }
    return calcAlignedSize(workspaceSizes, 256);
}

template size_t getAirTopPWorkspaceSize<float>(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic);
template size_t getAirTopPWorkspaceSize<half>(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic);

template uint32_t calcAirTopPBlockNum<float>(int batchSize, int len, int smCnt, bool isDeterministic);
template uint32_t calcAirTopPBlockNum<half>(int batchSize, int len, int smCnt, bool isDeterministic);
}
}
