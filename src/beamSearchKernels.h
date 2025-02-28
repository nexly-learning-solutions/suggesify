#pragma once

#include "decodingCommon.h"
#include "../runtime/common.h"

#include "topkLastDim.h"

namespace sugesstify
{
namespace kernels
{
static constexpr size_t nMaxBeamWidth = 1024;
static constexpr size_t nMaxBeamWidthForV1 = 8;
static constexpr size_t nThreadForSmallBeamWidth = 256;
static constexpr size_t nMaxVPartStage1 = 128;

struct BeamHypotheses
{

    bool bReturnNormedScore{false};
    size_t nMaxBatchSize{0};
    size_t nBatchSize{0};
    size_t nBeamWidth{0};
    size_t nMaxSeqLen{0};
    size_t nVocabSize{0};
    size_t nVPart{0};
    size_t nByteMaxSharedMemoryPerBlock{0};
    size_t nByteSharedMemoryStage1{0};
    size_t nByteSharedMemoryStage3{0};

    float const* diversityRates{nullptr};
    float const* lengthPenalties{nullptr};
    int const* earlyStoppings{nullptr};

    int const* inputLengths{nullptr};
    int const* endIds{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};

    int* outputIds{nullptr};
    float* logProbs{nullptr};
    float* logProbsTiled{nullptr};
    int* sequenceLengths{nullptr};
    float* cumLogProbs{nullptr};

    int* outputIdsCBA{nullptr};
    float* logProbsCBA{nullptr};
    int* sequenceLengthsCBA{nullptr};
    float* cumLogProbsCBA{nullptr};
    float* normedScoresCBA{nullptr};
    int* numBeamsCBA{nullptr};
    float* minNormedScoresCBA{nullptr};

    bool* batchDones{nullptr};
    FinishedState* finished{nullptr};

    int** outputIdsPtr{nullptr};
    int** parentIdsPtr{nullptr};

    int const* outputIdsUnfinish{nullptr};
    int const* parentIdsUnfinish{nullptr};

};

__inline__ int padToNextPowerOfTwo(int const n)
{
    int recursor = n - 1;
    int res = 2;
    while (recursor >>= 1)
        res <<= 1;
    return res;
}

template <typename T>
__device__ __forceinline__ T applyLengthPenalty(T const log_prob, int const length, float const length_penalty)
{
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf(static_cast<float>(length), length_penalty));
}

template <typename T, bool IS_V2>
void invokeTopkBeamSearch(T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template <typename T>
__global__ void addCumLogProbs(T* __restrict pStage1Probs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBM);

__global__ void gatherId(
    int const* __restrict pStage1Id, int* __restrict pStage2Id, size_t const nBS, size_t const nBM, size_t const nV);

}
}
