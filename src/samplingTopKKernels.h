#pragma once

#include "memoryUtils.h"
#include "suggestify/kernels/decodingCommon.h"
#include "suggestify/runtime/common.h"
#include <curand_kernel.h>

namespace suggestify::kernels
{

static constexpr runtime::SizeType32 TOP_K_MAX = 1024;

template <typename T>
struct TopKSamplingKernelParams
{

    T const* logProbs{nullptr};
    T const* const* logProbsPtrs{nullptr};

    runtime::TokenIdType** outputIdsPtrs{nullptr};
    runtime::TokenIdType* outputIds{nullptr};

    void* workspace{nullptr};

    runtime::TokenIdType const* endIds{nullptr};

    runtime::SizeType32* sequenceLengths{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};
    runtime::SizeType32 const* tokensPerStep{nullptr};

    FinishedState const* finishedInput{nullptr};
    FinishedState* finishedOutput{nullptr};
    bool const* skipDecode{nullptr};

    float* cumLogProbs{nullptr};
    float* outputLogProbs{nullptr};

    curandState_t* curandState{nullptr};
    runtime::SizeType32 const* topKs{nullptr};
    float const* topPs{nullptr};
    runtime::SizeType32 maxTopK{TOP_K_MAX};
    float maxTopP{1.0f};

    runtime::SizeType32 batchSize{-1};
    runtime::SizeType32 maxBatchSize{-1};
    runtime::SizeType32 vocabSizePadded{-1};
    runtime::SizeType32 maxTokensPerStep{-1};
    runtime::SizeType32 maxSeqLen{-1};

    bool normalizeLogProbs{false};
    bool logitsHasProbs{false};
    bool returnAllSelectedTokens{false};

    runtime::TokenIdType* outputIdCurrentStep{nullptr};
    bool const* skipOutputIdCurrentStep{nullptr};

    void checkParams() const
    {
        CHECK(batchSize > 0);
        CHECK(maxBatchSize > 0);
        CHECK(maxBatchSize >= batchSize);
        CHECK(vocabSizePadded > 0);
        CHECK(maxTokensPerStep > 0);

        CHECK(logProbs || logProbsPtrs);
        CHECK(outputIds || outputIdsPtrs);

        if (maxTokensPerStep > 1)
        {
            CHECK(tokensPerStep);
        }

        if (outputIds)
        {
            CHECK(maxSeqLen > 0);
        }

        CHECK(workspace);

        CHECK(maxTokensPerStep != 1 || returnAllSelectedTokens || sequenceLengths);
        CHECK(maxTokensPerStep != 1 || returnAllSelectedTokens || endIds);
        if (cumLogProbs != nullptr || outputLogProbs != nullptr)
        {
            CHECK(maxTokensPerStep == 1);
            if (cumLogProbs != nullptr)
            {
                CHECK(!returnAllSelectedTokens);
            }
        }

        CHECK(((finishedOutput == nullptr) ^ (endIds == nullptr)) == 0);

        CHECK(0 < maxTopP && maxTopP <= 1.f);
        CHECK(0 <= maxTopK && maxTopK <= TOP_K_MAX);
        CHECK((skipOutputIdCurrentStep && outputIdCurrentStep && returnAllSelectedTokens)
            || (skipOutputIdCurrentStep == nullptr && outputIdCurrentStep == nullptr));
    }
};

template <typename T>
void invokeBatchTopKSampling(TopKSamplingKernelParams<T> const& params, cudaStream_t stream);

template <typename T>
[[nodiscard]] std::vector<size_t> getTopKWorkspaceSizes(runtime::SizeType32 batchSize,
    runtime::SizeType32 maxTokensPerStep, runtime::SizeType32 maxTopK, runtime::SizeType32 vocabSizePadded)
{
    runtime::SizeType32 constexpr maxBlockPerBeam = 8;
    auto const tempLogProbsBufSize = sizeof(T) * batchSize * maxTokensPerStep * vocabSizePadded;
    auto const topKTmpIdsBufSize
        = sizeof(runtime::SizeType32) * batchSize * maxTokensPerStep * maxTopK * maxBlockPerBeam;
    auto const topKTmpValBufSize = sizeof(T) * batchSize * maxTokensPerStep * maxTopK * maxBlockPerBeam;

    return {tempLogProbsBufSize, topKTmpIdsBufSize, topKTmpValBufSize};
}

[[nodiscard]] inline std::vector<size_t> getTopKInitWorkspaceSizes(runtime::SizeType32 batchSize)
{
    auto const tempTopKsBufSize = batchSize * sizeof(runtime::SizeType32);
    auto const tempTopPsBufSize = batchSize * sizeof(float);

    return {tempTopKsBufSize, tempTopPsBufSize};
}

template <typename T>
[[nodiscard]] size_t getTopKWorkspaceSize(runtime::SizeType32 batchSize, runtime::SizeType32 maxTokensPerStep,
    runtime::SizeType32 maxTopK, runtime::SizeType32 vocabSizePadded)
{
    auto const workspaceSizes = getTopKWorkspaceSizes<T>(batchSize, maxTokensPerStep, maxTopK, vocabSizePadded);
    auto const initWorkspaceSizes = getTopKInitWorkspaceSizes(batchSize);
    return std::max(suggestify::common::calcAlignedSize(workspaceSizes, 256),
        suggestify::common::calcAlignedSize(initWorkspaceSizes, 256));
}

void invokeSetupTopKRuntimeArgs(runtime::SizeType32 batchSize, ScatterDecodingParamEntry<runtime::SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodePtr, runtime::SizeType32 const* batchSlotsPtr, bool onDevice,
    cudaStream_t stream = nullptr);

void invokeSetupTopKTopPRuntimeArgs(runtime::SizeType32 batchSize, ScatterDecodingParamEntry<runtime::SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodeTopKPtr, bool* skipDecodeTopPPtr,
    runtime::SizeType32 const* batchSlotsPtr, bool onDevice, cudaStream_t stream = nullptr);

inline bool clampTopP(float& topP)
{
    if (topP < 0.f || topP > 1.0f)
    {
        LOG_WARNING("TopP (%f) is out of range ([0.0, 1.0f]). Clip to closest number.", topP);
        topP = std::clamp(topP, 0.f, 1.f);
        return true;
    }

    return false;
}

inline bool clampTopK(runtime::SizeType32& topK)
{
    if (topK < 0 || topK > TOP_K_MAX)
    {
        LOG_WARNING(
            "TopK (%d) is larger than max supported number (%d). Clip to max supported number.", topK, TOP_K_MAX);
        topK = std::clamp(topK, 0, TOP_K_MAX);
        return true;
    }

    return false;
}

inline bool regularizeTopKTopP(runtime::SizeType32& topK, float& topP)
{
    bool modified = false;
    if (topK == 0 && topP == 0.0f)
    {
        topK = 1;
        modified = true;
    }
    if (topK > 0 && topP == 0.0f)
    {
        topP = 1.0f;
        modified = true;
    }

    return modified;
}

__device__ __host__ inline void setupTopKTopPRuntimeArgOne(runtime::SizeType32 batchIndex,
    ScatterDecodingParamEntry<runtime::SizeType32> topK, ScatterDecodingParamEntry<float> topP,
    runtime::SizeType32 const* batchSlots, bool* skipDecodeTopK, bool* skipDecodeTopP, float* initialTopPBuf)
{
    auto const batchSlot = batchSlots[batchIndex];
    auto const k = topK.mVector == nullptr ? topK.mScalar : topK.mVector[batchIndex];
    auto const p = topP.mVector == nullptr ? topP.mScalar : topP.mVector[batchIndex];
    if (topK.mTarget != nullptr)
    {
        topK.mTarget[batchSlot] = k;
    }
    if (topP.mTarget != nullptr)
    {
        topP.mTarget[batchSlot] = p;
    }
    if (skipDecodeTopK != nullptr)
    {
        skipDecodeTopK[batchSlot] = k == 0;
    }
    if (skipDecodeTopP != nullptr)
    {
        skipDecodeTopP[batchSlot] = k != 0;
    }
    if (initialTopPBuf != nullptr)
    {
        initialTopPBuf[batchSlot] = p;
    }
}

}
