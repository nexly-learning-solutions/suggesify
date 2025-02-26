#pragma once

#include "assert.h"
#include "suggestify/kernels/decodingCommon.h"
#include "common.h"
#include <curand_kernel.h>

namespace suggestify::kernels
{
template <typename T>
struct TopPSamplingKernelParams
{
    T const* probs{nullptr};

    runtime::TokenIdType** outputIdsPtrs{nullptr};

    runtime::TokenIdType* outputIds{nullptr};

    void* workspace{nullptr};

    float const* topPs{nullptr};

    runtime::SizeType32* sequenceLength{nullptr};
    runtime::TokenIdType const* endIds{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};

    FinishedState const* finishedInput{nullptr};
    FinishedState* finishedOutput{nullptr};
    bool const* skipDecode{nullptr};

    float* cumLogProbs{nullptr};
    float* outputLogProbs{nullptr};
    curandState_t* curandState{nullptr};
    float const* randomVals{nullptr};

    runtime::SizeType32 blockNum{-1};
    bool isDeterministic{true};

    runtime::SizeType32 batchSize{-1};
    runtime::SizeType32 maxBatchSize{-1};
    runtime::SizeType32 vocabSizePadded{-1};
    runtime::SizeType32 maxSeqLen{-1};

    bool returnAllSelectedTokens{false};

    runtime::TokenIdType* outputIdCurrentStep{nullptr};
    bool const* skipOutputIdCurrentStep{nullptr};

    void checkParams() const
    {
        CHECK(batchSize > 0);
        CHECK(maxBatchSize > 0);
        CHECK(maxBatchSize >= batchSize);
        CHECK(vocabSizePadded > 0);
        CHECK(probs);
        CHECK(outputIds || outputIdsPtrs);
        CHECK(workspace);
        CHECK((curandState != nullptr) || (randomVals != nullptr));
        CHECK(((curandState != nullptr) & (randomVals != nullptr)) == 0);
        CHECK(topPs);

        if (outputIds)
        {
            CHECK(maxSeqLen > 0);
        }

        CHECK(((finishedOutput == nullptr) ^ (endIds == nullptr)) == 0);
        CHECK((skipOutputIdCurrentStep && outputIdCurrentStep && returnAllSelectedTokens)
            || (skipOutputIdCurrentStep == nullptr && outputIdCurrentStep == nullptr));
    }
};

template <typename T>
[[nodiscard]] size_t getTopPWorkspaceSize(runtime::SizeType32 batchSize, runtime::SizeType32 vocabSizePadded);

[[nodiscard]] std::vector<size_t> getTopPInitWorkspaceSizes(runtime::SizeType32 batchSize);

template <typename T>
void invokeBatchTopPSampling(TopPSamplingKernelParams<T> const& params, cudaStream_t stream);

void invokeComputeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, runtime::TokenIdType const** outputIds,
    float const* topPDecay, float const* topPMin, runtime::TokenIdType const* topPResetIds,
    runtime::SizeType32 const* sequenceLengths, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 localBatchSize, cudaStream_t stream);

template <typename T>
void invokeBatchAirTopPSampling(TopPSamplingKernelParams<T> const& params, cudaStream_t stream);

template <typename T>
uint32_t calcAirTopPBlockNum(int batchSize, int len, int smCnt, bool isDeterministic = false);

template <typename T>
[[nodiscard]] size_t getAirTopPWorkspaceSize(int32_t batchSize, int32_t vocabSizePadded, bool isDeterministic = false);

void invokeSetTopPRuntimeArgs(runtime::SizeType32 batchSize, ScatterDecodingParamEntry<runtime::SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodePtr, float* initialTopPPtr,
    runtime::SizeType32 const* batchSlotsPtr, bool onDevice, cudaStream_t stream = nullptr);

}
