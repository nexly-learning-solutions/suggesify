
#pragma once

#include "../common/assert.h"
#include "../src/decodingCommon.h"
#include "../runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace sugesstify::kernels::speculative_decoding
{

void invokePackAcceptedPaths(runtime::SizeType32* acceptedLengthsCumSum, runtime::SizeType32* pathsOffsets,
    runtime::SizeType32 const* acceptedLengths, runtime::SizeType32 const* bestPathIds,
    runtime::SizeType32 const* paths, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize,
    runtime::SizeType32 engineBatchSize, runtime::SizeType32 numPaths, runtime::SizeType32 maxPathLen,
    bool isPathsLinearBatchIdx, cudaStream_t stream);

template <typename T>
struct AcceptDraftTokensByIdsWithPathsParams
{
    runtime::TokenIdType* outputIds{nullptr};
    runtime::TokenIdType const* draftIds{nullptr};
    runtime::TokenIdType const* targetIds{nullptr};
    runtime::SizeType32* sequenceLengths{nullptr};
    runtime::SizeType32* acceptedLengths{nullptr};
    FinishedState* finishedFinal{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};
    runtime::SizeType32 const* paths{nullptr};
    runtime::TokenIdType const* endIds{nullptr};
    T const** medusaLogits{nullptr};
    T const** logitsPtrs{nullptr};
    runtime::SizeType32* curTokensPerStep{nullptr};
    runtime::SizeType32 const* targetTokensPerStep{nullptr};
    runtime::SizeType32* bestPathIds{nullptr};
    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 maxBatchSize{0};
    runtime::SizeType32 vocabSize{0};
    runtime::SizeType32 maxSeqLen{0};
    runtime::SizeType32 maxDraftPathLen{0};
    runtime::SizeType32 maxDecodingTokens{0};
    cudaStream_t stream;

    void checkParams() const
    {
        CHECK(outputIds);
        CHECK(draftIds);
        CHECK(targetIds);
        CHECK(acceptedLengths);
        CHECK(paths);
        CHECK(bestPathIds);
        CHECK((curTokensPerStep == nullptr) ^ (targetTokensPerStep == nullptr) == 0);
        CHECK((medusaLogits == nullptr) ^ (logitsPtrs == nullptr) == 0);

        CHECK(batchSize > 0);
        CHECK(batchSize <= maxBatchSize);
        CHECK(vocabSize > 0);
        CHECK(maxSeqLen > 0);
        CHECK(maxDraftPathLen > 0);
        CHECK(maxDecodingTokens > 0);
    }
};

template <typename T>
void acceptDraftTokensByIdsWithPaths(AcceptDraftTokensByIdsWithPathsParams<T> const&);

template <typename T>
struct TypicalAcceptanceSampling
{
    T** logitsPtrs{nullptr};

    runtime::SizeType32 const* batchSlots{nullptr};

    runtime::SizeType32 const* generationLengths{nullptr};
    float const* temperatures{nullptr};
    float const* posteriorThresholds{nullptr};
    float const* posteriorAlphas{nullptr};

    runtime::TokenIdType* outputIds{nullptr};

    int8_t* workspace{nullptr};

    curandState_t* curandStats{nullptr};
    float const* randomVals{nullptr};

    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 maxBatchSize{0};
    runtime::SizeType32 maxDecodingTokens{0};
    runtime::SizeType32 vocabSize{0};
    runtime::SizeType32 smCnt{0};

    void checkParams()
    {
        CHECK(logitsPtrs);

        CHECK(generationLengths);
        CHECK(temperatures);
        CHECK(posteriorThresholds);
        CHECK(posteriorAlphas);
        CHECK(outputIds);
        CHECK(workspace);

        CHECK((curandStats != nullptr) || (randomVals != nullptr));
        CHECK(((curandStats != nullptr) & (randomVals != nullptr)) == 0);

        CHECK(batchSize > 0);
        CHECK(maxBatchSize > 0);
        CHECK(vocabSize > 0);
        CHECK(maxDecodingTokens > 0);
        CHECK(smCnt > 0);
    }
};

template <typename T>
void typicalAcceptanceSampling(TypicalAcceptanceSampling<T> const&, cudaStream_t);

template <typename T>
size_t getTypicalAcceptanceWorkspaceSize(
    runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 vocabSizePadded);

}
