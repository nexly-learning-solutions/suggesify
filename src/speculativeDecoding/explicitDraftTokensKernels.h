
#pragma once

#include "../common/assert.h"
#include "../runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace sugesstify::kernels::speculative_decoding
{

template <typename T>
struct FillRandDataExplicitDraftTokensParams
{
    T* randDataSample{nullptr};
    T* randDataVerification{nullptr};
    curandState_t* curandState{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};

    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 numPaths{0};
    runtime::SizeType32 draftLength{0};

    bool skipVerification{false};

    void checkParams() const
    {
        CHECK(randDataSample);
        CHECK(randDataVerification);
        CHECK(curandState);

        CHECK(batchSize > 0);
        CHECK(numPaths > 0);
        CHECK(draftLength > 0);
    }
};

template <typename T>
void invokeFillRandData(FillRandDataExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

template <typename T>
struct FillContextExplicitDraftTokensParams
{
    T* randDataSample{nullptr};
    T* outputTemperatures{nullptr};
    float const* inputTemperatures{nullptr};
    curandState_t* curandState{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};

    runtime::SizeType32 batchSize{0};

    void checkParams() const
    {
        CHECK(randDataSample);
        CHECK(outputTemperatures);
        CHECK(inputTemperatures);
        CHECK(curandState);
        CHECK(batchSlots);

        CHECK(batchSize > 0);
    }
};

template <typename T>
void invokeFillContextBuffers(FillContextExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

template <typename T>
struct ExtractExplicitDraftTokensParams
{
    runtime::TokenIdType* outputIds{nullptr};
    runtime::SizeType32* outputPositionIdsBase{nullptr};
    runtime::SizeType32* outputPositionIds{nullptr};
    runtime::TokenIdType* outputNextDraftTokens{nullptr};
    runtime::TokenIdType* unpackedNextDraftTokens{nullptr};
    runtime::SizeType32* unpackedNextDraftIndices{nullptr};
    runtime::SizeType32* acceptedLengths{nullptr};
    runtime::SizeType32* prevDraftLengths{nullptr};
    runtime::SizeType32* nextDraftLengths{nullptr};
    runtime::SizeType32* sequenceLengths{nullptr};
    runtime::SizeType32* outputGenerationLengths{nullptr};
    runtime::SizeType32* outputBestPathIndices{nullptr};
    runtime::SizeType32* outputLastDraftIndices{nullptr};
    T* randDataSample{nullptr};
    T* randDataVerification{nullptr};
    T* outputDraftProbs{nullptr};
    T* outputTemperatures{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};
    runtime::TokenIdType const* nextDraftTokens{nullptr};
    runtime::TokenIdType const* lastDraftTokens{nullptr};
    runtime::SizeType32 const* inputUnpackedNextDraftIndices{nullptr};
    runtime::SizeType32 const* bestPathLengths{nullptr};
    runtime::SizeType32 const* bestPathIndices{nullptr};
    runtime::SizeType32 const* inputPositionIdsBase{nullptr};
    runtime::SizeType32 const* packedPositionIds{nullptr};
    runtime::TokenIdType const* nextFlatTokens{nullptr};
    runtime::SizeType32 const* generationLengthInclusiveSum{nullptr};
    runtime::SizeType32 const* lastGenerationLengths{nullptr};
    runtime::SizeType32 const* lastDraftIndices{nullptr};
    T const* nextDraftProbs{nullptr};
    float const* inputTemperatures{nullptr};
    curandState_t* curandState{nullptr};
    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 numPaths{0};
    runtime::SizeType32 maxPathLength{0};
    runtime::SizeType32 maxSeqLen{0};
    runtime::SizeType32 vocabSize{0};
    runtime::SizeType32 numContextRequests{0};
    runtime::SizeType32 numGenerationRequests{0};

    void checkParams() const
    {
        CHECK(outputIds);

        CHECK(outputPositionIdsBase);
        CHECK(inputPositionIdsBase);

        CHECK(outputPositionIds);
        CHECK(packedPositionIds);

        CHECK(outputTemperatures);
        CHECK(inputTemperatures);

        CHECK(outputDraftProbs);
        CHECK(nextDraftProbs);

        CHECK(outputNextDraftTokens);
        CHECK(unpackedNextDraftTokens);

        CHECK(unpackedNextDraftIndices);
        CHECK(inputUnpackedNextDraftIndices);

        CHECK(outputLastDraftIndices);

        CHECK(bestPathIndices);
        CHECK(outputBestPathIndices);

        CHECK(curandState);
        CHECK(batchSlots);
        CHECK(nextDraftTokens);
        CHECK(nextFlatTokens);
        CHECK(generationLengthInclusiveSum);
        CHECK(bestPathLengths);

        CHECK(randDataSample);
        CHECK(randDataVerification);
        CHECK(acceptedLengths);
        CHECK(nextDraftLengths);
        CHECK(prevDraftLengths);
        CHECK(sequenceLengths);
        CHECK(outputGenerationLengths);

        CHECK(batchSize > 0);
        CHECK(numPaths > 0);
        CHECK(maxPathLength > 0);
        CHECK(maxSeqLen > 0);
        CHECK(vocabSize > 0);
        CHECK(numContextRequests >= 0);
        CHECK(numGenerationRequests >= 0);
        CHECK(numContextRequests + numGenerationRequests != 0);
    }
};

template <typename T>
void invokeExtractExplicitDraftTokens(ExtractExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

template <typename T>
void invokeCopyProbs(ExtractExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

template <typename T>
struct PackExplicitDraftTokensParams
{
    runtime::SizeType32 const* batchSlots{nullptr};
    runtime::SizeType32 const* cumSumGenerationLengths{nullptr};
    runtime::SizeType32 const* maxGenerationLength{nullptr};

    runtime::SizeType32* outputPositionIdsBase{nullptr};
    runtime::SizeType32 const* inputPositionIdsBase{nullptr};

    runtime::SizeType32* outputGenerationLengths{nullptr};
    runtime::SizeType32 const* inputGenerationLengths{nullptr};

    T* outputRandomDataSample{nullptr};
    T const* inputRandomDataSample{nullptr};

    T* outputRandomDataValidation{nullptr};
    T const* inputRandomDataValidation{nullptr};

    runtime::TokenIdType* outputNextDraftTokens{nullptr};
    runtime::TokenIdType const* inputNextDraftTokens{nullptr};

    runtime::SizeType32* outputNextDraftIndices{nullptr};
    runtime::SizeType32 const* inputNextDraftIndices{nullptr};

    int32_t* outputPackedMask{nullptr};
    int32_t const* inputPackedMask{nullptr};

    runtime::SizeType32* outputPositionIds{nullptr};
    runtime::SizeType32* outputPositionOffsets{nullptr};
    runtime::SizeType32 const* inputPositionIds{nullptr};

    T* outputDraftProbs{nullptr};
    T const* inputDraftProbs{nullptr};

    T* outputTemperatures{nullptr};
    T const* inputTemperatures{nullptr};

    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 numPaths{0};
    runtime::SizeType32 maxPathLength{0};
    runtime::SizeType32 vocabSize{0};
    runtime::SizeType32 numContextTokens{0};
    runtime::SizeType32 numContextRequests{0};
    runtime::SizeType32 numGenerationRequests{0};

    void checkParams() const
    {
        CHECK(batchSlots);
        CHECK(cumSumGenerationLengths);
        CHECK(maxGenerationLength);

        CHECK(inputPositionIdsBase);

        CHECK(inputGenerationLengths);

        CHECK(outputRandomDataSample);
        CHECK(inputRandomDataSample);

        CHECK(inputRandomDataValidation);

        CHECK(inputNextDraftTokens);

        CHECK(inputNextDraftIndices);

        CHECK(inputPackedMask);

        CHECK(inputPositionIds);

        CHECK(inputDraftProbs);

        CHECK(outputTemperatures);
        CHECK(inputTemperatures);

        CHECK(batchSize > 0);
        CHECK(numPaths > 0);
        CHECK(maxPathLength > 0);
        CHECK(vocabSize > 0);
        CHECK(numContextRequests >= 0);
        CHECK(numGenerationRequests >= 0);
        CHECK(
            (numContextTokens == 0 && numContextRequests == 0) || (numContextTokens > 0 && numContextRequests > 0));
        CHECK(numContextRequests + numGenerationRequests != 0);
    }
};

template <typename T>
void invokePackGenerationLengths(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

template <typename T>
void invokePackExplicitDraftTokens(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

template <typename T>
void invokeCopyProbs(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream);

size_t invokeScanGenerationLengths(void* __restrict__ scanTempStorage, size_t scanTempStorageBytes,
    runtime::SizeType32 const* __restrict__ generationLengths,
    runtime::SizeType32* __restrict__ scannedGenerationLengths, runtime::SizeType32 batchSize, cudaStream_t stream);
size_t invokeReduceMaxGenerationLengths(void* __restrict__ reduceMaxTempStorage, size_t reduceTempStorageBytes,
    runtime::SizeType32 const* __restrict__ generationLengths, runtime::SizeType32* __restrict__ maxGenerationLengths,
    runtime::SizeType32 batchSize, cudaStream_t stream);

void invokeScanReduceGenerationLengths(runtime::SizeType32 batchSize,
    runtime::SizeType32 const* __restrict__ generationLengths, void* __restrict__ scanTempStorage,
    size_t scanTempStorageBytes, runtime::SizeType32* __restrict__ scanedGenerationLengths,
    void* __restrict__ reduceMaxTempStorage, size_t reduceMaxTempStorageBytes,
    runtime::SizeType32* maxGenerationLengths, cudaStream_t stream);

void invokeConvertMaskToPackedMask(runtime::SizeType32 batchSize,
    runtime::SizeType32 const* __restrict__ cumGenerationLengths,
    runtime::SizeType32 const* __restrict__ maxGenerationLengths, bool const* __restrict__ mask,
    runtime::SizeType32 const* __restrict__ batchSlots, runtime::SizeType32 maxDraftTokens,
    runtime::SizeType32 maxGenerationLength, runtime::SizeType32* __restrict__ packedMask, cudaStream_t stream);

}
