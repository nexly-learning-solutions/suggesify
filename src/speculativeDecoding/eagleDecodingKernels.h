
#pragma once

#include "../src/decodingCommon.h"
#include "../src/speculativeDecoding/common.h"
#include "../runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace sugesstify::kernels::speculative_decoding
{

template <typename T>
void invokeAssembleTargetLogitsOffsets(T const** logitsPtrs, runtime::SizeType32* decodingTokens, T const* logits,
    runtime::SizeType32 const* draftDecodingTokens, runtime::SizeType32 batchSize,
    runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 vocabSizePadded, cudaStream_t stream);

template <typename T>
void invokeAssembleDraftLogitsOffsets(T const** logitsPtrs, T const* logits, runtime::TokenIdType** outputIdsPtrs,
    runtime::TokenIdType* outputIds, bool* skipDecode, runtime::SizeType32 const* numValidLogits,
    runtime::SizeType32 numInputLogits, runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingDraftTokens,
    runtime::SizeType32 vocabSizePadded, cudaStream_t stream);

void invokePrepareCtxEagleNetInputs(runtime::SizeType32* eagleNetSequenceLengths,
    runtime::SizeType32* eagleNetContextLengths, runtime::TokenIdType* outputIds, runtime::SizeType32* positionIds,
    runtime::SizeType32* hiddenStatesIndices, runtime::SizeType32* lastTokenIndices,
    runtime::SizeType32* numLastTokenIndices, runtime::SizeType32* hiddenSizeBatchLevelStarts,
    runtime::TokenIdType const* inputIds, runtime::TokenIdType const* chunkedContextNextTokens,
    runtime::SizeType32 const* baseNetSequenceLengths, runtime::SizeType32 const* baseNetContextLengths,
    runtime::TokenIdType const* acceptedTokens, runtime::SizeType32 const* acceptedLens,
    runtime::SizeType32 const* prevDraftLens, runtime::SizeType32 const* prevPaths,
    runtime::SizeType32 const* bestPathIds, runtime::SizeType32 batchSize, runtime::SizeType32 maxPathLen,
    runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 maxNonLeavesPerLayer, cudaStream_t stream);

struct PrepareGenEagleNetInputsParams
{
    runtime::SizeType32* nextSequenceLengths{nullptr};
    runtime::SizeType32* nextContextLengths{nullptr};
    runtime::TokenIdType* outputIds{nullptr};
    runtime::SizeType32* positionIds{nullptr};
    runtime::SizeType32* specDecodingGenLengths{nullptr};
    runtime::SizeType32* specDecodingPositionOffsets{nullptr};
    runtime::SizeType32* specDecodingPackedMasks{nullptr};
    runtime::SizeType32* hiddenStatesIndices{nullptr};
    runtime::SizeType32* lastTokenIndices{nullptr};
    runtime::SizeType32* numLastTokenIndices{nullptr};
    runtime::SizeType32* outputHiddenSizeBatchStartsPerLevel{nullptr};

    int8_t* isLeafMask{nullptr};
    runtime::SizeType32* selectedDraftIndices{nullptr};
    runtime::SizeType32* selectedDraftPosOffsets{nullptr};
    runtime::SizeType32* numSelectedDraftIndices{nullptr};
    bool* selectedMasks{nullptr};
    runtime::SizeType32* cumSumGenerationLengths{nullptr};
    runtime::SizeType32* maxGenerationLength{nullptr};
    runtime::SizeType32* nonLeavesInLevelOffsets{nullptr};
    runtime::SizeType32* parentNonLeafInLevelOffset{nullptr};

    runtime::TokenIdType const* nextDraftIds{nullptr};
    runtime::SizeType32 const* eagleNet0SequenceLengths{nullptr};
    runtime::SizeType32 const* prevContextLengths{nullptr};
    runtime::SizeType32 const* nextPaths{nullptr};
    runtime::SizeType32 const* inputHiddenSizeBatchStartsPerLevel{nullptr};

    runtime::SizeType32 levelIdx{0};
    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 maxPathLen{0};
    runtime::SizeType32 maxDecodingTokens{0};
    runtime::SizeType32 maxNonLeavesPerLayer{0};
    cudaStream_t stream;

    void checkParams()
    {
        CHECK(nextSequenceLengths);
        CHECK(nextContextLengths);
        CHECK(outputIds);
        CHECK(positionIds);
        CHECK(specDecodingGenLengths);
        CHECK(specDecodingPositionOffsets);
        CHECK(specDecodingPackedMasks);
        CHECK(hiddenStatesIndices);
        CHECK(lastTokenIndices);
        CHECK(numLastTokenIndices);
        CHECK(outputHiddenSizeBatchStartsPerLevel);

        CHECK(isLeafMask);
        CHECK(selectedDraftIndices);
        CHECK(selectedDraftPosOffsets);
        CHECK(numSelectedDraftIndices);
        CHECK(selectedMasks);
        CHECK(cumSumGenerationLengths);
        CHECK(maxGenerationLength);
        CHECK(nonLeavesInLevelOffsets);
        CHECK(parentNonLeafInLevelOffset);

        CHECK(nextDraftIds);
        CHECK(eagleNet0SequenceLengths);
        CHECK(prevContextLengths);
        CHECK(nextPaths);
        CHECK(inputHiddenSizeBatchStartsPerLevel);

        CHECK(batchSize > 0);
        CHECK(maxPathLen > 0);
        CHECK(maxDecodingTokens > 0);
        CHECK(0 < levelIdx && levelIdx < maxPathLen - 1);
        CHECK(maxNonLeavesPerLayer > 0);
    }
};

void invokePrepareGenEagleNetInputs(PrepareGenEagleNetInputsParams const& params);

struct PackEagleParams
{
    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 maxNumPaths{0};
    runtime::SizeType32 maxDecodingTokens{0};
    runtime::SizeType32 maxPathLength{0};
    runtime::SizeType32 numContextRequests{0};
    runtime::SizeType32 numGenerationRequests{0};

    runtime::SizeType32 const* batchSlots{nullptr};

    float const* inputTemperatures{nullptr};
    float const* inputRandomDataSample{nullptr};
    float const* inputRandomDataValidation{nullptr};
    runtime::TokenIdType const* inputNextDraftTokens{nullptr};
    runtime::SizeType32 const* inputNextDraftPaths{nullptr};
    runtime::SizeType32 const* inputSpecDecodingGenerationLengths{nullptr};
    runtime::SizeType32 const* inputSpecDecodingPositionOffsets{nullptr};
    int32_t const* inputSpecDecodingPackedMasks{nullptr};

    float* outputTemperatures{nullptr};
    float* outputRandomDataSample{nullptr};
    float* outputRandomDataValidation{nullptr};
    runtime::TokenIdType* outputNextDraftTokens{nullptr};
    runtime::SizeType32* outputNextDraftLens{nullptr};
    runtime::SizeType32* outputNextDraftPaths{nullptr};
    runtime::SizeType32* outputSpecDecodingGenerationLengths{nullptr};
    runtime::SizeType32* outputSpecDecodingPositionOffsets{nullptr};
    int32_t* outputSpecDecodingPackedMasks{nullptr};

    runtime::SizeType32* maxGenerationLength{nullptr};
    runtime::SizeType32* cumSumGenerationLengths{nullptr};

    void checkParams()
    {
        CHECK(batchSlots);

        CHECK(inputTemperatures);
        CHECK(inputRandomDataSample);
        CHECK(inputRandomDataValidation);
        CHECK(inputNextDraftTokens);
        CHECK(inputNextDraftPaths);
        CHECK(inputSpecDecodingGenerationLengths);
        CHECK(inputSpecDecodingPositionOffsets);
        CHECK(inputSpecDecodingPackedMasks);

        CHECK(outputTemperatures);
        CHECK(outputRandomDataSample);
        CHECK(outputRandomDataValidation);
        CHECK(outputNextDraftTokens);
        CHECK(outputNextDraftLens);
        CHECK(outputNextDraftPaths);
        CHECK((numGenerationRequests > 0 && outputSpecDecodingGenerationLengths) || numGenerationRequests == 0);
        CHECK((numGenerationRequests > 0 && outputSpecDecodingPositionOffsets) || numGenerationRequests == 0);
        CHECK((numGenerationRequests > 0 && outputSpecDecodingPackedMasks) || numGenerationRequests == 0);

        CHECK(maxGenerationLength);
        CHECK(cumSumGenerationLengths);

        CHECK(batchSize > 0);
        CHECK(batchSize == numContextRequests + numGenerationRequests);
        CHECK(maxDecodingTokens > 0);
        CHECK(maxPathLength > 0);
        CHECK(maxNumPaths > 0);
    }
};

void invokePackEagleGenerationLengths(PackEagleParams const& params, cudaStream_t stream);
void invokePackEagle(PackEagleParams const& params, cudaStream_t stream);

struct UnpackEagleDataParams
{
    runtime::SizeType32 const* batchSlots{nullptr};
    curandState_t* inputCurandState{nullptr};
    float const* inputTemperatures{nullptr};
    runtime::TokenIdType const* inputNextDraftTokens{nullptr};
    runtime::SizeType32 const* inputNextDraftLens{nullptr};
    runtime::SizeType32 const* inputNextDraftPaths{nullptr};
    runtime::TokenIdType const* inputLastDraftTokens{nullptr};
    runtime::SizeType32 const* inputLastDraftLens{nullptr};
    runtime::TokenIdType const* inputAcceptedTokens{nullptr};
    runtime::SizeType32 const* inputAcceptedLens{nullptr};

    runtime::TokenIdType* outputIds{nullptr};
    runtime::SizeType32* outputNumNewTokens{nullptr};
    runtime::SizeType32* outputSequenceLengths{nullptr};
    runtime::TokenIdType* outputUnpackedNextDraftTokens{nullptr};
    runtime::TokenIdType* outputNextDraftTokens{nullptr};
    runtime::SizeType32* outputNextDraftLengths{nullptr};
    runtime::SizeType32* outputNextDraftPaths{nullptr};
    runtime::SizeType32* outputPrevDraftLengths{nullptr};
    runtime::SizeType32* outputNextGenerationLength{nullptr};
    runtime::SizeType32* outputPositionIds{nullptr};

    float* outputRandDataSample{nullptr};
    float* outputRandDataVerification{nullptr};
    float* outputTemperatures{nullptr};

    runtime::SizeType32* outputEagleNetCtxRequestTypes{nullptr};
    runtime::SizeType32* outputEagleNetCtxContextLengths{nullptr};
    runtime::SizeType32* outputEagleNetCtxPastKeyValueLengths{nullptr};
    runtime::SizeType32* outputEagleNetGenRequestTypes{nullptr};
    runtime::SizeType32* outputEagleNetGenContextLengths{nullptr};
    runtime::SizeType32* outputEagleNetGenPastKeyValueLengths{nullptr};

    runtime::SizeType32 batchSize{0};
    runtime::SizeType32 maxDecodingTokens{0};
    runtime::SizeType32 maxPathLength{0};
    runtime::SizeType32 maxSeqLen{0};

    void checkParams()
    {
        CHECK(batchSlots);
        CHECK(inputCurandState);
        CHECK(inputTemperatures);
        CHECK(inputNextDraftTokens);
        CHECK(inputNextDraftLens);
        CHECK(inputNextDraftPaths);
        CHECK(inputLastDraftTokens);
        CHECK(inputLastDraftLens);
        CHECK(inputAcceptedTokens);
        CHECK(inputAcceptedLens);

        CHECK(outputIds);
        CHECK(outputNumNewTokens);
        CHECK(outputSequenceLengths);
        CHECK(outputUnpackedNextDraftTokens);
        CHECK(outputNextDraftTokens);
        CHECK(outputNextDraftLengths);
        CHECK(outputNextDraftPaths);
        CHECK(outputPrevDraftLengths);
        CHECK(outputNextGenerationLength);
        CHECK(outputPositionIds);

        CHECK(outputRandDataSample);
        CHECK(outputRandDataVerification);
        CHECK(outputTemperatures);

        CHECK(outputEagleNetCtxRequestTypes);
        CHECK(outputEagleNetCtxContextLengths);
        CHECK(outputEagleNetCtxPastKeyValueLengths);
        CHECK(outputEagleNetGenRequestTypes);
        CHECK(outputEagleNetGenContextLengths);
        CHECK(outputEagleNetGenPastKeyValueLengths);

        CHECK(batchSize > 0);
        CHECK(maxDecodingTokens > 0);
        CHECK(maxPathLength > 0);
        CHECK(maxSeqLen > 0);
    }
};

void invokeUnpackEagleData(UnpackEagleDataParams const& params, cudaStream_t stream);

struct FillContextEagleParams
{
    float* outputRandDataSample{nullptr};
    float* outputTemperatures{nullptr};

    float const* inputTemperatures{nullptr};
    curandState_t* inputCurandState{nullptr};
    runtime::SizeType32 const* batchSlots{nullptr};

    runtime::SizeType32 batchSize{0};

    void checkParams()
    {
        CHECK(outputRandDataSample);
        CHECK(outputTemperatures);

        CHECK(inputTemperatures);
        CHECK(inputCurandState);
        CHECK(batchSlots);

        CHECK(batchSize > 0);
    }
};

void invokeFillContextEagleData(FillContextEagleParams const& params, cudaStream_t stream);

void invokeGetPackedMaskFromPath(int32_t* specDecodingPackedMasks, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 const* nextDraftPaths, runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingTokens,
    runtime::SizeType32 maxPathLen, cudaStream_t stream);

void invokeExtractTopKsFromPath(runtime::SizeType32 const* paths, runtime::SizeType32* topKs,
    runtime::SizeType32* topKOffset, runtime::SizeType32* numSuccessorsForEachNode, runtime::SizeType32 layerId,
    runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 maxPathLen,
    cudaStream_t stream);

void invokeCopyOutputTokensIds(runtime::TokenIdType** tmpOutputIdsPtrs, runtime::SizeType32 const* topKs,
    runtime::SizeType32 const* topKOffset, runtime::TokenIdType const* pluginInputDraftIdsPtrs,
    runtime::SizeType32 const* pluginInputDraftLens, runtime::SizeType32 const* numValidLogits,
    runtime::TokenIdType* pluginOutputDraftIdsPtrs, runtime::SizeType32* pluginOutputDraftLens,
    runtime::SizeType32 layerId, runtime::SizeType32 batchSize, runtime::SizeType32 maxDecodingDraftTokens,
    cudaStream_t stream);

void invokeAugmentBatchSlots(runtime::SizeType32* augmentedSeqSlots, runtime::SizeType32* augmentedBatchSlots,
    runtime::SizeType32 const* chunkedContextNextTokens, runtime::SizeType32 const* lastDraftLens,
    runtime::SizeType32 const* seqSlots, runtime::SizeType32 const* batchSlots, runtime::SizeType32 engineBatchSize,
    runtime::SizeType32 batchSize, cudaStream_t stream);

void invokeSetTopKsFromDyanmicTreeMaxTopK(runtime::SizeType32 layerIdx, runtime::SizeType32 batchSize,
    runtime::SizeType32 numInputLogits, runtime::SizeType32* topKs, runtime::SizeType32* topKOffset,
    runtime::SizeType32 const dynamicTreeMaxTopK, cudaStream_t stream);

void invokeCopyScoresAndDraftTokenIds(runtime::SizeType32 layerIdx, runtime::SizeType32 mNumEagleLayers,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 batchSize, runtime::SizeType32 numInputLogits,
    runtime::SizeType32 const dynamicTreeMaxTopK, runtime::SizeType32* topKOffset,
    runtime::TokenIdType const* pluginInputCurrentExpandIndices, float const* pluginInputAllLayersScores,
    runtime::TokenIdType const* pluginInputAllLayersDraftTokenIds,
    runtime::TokenIdType const* pluginInputAllLayersDraftTokenIdsPredecessor, float* pluginOutputAllLayersScores,
    runtime::TokenIdType* pluginOutputAllLayersDraftTokenIds,
    runtime::TokenIdType* pluginOutputAllLayersDraftTokenIdsPredecessor, float* firstTopKOutputLogProbs,
    runtime::TokenIdType* firstTopKOutputIdsPtrs, cudaStream_t stream);

void invokeUpdateScores(runtime::SizeType32 batchSize, runtime::SizeType32 numInputLogits,
    runtime::SizeType32 const dynamicTreeMaxTopK, runtime::SizeType32 maxDecodingDraftTokens, float* curLogProbs,
    float const* prevLayerScores, cudaStream_t stream);

void invokeAssembleSecondTopKSamplingInputs(runtime::SizeType32 batchSize, runtime::SizeType32 const dynamicTreeMaxTopK,
    runtime::SizeType32 maxDecodingDraftTokens, float* firstTopKOutputLogProbs, float** secondTopKInputScoresPtrs,
    runtime::TokenIdType* secondTopKOutputIdsFlatten, runtime::TokenIdType** secondTopKOutputIdsPtrs,
    cudaStream_t stream);

void invokeUpdatePath(runtime::SizeType32 layerIdx, runtime::SizeType32 batchSize,
    runtime::SizeType32 dynamicTreeMaxTopK, runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 maxPathLen,
    runtime::SizeType32 const* prevPaths, runtime::SizeType32* newPaths, runtime::TokenIdType** secondTopKOutputIdsPtrs,
    runtime::TokenIdType* pluginOutputNextExpandIndices, cudaStream_t stream);

void invokeUpdateDraftTokensAndLensAndCurScores(runtime::SizeType32 layerIdx, runtime::SizeType32 batchSize,
    runtime::SizeType32 dynamicTreeMaxTopK, runtime::SizeType32 maxDecodingDraftTokens,
    runtime::TokenIdType** curDraftIds, runtime::TokenIdType const* pluginInputDraftIds,
    runtime::SizeType32 const* pluginInputDraftLens, runtime::TokenIdType* pluginOutputDraftIds,
    runtime::SizeType32* pluginOutputDraftLens, float const* curLayerScores, float* pluginOutputCurrentScores,
    cudaStream_t stream);

void invokeExtractScoresAndRealDraftTokensIds(runtime::SizeType32 batchSize, runtime::SizeType32 dynamicTreeMaxTopK,
    runtime::SizeType32 maxDecodingDraftTokens, float** secondTopKInputScoresPtrs,
    runtime::TokenIdType** secondTopKOutputIdsPtrs, runtime::TokenIdType* firstTopKOutputIds,
    float* secondTopKOutputLogProbs, cudaStream_t stream);

void invokeAssembleThridTopKSamplingInputs(runtime::SizeType32 batchSize, runtime::SizeType32 const dynamicTreeMaxTopK,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 mNumEagleLayers, float* pluginOutputAllLayersScores,
    float** thirdTopKInputScoresPtrs, runtime::TokenIdType* thirdTopKOutputIds,
    runtime::TokenIdType** thirdTopKOutputIdsPtrs, cudaStream_t stream);

void invokeReconstructFinalPath(runtime::SizeType32 batchSize, runtime::SizeType32 const dynamicTreeMaxTopK,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 maxPathLen,
    runtime::SizeType32 mNumEagleLayers, runtime::TokenIdType** thirdTopKOutputIdsPtrs,
    runtime::TokenIdType* pluginOutputAllLayersDraftTokenIdsPredecessor, runtime::SizeType32* newPaths,
    cudaStream_t stream);

void invokeCopyFinalDraftTokens(runtime::SizeType32 batchSize, runtime::SizeType32 const dynamicTreeMaxTopK,
    runtime::SizeType32 maxDecodingDraftTokens, runtime::SizeType32 mNumEagleLayers,
    runtime::TokenIdType** thirdTopKOutputIdsPtrs, runtime::TokenIdType* pluginOutputAllLayersDraftTokenIds,
    runtime::TokenIdType* pluginOutputDraftTokenIds, runtime::SizeType32* pluginOutputDraftLens, cudaStream_t stream);

}
