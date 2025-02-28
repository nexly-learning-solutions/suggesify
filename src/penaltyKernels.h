#pragma once

#include <cuda_fp16.h>

#include "../src/decodingCommon.h"
#include "../runtime/common.h"

namespace suggestify::kernels
{

template <typename T>
struct InvokeBatchApplyPenaltyParams
{
    T const* const* inputLogits;
    T* outputLogits;
    T const* biases;
    runtime::TokenIdType* penaltyWorkspace;
    runtime::TokenIdType const* penaltyWorkspacePrev;
    float const* temperatures;
    float const* repetitionPenalties;
    float const* presencePenalties;
    float const* frequencyPenalties;
    runtime::SizeType32 batchSize;
    runtime::SizeType32 beamWidth;
    runtime::SizeType32 maxSeqLen;
    runtime::SizeType32 vocabSize;
    runtime::SizeType32 vocabSizePadded;
    runtime::TokenIdType const** outputIdsPtr;
    runtime::SizeType32 const** parentIdsPtr;
    runtime::SizeType32 const* inputLengths;
    runtime::SizeType32 const* sequenceLengths;
    runtime::SizeType32 const* minLengths;
    runtime::TokenIdType const* endIds;
    runtime::SizeType32 const* batchSlots;
    runtime::SizeType32 maxTokensPerStep;
    runtime::SizeType32 const* tokensPerStep;
    FinishedState const* finished;
    cudaStream_t stream;
};

template <typename T>
void invokeBatchApplyPenalty(InvokeBatchApplyPenaltyParams<T> const& params);

}
