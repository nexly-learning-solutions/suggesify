
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
void invokeAcceptDraftTokens(runtime::SizeType32 batchSize, T* draftProbs, T* targetProbs,
    runtime::SizeType32 const* numsDraftTokens, bool const* batchUseDraftLogits, runtime::TokenIdType const* draftIds,
    FinishedState const* finishedInput, FinishedState* finishedOutput, curandState_t* curandState,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 maxDraftTokens, runtime::SizeType32 beamWidth,
    runtime::SizeType32 vocabSizePadded, bool randomThreshold, float constantThreshold, runtime::SizeType32 step,
    bool* batchIsAccepted, runtime::SizeType32* targetOutputIds, cudaStream_t stream);

template <typename T>
void invokeMaskTargetLogits(runtime::SizeType32 batchSize, T* targetLogits, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 beamWidth, runtime::SizeType32 vocabSizePadded, FinishedState const* finishedInput,
    runtime::SizeType32 maxBatchSize, runtime::SizeType32* outputIdsAfterSampling,
    runtime::SizeType32* runtimeTopKDevicePtr, bool* maskBuffer, cudaStream_t stream);

void invokeForwardAcceptedTokens(runtime::SizeType32 batchSize, runtime::SizeType32 const* batchSlots,
    bool* batchIsAccepted, runtime::SizeType32* outputSequenceLengths, runtime::TokenIdType const* draftIds,
    runtime::TokenIdType** idsPtrs, runtime::SizeType32 step, runtime::SizeType32 maxDraftTokens,
    runtime::TokenIdType const* endIds, FinishedState* finishedOutput, cudaStream_t stream);

}
