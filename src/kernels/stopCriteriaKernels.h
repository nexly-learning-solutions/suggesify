#pragma once

#include "../src/decodingCommon.h"
#include "../runtime/common.h"
#include <cuda_runtime.h>

namespace sugesstify
{
namespace kernels
{
void invokeStopWordsCriterion(runtime::TokenIdType const** outputIds, runtime::SizeType32 const** parentIds,
    runtime::TokenIdType const* const* stopWords, FinishedState* finished, runtime::SizeType32* sequenceLengths,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 const* stopWordsLen, runtime::SizeType32* numNewTokens,
    runtime::SizeType32 maxStopWordsLen, runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth,
    runtime::SizeType32 maxSeqLen, cudaStream_t stream);

void invokeLengthCriterion(FinishedState* finished, runtime::SizeType32* finishedSum,
    runtime::SizeType32 const* sequenceLimitLength, runtime::SizeType32* sequenceLengths,
    runtime::SizeType32* numNewTokens, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize,
    runtime::SizeType32 beamWidth, cudaStream_t stream);

void invokeExplicitEOSCriterion(runtime::TokenIdType const** outputIds, runtime::TokenIdType const* endIds,
    FinishedState* finished, runtime::SizeType32* sequenceLengths, runtime::SizeType32* numNewTokens,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth,
    runtime::SizeType32 maxTokensPerStep, cudaStream_t stream);
}
}
