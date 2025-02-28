
#pragma once

#include "../src/decodingCommon.h"
#include "../src/speculativeDecoding/common.h"
#include "../runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace sugesstify::kernels::speculative_decoding
{

void scatterMedusaDraftTokens(runtime::TokenIdType* treeDraftIds, runtime::TokenIdType const* sourceDraftIds,
    runtime::SizeType32 const* treeIds, runtime::SizeType32 const* tokensPerStep, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 maxDecodingTokens, runtime::SizeType32 batchSize, cudaStream_t stream);

}
