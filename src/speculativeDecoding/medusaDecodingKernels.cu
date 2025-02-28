

#include "../common/assert.h"
#include "../common/cudaTypeUtils.cuh"
#include "../common/cudaUtils.h"
#include "../common/memoryUtils.h"
#include "../common/reduceKernelUtils.cuh"
#include "../src/speculativeDecoding/medusaDecodingKernels.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace sugesstify::common;
using namespace sugesstify::runtime;

namespace sugesstify::kernels::speculative_decoding
{
namespace
{
__global__ void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds,
    SizeType32 const* treeIds, SizeType32 const* tokensPerStepData, SizeType32 const* batchSlots,
    SizeType32 maxDecodingTokens)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchIdx];
    auto const tokensPerStep = tokensPerStepData[batchSlot];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    for (auto index = static_cast<SizeType32>(threadIdx.x); index < tokensPerStep - 1;
         index += static_cast<SizeType32>(blockDim.x))
    {
        auto const indexInTree = treeIds[batchSlot * maxDecodingDraftTokens + index];
        auto const treeDraftIdx = batchSlot * maxDecodingDraftTokens + index;
        auto const sourceDraftIdx = batchSlot * maxDecodingTokens + indexInTree;
        treeDraftIds[treeDraftIdx] = sourceDraftIds[sourceDraftIdx];
    }
}
} // namespace

void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds, SizeType32 const* treeIds,
    SizeType32 const* tokensPerStep, SizeType32 const* batchSlots, SizeType32 maxDecodingTokens, SizeType32 batchSize,
    cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 256;
    scatterMedusaDraftTokens<<<batchSize, BLOCK_SIZE, 0, stream>>>(
        treeDraftIds, sourceDraftIds, treeIds, tokensPerStep, batchSlots, maxDecodingTokens);
}
} // namespace sugesstify::kernels::speculative_decoding
