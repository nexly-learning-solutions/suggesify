
#pragma once

#include "../src/kvCacheUtils.h"
#include "../runtime/common.h"
#include <cstdint>
#include <cuda_runtime_api.h>

namespace sugesstify::kernels::speculative_decoding
{

using IndexType = int;

void updateLinearKVCacheDraftTokenLocationCommonRewind(runtime::SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, runtime::SizeType32 const* pastKeyValueLengths,
    KVLinearBuffer::DataType* const* pastKeyValueList, runtime::SizeType32 layerCount, runtime::SizeType32 seqCount,
    runtime::SizeType32 numKVHeads, runtime::SizeType32 sizeInBytesPerKVHead, runtime::SizeType32 rewindDraftTokenCount,
    runtime::SizeType32 const* seqSlotRemapping, runtime::SizeType32 maxKVCacheLen, cudaStream_t stream);

void updateKVBlockArrayDraftTokenLocationCommonRewind(runtime::SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, runtime::SizeType32 const* pastKeyValueLengths,
    void* const* pointerArray, KVBlockArray::DataType* offsetArray, runtime::SizeType32 layerCount,
    runtime::SizeType32 seqCount, runtime::SizeType32 numKVHeads, runtime::SizeType32 sizeInBytesPerKVHead,
    runtime::SizeType32 rewindDraftTokenCount, runtime::SizeType32 const* seqSlotRemapping,
    runtime::SizeType32 maxKVCacheLen, runtime::SizeType32 maxBlocksPerSeq, runtime::SizeType32 tokensPerBlock,
    bool canUseOneMoreBlock, cudaStream_t stream);

void updateLinearKVCacheDraftTokenLocationSeparateRewind(runtime::SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, runtime::SizeType32 const* pastKeyValueLengths,
    KVLinearBuffer::DataType* const* pastKeyValueList, runtime::SizeType32 layerCount, runtime::SizeType32 seqCount,
    runtime::SizeType32 numKVHeads, runtime::SizeType32 sizeInBytesPerKVHead,
    runtime::SizeType32* rewindDraftTokenCounts, runtime::SizeType32 const* seqSlotRemapping,
    runtime::SizeType32 maxKVCacheLen, cudaStream_t stream);

void updateKVBlockArrayDraftTokenLocationSeparateRewind(runtime::SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, runtime::SizeType32 const* pastKeyValueLengths,
    void* const* pointerArray, KVBlockArray::DataType* offsetArray, runtime::SizeType32 layerCount,
    runtime::SizeType32 seqCount, runtime::SizeType32 numKVHeads, runtime::SizeType32 sizeInBytesPerKVHead,
    runtime::SizeType32* rewindDraftTokenCounts, runtime::SizeType32 const* seqSlotRemapping,
    runtime::SizeType32 maxKVCacheLen, runtime::SizeType32 maxBlocksPerSeq, runtime::SizeType32 tokensPerBlock,
    bool canUseOneMoreBlock, cudaStream_t stream);

void updateLinearKVCacheDraftTokenLocation(runtime::SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, runtime::SizeType32 const* pastKeyValueLengths,
    KVLinearBuffer::DataType* const* pastKeyValueList, runtime::SizeType32 layerCount, runtime::SizeType32 seqCount,
    runtime::SizeType32 numKVHeads, runtime::SizeType32 sizeInBytesPerKVHead,
    runtime::SizeType32 rewindDraftTokenCommonCount, runtime::SizeType32 const* rewindDraftTokenSeparateAdjustments,
    runtime::SizeType32 const* seqSlotRemapping, runtime::SizeType32 maxKVCacheLen, cudaStream_t stream);

void updateKVBlockArrayDraftTokenLocation(runtime::SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, runtime::SizeType32 const* pastKeyValueLengths,
    void* const* pointerArray, KVBlockArray::DataType* offsetArray, runtime::SizeType32 layerCount,
    runtime::SizeType32 seqCount, runtime::SizeType32 numKVHeads, runtime::SizeType32 sizeInBytesPerKVHead,
    runtime::SizeType32 rewindDraftTokenCommonCount, runtime::SizeType32 const* rewindDraftTokenSeparateAdjustments,
    runtime::SizeType32 const* seqSlotRemapping, runtime::SizeType32 const* batchSlots,
    runtime::SizeType32 maxKVCacheLen, runtime::SizeType32 maxBlocksPerSeq, runtime::SizeType32 tokensPerBlock,
    bool canUseOneMoreBlock, cudaStream_t stream);

}
