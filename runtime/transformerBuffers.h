
#pragma once

#include "bufferManager.h"
#include "common.h"
#include "generationConfig.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "tllmRuntime.h"
#include "worldConfig.h"

namespace suggestify::batch_manager::kv_cache_manager
{
class BaseKVCacheManager;
}

namespace suggestify::runtime
{

class RuntimeBuffers;

class TransformerBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using BaseKVCacheManager = batch_manager::kv_cache_manager::BaseKVCacheManager;
    using TensorMap = StringPtrMap<ITensor>;

    TransformerBuffers();

    TransformerBuffers(
        TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void reshape(
        GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq,
        runtime::TllmRuntime const& runtime);

    void setKvPoolPointers(BaseKVCacheManager const* kvCacheManager);
    void setKvPoolMapping(BaseKVCacheManager const* kvCacheManager);

    void reset(BufferManager& manager){};

    TransformerBuffers sliceTo(GenerationConfig const& generationConfig, ModelConfig const& modelConfig,
        SizeType32 offset, SizeType32 batchSize);

    void prepareContextStep(RuntimeBuffers* runtimeBuffers, TensorPtr const& inputIds, TokenIdType padId,
        BufferManager& manager, BaseKVCacheManager const* kvCacheManager, SizeType32 firstBatchSlotIdx,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
        BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void prepareNextStep(RuntimeBuffers* runtimeBuffers, SizeType32 step, BufferManager& manager,
        BaseKVCacheManager* kvCacheManager, SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
        SizeType32 step, TensorPtr const& inputIds, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

protected:
    void copyAttentionMasks(
        RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBatches, BufferManager& manager);

    void tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

public:
    TensorPtr pastKeyValueLengths;
    TensorPtr attentionMask;
    TensorPtr positionIds;

    std::vector<TensorPtr> presentKeysVals;
    std::vector<TensorPtr> presentKeysValsAlt;
    TensorPtr maxAttentionWindows;
    TensorPtr sinkTokenLengths;
    TensorPtr kvCacheBlockPoolPointers;
    TensorPtr kvCacheBlockPoolMapping;
    TensorPtr kvCacheBlockOffsetsHost;
    TensorPtr kvCacheBlockOffsetsDevice;
    TensorPtr runtimePerfKnobsHost;
    TensorPtr contextProgressHost;
};

}
