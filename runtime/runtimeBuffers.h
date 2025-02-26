
#pragma once

#include "bufferManager.h"
#include "generationConfig.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "promptTuningParams.h"
#include "rnnStateBuffers.h"
#include "transformerBuffers.h"
#include "worldConfig.h"

#include <array>
#include <vector>

namespace suggestify::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace suggestify::runtime
{
class TllmRuntime;

class RuntimeBuffers
{
protected:
    using TensorPtr = ITensor::SharedPtr;
    using BaseKVCacheManager = batch_manager::kv_cache_manager::BaseKVCacheManager;

public:
    using TensorMap = StringPtrMap<ITensor>;

public:
    GenerationConfig generationConfig{};
    std::array<TensorMap, 2> inputBuffers{};
    std::array<TensorMap, 2> outputBuffers{};

    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;

    TensorPtr logits;
    TensorPtr sequenceLengths;
    TensorPtr lastTokenIds;
    TensorPtr requestTypes;
    TensorPtr allGenerationLogits;
    TensorPtr originalLogitsPtr;

    TensorPtr newTokens;
    TensorPtr outputIds;
    TensorPtr outputLengths;

    TensorPtr cacheIndirectionDecoderInput;
    TensorPtr cacheIndirectionDecoderOutput;

    TensorPtr nbFinished;

    TensorPtr cumLogProbs;
    TensorPtr logProbs;

    TensorPtr hiddenStates;

    std::optional<TransformerBuffers> transformerBuffers;

    PromptTuningParams promptTuningParams;
    TensorPtr promptTuningTasksHost;

    std::optional<RnnStateBuffers> rnnStateBuffers;

    std::shared_ptr<std::vector<TensorPtr>> generationLogitsFragments;
    TensorPtr
        cacheGenerationFragmentPointerDevice;
    TensorPtr
        cacheGenerationFragmentPointerHost;

    bool allocated{false};

public:
    void clear();
    void clearTensorMaps();

    void create(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void initFromInput(ITensor const& inputIds, TensorPtr const& inputLengths, bool inputPacked, SizeType32 beamWidth,
        std::vector<SizeType32> maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength,
        SizeType32 maxSequenceLength, BufferManager& manager);

    void reshape(ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    std::vector<RuntimeBuffers> split(
        SizeType32 contextBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void prepareContextStep(TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager,
        BaseKVCacheManager const* kvCacheManager, SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);
    TensorPtr prepareNextStep(SizeType32 step, BufferManager& manager, BaseKVCacheManager* kvCacheManager,
        SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType32 const step,
        TensorPtr const& inputIds, TensorPtr const& commPtrs, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

    void gatherLastTokenLogits(BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig);
};

}
