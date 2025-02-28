
#pragma once

#include "bufferManager.h"
#include "common.h"
#include "loraCache.h"
#include "loraModule.h"
#include "modelConfig.h"
#include "worldConfig.h"
#include <unordered_map>

namespace suggestify::runtime
{

class LoraManager
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using ReqIdsVec = std::vector<uint64_t>;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using LoraWeightsTensorPtr = TensorPtr;
    using LoraConfigTensorPtr = TensorPtr;
    using LoraReqTensors = std::tuple<LoraWeightsTensorPtr, LoraConfigTensorPtr>;
    using TaskIdType = std::int64_t;
    using PeftValues = std::vector<runtime::LoraCache::TaskLayerModuleConfig>;
    using PeftTable = std::map<uint64_t, std::vector<runtime::LoraCache::TaskLayerModuleConfig>>;

    explicit LoraManager() {}

    void create(ModelConfig const& modelConfig);

    void fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, PeftTable const& peftTable,
        ReqIdsVec const& reqIds, std::vector<SizeType32> const& reqBeamWidth, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, PeftValues const& peftValues,
        SizeType32 batchIdx, SizeType32 beamWidth, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig) const;

private:
    std::unordered_map<SizeType32, LoraModule> mModuleIdToModule;
    std::unordered_map<SizeType32, SizeType32> mModuleOffset;
};
}
