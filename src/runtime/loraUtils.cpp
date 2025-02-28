
#include "loraUtils.h"
#include "../common/assert.h"
#include "common.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "worldConfig.h"
#include <string>

namespace suggestify::runtime::lora
{

void loraValidateRequestTensorDims(std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig)
{
    CHECK_WITH_INFO(optReqLoraWeights.has_value() && optReqLoraConfig.has_value(),
        "Request for LoRA inference must have both lora_weights and lora_keys");

    SizeType32 constexpr expectedBatchSize = 1;
    SizeType32 constexpr expectedWeightsDims = 3;
    SizeType32 constexpr expectedKeysDims = 3;

    auto weights = optReqLoraWeights.value();
    auto keys = optReqLoraConfig.value();
    CHECK_WITH_INFO(weights->getShape().nbDims == expectedWeightsDims, "Invalid shape for lora_weights tensor");
    CHECK_WITH_INFO(keys->getShape().nbDims == expectedKeysDims, "Invalid shape for lora_keys tensor");
    CHECK_WITH_INFO(
        weights->getShape().d[0] == expectedBatchSize, "Expected batch dimension to be 1 for each lora request");
    CHECK_WITH_INFO(
        keys->getShape().d[0] == expectedBatchSize, "Expected batch dimension to be 1 for each lora request");
    CHECK_WITH_INFO(weights->getMemoryType() != MemoryType::kGPU, "Expected lora weights to be in CPU memory");
    CHECK_WITH_INFO(keys->getMemoryType() != MemoryType::kGPU, "Expected lora weights to be in CPU memory");
    CHECK_WITH_INFO(keys->getDataType() == nvinfer1::DataType::kINT32,
        "Expected  lora keys to have TYPE_INT32 but was " + std::string(keys->getDataTypeName()));

    CHECK_WITH_INFO(keys->getShape().d[1] == weights->getShape().d[1],
        "Expected dim1 lora_weights and lora_keys to have the same size");
    CHECK_WITH_INFO(keys->getShape().d[2] == kLORA_CONFIG_ROW_SIZE,
        "Expected dim2 of lora_keys to have a size of " + std::to_string(kLORA_CONFIG_ROW_SIZE));
}

void loraValidateRequestTensors(std::optional<std::uint64_t> const& optTaskId,
    std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig)
{
    CHECK_WITH_INFO(optTaskId.has_value(), "lora_task_id must be set for LoRA inference");
    if (optReqLoraWeights.has_value() || optReqLoraConfig.has_value())
    {
        loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig);

        auto weights = optReqLoraWeights.value();
        auto config = optReqLoraConfig.value();
        SizeType32 nbModelLayers = modelConfig.getNbAttentionLayers();
        CHECK_WITH_INFO(weights->getDataType() == modelConfig.getDataType(),
            "Expected lora weights to be the same data type as base model");

        auto loraModules = modelConfig.getLoraModules();
        auto configPtr = bufferCast<SizeType32>(*config);
        auto maxAdapterSize = modelConfig.getMaxLoraRank();
        for (SizeType32 row = 0; row < config->getShape().d[1]; ++row)
        {
            auto modId = configPtr[row * kLORA_CONFIG_ROW_SIZE + kLORA_CONFIG_MODULE_OFF];
            auto layerId = configPtr[row * kLORA_CONFIG_ROW_SIZE + kLORA_CONFIG_LAYER_OFF];
            auto adapterSize = configPtr[row * kLORA_CONFIG_ROW_SIZE + kLORA_CONFIG_ADAPTER_SIZE_OFF];

            CHECK_WITH_INFO(
                layerId >= 0 && layerId < nbModelLayers, "Expected layerId to be in the range [0, numModelLayers)");
            CHECK_WITH_INFO(adapterSize > 0, "Expected adapterSize to be > 0");
            auto it = std::find_if(
                loraModules.begin(), loraModules.end(), [modId](LoraModule const& m) { return m.value() == modId; });
            std::string moduleName(LoraModule::toModuleName(modId));
            CHECK_WITH_INFO(it != loraModules.end(), "lora module " + moduleName + " not enabled for this model");
            CHECK_WITH_INFO(it->flattenedInOutSize(adapterSize) <= weights->getShape().d[2],
                "lora_weights has to few values for " + moduleName);
            CHECK_WITH_INFO(adapterSize <= maxAdapterSize,
                "Invalid low_rank (" + std::to_string(adapterSize) + "). low_rank must be smaller than mMaxLowRank ("
                    + std::to_string(maxAdapterSize) + ")");
        }
    }
}
}
