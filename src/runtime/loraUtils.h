
#pragma once

#include "common.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "worldConfig.h"

namespace suggestify::runtime::lora
{

SizeType32 constexpr kLORA_CONFIG_ROW_SIZE = 3;
SizeType32 constexpr kLORA_CONFIG_MODULE_OFF = 0;
SizeType32 constexpr kLORA_CONFIG_LAYER_OFF = 1;
SizeType32 constexpr kLORA_CONFIG_ADAPTER_SIZE_OFF = 2;

SizeType32 constexpr kLORA_NUM_WEIGHTS_POINTERS = 2;

void loraValidateRequestTensorDims(std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig);

void loraValidateRequestTensors(std::optional<std::uint64_t> const& optTaskId,
    std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig);
}
