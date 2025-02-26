
#pragma once

#include "../runtime/iTensor.h"
#include "../runtime/speculativeDecodingModule.h"

#include <vector>

namespace suggestify::runtime::utils
{
struct TreeNode
{
    SizeType32 nodeId;
    SizeType32 depth;
    SizeType32 parentLinearIdx;
    SizeType32 linearIdx;
    std::vector<SizeType32> childLinearIndices;
};

SizeType32 initTensorsFromChoices(SpeculativeDecodingModule const& speculativeDecodingModule,
    std::vector<std::vector<SizeType32>> const& choices, std::vector<SizeType32>& topKs,
    ITensor::SharedPtr generationInputLengths, ITensor::SharedPtr positionOffsets, ITensor::SharedPtr treeIds,
    ITensor::SharedPtr paths, ITensor::SharedPtr packedMask,
    std::optional<SizeType32> maxNonLeafNodesPerLayer = std::nullopt);

}
