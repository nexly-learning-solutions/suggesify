
#pragma once

#include "suggestify/common/assert.h"
#include "../types.h"
#include "speculativeDecodingModule.h"

namespace suggestify::runtime
{

class EagleModule : public SpeculativeDecodingModule
{
public:
    explicit EagleModule(SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens, SizeType32 numTransformersLayer,
        SizeType32 maxNonLeafNodesPerLayer) noexcept
        : SpeculativeDecodingModule(maxDraftPathLen, maxDecodingDraftTokens, maxDecodingDraftTokens + 1)
        , mNumTransformersLayer(numTransformersLayer)
        , mMaxNonLeafNodesPerLayer(maxNonLeafNodesPerLayer)
    {
    }

    explicit EagleModule() noexcept
        : EagleModule(0, 0, 0, 0)
    {
    }

    [[nodiscard]] executor::EagleChoices const& getDefaultEagleChoices() const noexcept
    {
        return mDefaultEagleChoices;
    }

    [[nodiscard]] SizeType32 getNumTransformerLayers() const noexcept
    {
        return mNumTransformersLayer;
    }

    [[nodiscard]] SizeType32 getMaxNonLeafNodesPerLayer() const noexcept
    {
        return mMaxNonLeafNodesPerLayer;
    }

private:
    SizeType32 mNumTransformersLayer;
    SizeType32 mMaxNonLeafNodesPerLayer;

    executor::EagleChoices mDefaultEagleChoices = {{0}, {0, 0}, {1}, {0, 1}, {2}, {0, 0, 0}, {1, 0}, {0, 2}, {3},
        {0, 3}, {4}, {0, 4}, {2, 0}, {0, 5}, {0, 0, 1}, {5}, {0, 6}, {6}, {0, 7}, {0, 1, 0}, {1, 1}, {7}, {0, 8},
        {0, 0, 2}, {3, 0}, {0, 9}, {8}, {9}, {1, 0, 0}, {0, 2, 0}, {1, 2}, {0, 0, 3}, {4, 0}, {2, 1}, {0, 0, 4},
        {0, 0, 5}, {0, 0, 0, 0}, {0, 1, 1}, {0, 0, 6}, {0, 3, 0}, {5, 0}, {1, 3}, {0, 0, 7}, {0, 0, 8}, {0, 0, 9},
        {6, 0}, {0, 4, 0}, {1, 4}, {7, 0}, {0, 1, 2}, {2, 0, 0}, {3, 1}, {2, 2}, {8, 0}, {0, 5, 0}, {1, 5}, {1, 0, 1},
        {0, 2, 1}, {9, 0}, {0, 6, 0}, {0, 0, 0, 1}, {1, 6}, {0, 7, 0}};
};
}
