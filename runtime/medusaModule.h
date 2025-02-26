
#pragma once

#include "iTensor.h"
#include "speculativeDecodingModule.h"

#include <vector>

namespace suggestify::runtime
{

class MedusaModule : public SpeculativeDecodingModule
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using MedusaChoices = std::vector<std::vector<SizeType32>>;

    explicit MedusaModule(SizeType32 maxAcceptedTokens, SizeType32 maxDraftTokens) noexcept
        : SpeculativeDecodingModule(maxAcceptedTokens, maxDraftTokens, maxDraftTokens)
    {
    }

    explicit MedusaModule() noexcept
        : MedusaModule(0, 0)
    {
    }

    [[nodiscard]] MedusaChoices const& getMedusaChoices() const noexcept
    {
        return mDefaultMedusaChoices;
    }

private:
    MedusaChoices mDefaultMedusaChoices = {{0}, {0, 0}, {1}, {0, 1}, {2}, {0, 0, 0}, {1, 0}, {0, 2}, {3}, {0, 3}, {4},
        {0, 4}, {2, 0}, {0, 5}, {0, 0, 1}, {5}, {0, 6}, {6}, {0, 7}, {0, 1, 0}, {1, 1}, {7}, {0, 8}, {0, 0, 2}, {3, 0},
        {0, 9}, {8}, {9}, {1, 0, 0}, {0, 2, 0}, {1, 2}, {0, 0, 3}, {4, 0}, {2, 1}, {0, 0, 4}, {0, 0, 5}, {0, 0, 0, 0},
        {0, 1, 1}, {0, 0, 6}, {0, 3, 0}, {5, 0}, {1, 3}, {0, 0, 7}, {0, 0, 8}, {0, 0, 9}, {6, 0}, {0, 4, 0}, {1, 4},
        {7, 0}, {0, 1, 2}, {2, 0, 0}, {3, 1}, {2, 2}, {8, 0}, {0, 5, 0}, {1, 5}, {1, 0, 1}, {0, 2, 1}, {9, 0},
        {0, 6, 0}, {0, 0, 0, 1}, {1, 6}, {0, 7, 0}};
};
}
