
#pragma once

#include "common.h"

namespace suggestify::runtime
{
struct RuntimeDefaults
{
    RuntimeDefaults(
        std::optional<std::vector<SizeType32>> maxAttentionWindowVec, std::optional<SizeType32> sinkTokenLength)
        : maxAttentionWindowVec(maxAttentionWindowVec)
        , sinkTokenLength(sinkTokenLength)
    {
    }

    std::optional<std::vector<SizeType32>> maxAttentionWindowVec;
    std::optional<SizeType32> sinkTokenLength;

    RuntimeDefaults() = default;
};

}
