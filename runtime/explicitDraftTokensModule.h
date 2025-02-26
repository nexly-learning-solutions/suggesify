
#pragma once

#include "suggestify/common/assert.h"
#include "speculativeDecodingModule.h"

namespace suggestify::runtime
{

class ExplicitDraftTokensModule : public SpeculativeDecodingModule
{
public:
    explicit ExplicitDraftTokensModule(
        SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens, SizeType32 maxNumPaths) noexcept
        : SpeculativeDecodingModule(maxDraftPathLen, maxDecodingDraftTokens, maxNumPaths)
    {
        CHECK(maxNumPaths * maxDraftPathLen == maxDecodingDraftTokens);
    }

    explicit ExplicitDraftTokensModule() noexcept
        : ExplicitDraftTokensModule(0, 0, 0)
    {
    }
};
}
