
#pragma once

#include "../executor/executor.h"
#include "common.h"
#include "speculativeDecodingModule.h"
#include <memory>

namespace suggestify::runtime
{

class LookaheadModule : public SpeculativeDecodingModule
{
public:
    explicit LookaheadModule(SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens) noexcept
        : SpeculativeDecodingModule(maxDraftPathLen, maxDecodingDraftTokens, maxDecodingDraftTokens)
        , mExecutionConfig()
    {
    }

    explicit LookaheadModule() noexcept
        : LookaheadModule(0, 0)
    {
    }

    void setExecutionConfig(executor::LookaheadDecodingConfig const& config)
    {
        mExecutionConfig = config;
    }

    executor::LookaheadDecodingConfig const getExecutionConfig() const
    {
        return mExecutionConfig;
    }

private:
    executor::LookaheadDecodingConfig mExecutionConfig;
};

}
