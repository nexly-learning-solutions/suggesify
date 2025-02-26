
#pragma once

#include "../common/cudaUtils.h"
#include "common.h"

namespace suggestify::runtime
{

class SpeculativeDecodingModule
{
public:
    explicit SpeculativeDecodingModule(
        SizeType32 maxDraftPathLen, SizeType32 maxDecodingDraftTokens, SizeType32 maxNumPaths) noexcept
        : mMaxDraftPathLen(maxDraftPathLen)
        , mMaxDecodingDraftTokens(maxDecodingDraftTokens)
        , mMaxNumPaths(maxNumPaths)
    {
        computeNumPackedMasks();
    }

    explicit SpeculativeDecodingModule() noexcept
        : SpeculativeDecodingModule(0, 0, 0)
    {
    }

    virtual ~SpeculativeDecodingModule() = default;

    SpeculativeDecodingModule(SpeculativeDecodingModule const& o) = default;
    SpeculativeDecodingModule& operator=(SpeculativeDecodingModule const& o) = default;

    [[nodiscard]] SizeType32 getMaxDraftPathLen() const noexcept
    {
        return mMaxDraftPathLen;
    }

    [[nodiscard]] SizeType32 getMaxPathLen() const noexcept
    {
        return getMaxDraftPathLen() + 1;
    }

    [[nodiscard]] SizeType32 getMaxDecodingDraftTokens() const noexcept
    {
        return mMaxDecodingDraftTokens;
    }

    [[nodiscard]] SizeType32 getMaxDecodingTokens() const noexcept
    {
        return getMaxDecodingDraftTokens() + 1;
    }

    [[nodiscard]] SizeType32 getNumPackedMasks() const noexcept
    {
        return mMaxNumPackedMasks;
    }

    [[nodiscard]] SizeType32 getMaxNumPaths() const noexcept
    {
        return mMaxNumPaths;
    }

    void setMaxDraftTokens(SizeType32 maxDraftTokens) noexcept
    {
        mMaxDecodingDraftTokens = maxDraftTokens;
        computeNumPackedMasks();
    }

    void setMaxDraftPathLen(SizeType32 maxDraftPathLen) noexcept
    {
        mMaxDraftPathLen = maxDraftPathLen;
    }

    void setMaxNumPaths(SizeType32 maxNumPaths) noexcept
    {
        mMaxNumPaths = maxNumPaths;
    }

private:
    void computeNumPackedMasks() noexcept
    {
        mMaxNumPackedMasks = suggestify::common::divUp(mMaxDecodingDraftTokens, 32);
    }

private:
    SizeType32 mMaxDraftPathLen;
    SizeType32 mMaxDecodingDraftTokens;
    SizeType32 mMaxNumPaths;
    SizeType32 mMaxNumPackedMasks;
};
}
