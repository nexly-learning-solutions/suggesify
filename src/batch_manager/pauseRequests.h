
#pragma once

#include "common.h"
#include "../kvCacheManager.h"
#include "../peftCacheManager.h"
#include "../sequenceSlotManager.h"
#include "../common/algorithm.h"
#include "../common/optionalRef.h"
#include "../runtime/common.h"

namespace sugesstify::batch_manager
{

class BasePeftCacheManager;
class LlmRequest;

namespace kv_cache_manager
{

class BaseKVCacheManager;

}
}

namespace sugesstify::batch_manager
{

namespace tle = sugesstify::executor;

class PauseRequests : Algorithm
{
    using BaseKVCacheManager = kv_cache_manager::BaseKVCacheManager;

    template <typename T>
    using OptionalRef = common::OptionalRef<T>;

public:
    constexpr static auto name{"PauseRequests"};

    using SizeType32 = sugesstify::runtime::SizeType32;

    PauseRequests(SizeType32 maxInputLen)
        : mMaxInputLen(maxInputLen)
    {
    }

    void operator()(RequestVector& requestsToPause, ReqIdsSet& inflightReqIds, ReqIdsSet& reqIdsToPause,
        bool pauseFlagged, SequenceSlotManager& seqSlotManager,
        OptionalRef<BaseKVCacheManager> kvCacheManager = std::nullopt,
        OptionalRef<BaseKVCacheManager> crossKvCacheManager = std::nullopt,
        OptionalRef<BasePeftCacheManager> peftCacheManager = std::nullopt) const;

private:
    SizeType32 mMaxInputLen;
};

}
