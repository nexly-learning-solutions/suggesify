
#pragma once

#include "common.h"
#include "../llmRequest.h"
#include "../common/algorithm.h"
#include "../common/optionalRef.h"
#include "../runtime/common.h"
#include <variant>

namespace sugesstify::batch_manager
{
namespace kv_cache_manager
{
class BaseKVCacheManager;
}
class BasePeftCacheManager;
}

namespace sugesstify::batch_manager
{

using sugesstify::runtime::SizeType32;
using common::OptionalRef;

class BaseCapacityScheduler
{
public:
    explicit BaseCapacityScheduler(LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
        : mNoScheduleUntilState(noScheduleUntilState)
        , mNoScheduleAfterState(noScheduleAfterState)
    {
    }

    [[nodiscard]] LlmRequestState constexpr getNoScheduleUntilState() const noexcept
    {
        return mNoScheduleUntilState;
    }

    [[nodiscard]] LlmRequestState constexpr getNoScheduleAfterState() const noexcept
    {
        return mNoScheduleAfterState;
    }

private:
    LlmRequestState mNoScheduleUntilState;
    LlmRequestState mNoScheduleAfterState;
};

class MaxRequestsScheduler : public BaseCapacityScheduler
{
public:
    explicit MaxRequestsScheduler(SizeType32 maxNumRequests,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests) const;

private:
    SizeType32 mMaxNumRequests;
};

class MaxUtilizationScheduler : public BaseCapacityScheduler
{
public:
    MaxUtilizationScheduler(SizeType32 maxNumRequests, bool manyMicroBatches,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(
        kv_cache_manager::BaseKVCacheManager& kvCacheManager, OptionalRef<BasePeftCacheManager const> peftCacheManager,
        RequestList const& activeRequests) const;

private:
    std::pair<bool, bool> trySchedulingRequestMaxUtilization(kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
        OptionalRef<BasePeftCacheManager const> peftCacheManager, std::shared_ptr<LlmRequest> const& req,
        RequestVector& scheduledRequests, SizeType32& numScheduledBlocks, SizeType32& numScheduledPeftPages,
        std::unordered_set<uint64_t>& seenTaskIds) const;

    SizeType32 mMaxNumRequests;
    bool mManyMicroBatches;
};

class GuaranteedNoEvictScheduler : public BaseCapacityScheduler
{
public:
    GuaranteedNoEvictScheduler(SizeType32 maxNumRequests,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(
        kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
        OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const;

protected:
    template <bool StaticBatchScheduling>
    [[nodiscard]] std::tuple<RequestVector, RequestVector> impl(
        kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
        OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const;

private:
    SizeType32 mMaxNumRequests;
};

class StaticBatchScheduler : public GuaranteedNoEvictScheduler
{
public:
    StaticBatchScheduler(SizeType32 maxNumRequests,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(
        kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
        OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const;
};

class CapacityScheduler : public Algorithm
{
public:
    constexpr static auto name{"CapacityScheduler"};

    explicit CapacityScheduler(SizeType32 maxNumRequests, executor::CapacitySchedulerPolicy capacitySchedulerPolicy,
        bool hasKvCacheManager, std::optional<bool> manyMicroBatches = std::nullopt,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests,
        OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager = std::nullopt,
        OptionalRef<BasePeftCacheManager const> peftCacheManager = std::nullopt,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager = std::nullopt) const;

private:
    std::variant<std::monostate, MaxRequestsScheduler, MaxUtilizationScheduler, GuaranteedNoEvictScheduler,
        StaticBatchScheduler>
        mScheduler;
};

}
