
#pragma once

#include "common.h"
#include "../batch_manager/llmRequest.h"
#include "../common/algorithm.h"
#include "../runtime/common.h"

namespace sugesstify::batch_manager
{

namespace batch_scheduler
{

struct ContextChunkingConfig
{
    ContextChunkingConfig() = default;

    executor::ContextChunkingPolicy chunkingPolicy;
    sugesstify::runtime::SizeType32 chunkUnitSize;
};

}

class MicroBatchScheduler : Algorithm
{
public:
    constexpr static auto name{"MicroBatchScheduler"};

    using SizeType32 = sugesstify::runtime::SizeType32;
    using ContextChunkingPolicy = sugesstify::executor::ContextChunkingPolicy;

    explicit MicroBatchScheduler(std::optional<batch_scheduler::ContextChunkingConfig> ctxChunkConfig = std::nullopt,
        std::optional<SizeType32> maxContextLength = std::nullopt,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    std::tuple<RequestVector, RequestVector> operator()(RequestVector& activeRequests, ReqIdsSet const& inflightReqIds,
        SizeType32 maxBatchSizeRuntime, std::optional<SizeType32> maxNumTokensRuntime) const;

    static void setCtxRequestsChunkSize(RequestVector& contextsToBeChunked, ContextChunkingPolicy ctxChunkPolicy,
        std::optional<SizeType32> ctxTokensCapacity, SizeType32 chunkUnitSize,
        std::optional<SizeType32> const& maxContextLength);

private:
    template <ContextChunkingPolicy tPolicy>
    static void setCtxRequestsChunkSize(RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity,
        SizeType32 chunkUnitSize, std::optional<SizeType32> const& maxContextLength);

    static void fitDraftTokens(RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity,
        SizeType32 chunkUnitSize, std::optional<SizeType32> const& maxContextLength);

    std::optional<SizeType32> mMaxContextLength;

    std::optional<batch_scheduler::ContextChunkingConfig> mCtxChunkConfig;

    LlmRequestState mNoScheduleUntilState;
    LlmRequestState mNoScheduleAfterState;
};

}
