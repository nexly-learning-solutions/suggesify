#pragma once

#include "../common/assert.h"
#include "../common/mpiUtils.h"
#include "../executor/executor.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace suggestify::executor::disagg_executor
{

namespace texec = suggestify::executor;

struct ResponseWithId
{

    suggestify::executor::Response response;
    IdType gid;

    ResponseWithId(suggestify::executor::Response&& response, IdType gid)
        : response(std::move(response))
        , gid(gid)
    {
    }

    ResponseWithId(suggestify::executor::Response const& response, IdType gid)
        : response(response)
        , gid(gid)
    {
    }

    ResponseWithId(ResponseWithId&& other) noexcept
        : response(std::move(other.response))
        , gid(other.gid)
    {
        other.gid = {};
    }

    ResponseWithId(ResponseWithId const& other) = default;

    ResponseWithId& operator=(ResponseWithId&& other) noexcept
    {
        if (this != &other)
        {
            response = std::move(other.response);
            gid = other.gid;
            other.gid = {};
        }
        return *this;
    }

    ResponseWithId& operator=(ResponseWithId const& other)
    {

        if (this != &other)
        {
            response = other.response;
            gid = other.gid;
        }
        return *this;
    }

    ~ResponseWithId() = default;
};

class DisaggExecutorOrchestrator
{
public:

    DisaggExecutorOrchestrator(std::vector<std::filesystem::path> const& ctxEnginePaths,
        std::vector<std::filesystem::path> const& genEnginePaths,
        std::vector<executor::ExecutorConfig> const& ctxExecutorConfigs,
        std::vector<executor::ExecutorConfig> const& genExecutorConfigs, bool hasContextAwaitThreads,
        bool hasGenAwaitThreads);

    [[nodiscard]] std::vector<IdType> enqueueContext(std::vector<texec::Request> const& requests,
        std::optional<int> selectContextId = std::nullopt, bool batch = false);


    void enqueueGeneration(std::vector<texec::Request> const& requests, std::vector<IdType> const& globalRequestIds,
        std::optional<int> selectGenIdx = std::nullopt, bool batch = false);


    [[nodiscard]] std::vector<ResponseWithId> awaitContextResponses(
        std::optional<std::chrono::milliseconds> const& timeout, std::optional<int> contextIdx = std::nullopt);

    [[nodiscard]] std::vector<ResponseWithId> awaitGenerationResponses(
        std::optional<std::chrono::milliseconds> const& timeout, std::optional<int> genIdx = std::nullopt);

    [[nodiscard]] bool canEnqueue() const;

    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getContextExecutors() const;

    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getGenExecutors() const;

    ~DisaggExecutorOrchestrator();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};
}
