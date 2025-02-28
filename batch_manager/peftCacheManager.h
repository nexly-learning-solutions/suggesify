
#pragma once

#include "../common.h"
#include "../llmRequest.h"
#include "../peftCacheManagerConfig.h"
#include "../common/tllmException.h"
#include "../runtime/loraCache.h"
#include "../runtime/modelConfig.h"
#include "../runtime/workerPool.h"
#include "../runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <future>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sugesstify::batch_manager
{

using runtime::SizeType32;

class PeftTaskNotCachedException : public runtime::LoraExpectedException
{
public:
    explicit PeftTaskNotCachedException(std::string const& msg);
    ~PeftTaskNotCachedException() noexcept override;
};

class BasePeftCacheManager
{
public:
    using LlmRequestPtr = std::shared_ptr<LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using PeftTable = std::map<uint64_t, std::vector<runtime::LoraCache::TaskLayerModuleConfig>>;

    virtual void addRequestPeft(LlmRequestPtr llmRequest, bool tryGpuCache = true) = 0;

    virtual PeftTable ensureBatch(
        RequestVector const& contextRequests, RequestVector const& generationRequests, bool resetGpuCache = false)
        = 0;

    virtual void resetDeviceCache() = 0;

    virtual void markRequestDone(LlmRequest const& llmReq, bool pause = false) = 0;

    [[nodiscard]] virtual SizeType32 getMaxDevicePages() const = 0;

    [[nodiscard]] virtual SizeType32 getMaxHostPages() const = 0;

    [[nodiscard]] virtual SizeType32 determineNumPages(std::shared_ptr<LlmRequest> llmRequest) const = 0;

    [[nodiscard]] virtual bool enabled() const = 0;
};

class PeftCacheManager : public BasePeftCacheManager
{
public:
    PeftCacheManager(PeftCacheManagerConfig const& config, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager);

    void addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache = true) override;

    PeftTable ensureBatch(RequestVector const& contextRequests, RequestVector const& generationRequests,
        bool resetGpuCache = false) override;

    [[nodiscard]] bool isTaskCached(uint64_t taskId) const;

    [[nodiscard]] bool isTaskDone(uint64_t taskId) const;

    [[nodiscard]] bool isTaskDoneDevice(uint64_t taskId) const;

    void resetDeviceCache() override;

    void markRequestDone(LlmRequest const& llmReq, bool pause = false) override;

    [[nodiscard]] SizeType32 getMaxDevicePages() const override;

    [[nodiscard]] SizeType32 getMaxHostPages() const override;

    [[nodiscard]] SizeType32 determineNumPages(std::shared_ptr<LlmRequest> llmRequest) const override;

    inline bool enabled() const override
    {
        return true;
    }

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> const& getActiveTasks() const;

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> const& getPausedTasks() const;

    void updateTaskState(uint64_t taskId, uint64_t reqId, bool terminate = false, bool pause = false);

    static std::pair<uint64_t, uint64_t> getMaxNumSlots(PeftCacheManagerConfig const& config,
        nvinfer1::DataType dataType, uint64_t pageWidth, uint64_t max1dModSize,
        runtime::BufferManager const& bufferManager);

    static std::pair<runtime::LoraCachePageManagerConfig, runtime::LoraCachePageManagerConfig> getPageManagerConfig(
        PeftCacheManagerConfig const& config, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager);

private:
    std::unique_ptr<runtime::LoraCache> mHostLoraCache;
    std::unique_ptr<runtime::LoraCache> mDeviceLoraCache;

    std::shared_ptr<runtime::WorkerPool> mPutWorkerPool;
    std::unique_ptr<runtime::WorkerPool> mEnsureWorkerPool;

    mutable std::mutex mPutFuturesMutex;
    std::unordered_map<std::uint64_t, std::future<void>> mPutFutures;

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> mTaskIdToReqIds;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> mTaskIdToPausedReqIds;

    std::tuple<std::map<uint64_t, std::future<void>>, std::map<uint64_t, std::vector<uint64_t>>> getTaskMaps(
        RequestVector const& contextRequests, RequestVector const& generationRequests);

    runtime::ModelConfig mModelConfig;
    runtime::WorldConfig mWorldConfig;

    int mDevice{-1};
};

class NoOpPeftCacheManager : public BasePeftCacheManager
{
    void addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache = true) override;

    PeftTable ensureBatch(RequestVector const& contextRequests, RequestVector const& generationRequests,
        bool resetGpuCache = false) override;

    void resetDeviceCache() override;

    void markRequestDone(LlmRequest const& llmReq, bool pause = false) override;

    [[nodiscard]] SizeType32 getMaxDevicePages() const override;

    [[nodiscard]] SizeType32 getMaxHostPages() const override;

    [[nodiscard]] SizeType32 determineNumPages(std::shared_ptr<LlmRequest> llmRequest) const override;

    inline bool enabled() const override
    {
        return false;
    }
};
}
