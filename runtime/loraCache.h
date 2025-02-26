
#pragma once

#include "../common/tllmException.h"
#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"
#include "loraCachePageManagerConfig.h"
#include "loraModule.h"
#include "modelConfig.h"
#include "worldConfig.h"

#include <NvInferRuntime.h>

#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace suggestify::runtime
{

class LoraExpectedException : public std::runtime_error
{
public:
    explicit LoraExpectedException(std::string const& msg);
    ~LoraExpectedException() noexcept override;
};

class LoraCacheFullException : public LoraExpectedException
{
public:
    explicit LoraCacheFullException(std::string const& msg);
    ~LoraCacheFullException() noexcept override;
};

class LoraCachePageManager
{
public:
    using TensorPtr = ITensor::SharedPtr;

    LoraCachePageManager(LoraCachePageManagerConfig const& config, BufferManager const& bufferManager);

    [[nodiscard]] std::optional<std::vector<std::size_t>> claimPages(SizeType32 numPages);

    [[nodiscard]] SizeType32 numAvailablePages() const;

    void releasePages(std::vector<std::size_t> const& pages);

    [[nodiscard]] ITensor::SharedConstPtr blockPtr(SizeType32 blockIdx) const;

    [[nodiscard]] ITensor::SharedConstPtr pagePtr(std::size_t pageIdx) const;

    [[nodiscard]] ITensor::SharedPtr mutablePagePtr(std::size_t pageIdx);

private:
    std::vector<TensorPtr> mPageBlocks;
    std::deque<std::size_t> mFreePageIds;
    std::vector<std::uint8_t> mIsPageFree;
    LoraCachePageManagerConfig const mConfig;

    void initialize(BufferManager const& bufferManager);
};

class LoraCache
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TaskIdType = std::uint64_t;

    struct TaskLayerModuleConfig
    {
        friend class TaskLayerModuleConfigBindings;

        std::size_t pageId;
        SizeType32 slotIdx;
        SizeType32 inSize;
        SizeType32 outSize;
        SizeType32 moduleId;
        SizeType32 layerId;
        SizeType32 adapterSize;
        SizeType32 numSlots;

        std::int64_t weightsInPointer;
        std::int64_t weightsOutPointer;

        std::string toString() const;

        bool operator==(LoraCache::TaskLayerModuleConfig const& o) const;
    };

    using TaskLayerModuleConfigListPtr = std::shared_ptr<std::vector<TaskLayerModuleConfig>>;

    LoraCache(LoraCachePageManagerConfig const& pageManagerConfig, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig, BufferManager const& bufferManager);

    void put(TaskIdType taskId, TensorPtr weights, TensorPtr config, bool load = true);

    void loadWeights(TaskIdType taskId, TensorPtr weights, TensorPtr config);

    [[nodiscard]] inline bool isLoaded(TaskIdType taskId) const
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        return kVALUE_STATUS_LOADED == getStatus(taskId);
    }

    [[nodiscard]] bool isDone(TaskIdType taskId) const;

    [[nodiscard]] inline bool has(TaskIdType taskId) const
    {
        std::lock_guard<std::mutex> lk(mCacheMutex);
        return kVALUE_STATUS_MISSING != getStatus(taskId);
    }

    [[nodiscard]] std::vector<TaskLayerModuleConfig> const& get(TaskIdType taskId);

    void bump(TaskIdType taskId);

    void markTaskDone(TaskIdType taskId);

    void markAllDone();

    [[nodiscard]] SizeType32 determineNumPages(TaskIdType taskId) const;

    [[nodiscard]] SizeType32 determineNumPages(TensorPtr config) const;

    [[nodiscard]] bool fits(TensorPtr config) const;

    void copyTask(TaskIdType taskId, LoraCache& deviceCache, bool markDone = false);

    [[nodiscard]] SizeType32 getNumPages() const;

    [[nodiscard]] ITensor::SharedConstPtr getPagePtr(size_t pageId) const;

    static std::vector<LoraCache::TaskLayerModuleConfig> copyToPages(TensorPtr weights, TensorPtr config,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::unordered_map<SizeType32, LoraModule> moduleIdToModel, BufferManager const& manager,
        std::vector<TensorPtr> const& pages, std::vector<std::size_t> const& pageIds);

    static void splitTransposeCpu(ITensor& output, ITensor const& input, SizeType32 tpSize, SizeType32 tpRank);

private:
    struct TaskValue
    {
        std::vector<std::size_t> pageIds;
        TaskLayerModuleConfigListPtr configs;
        std::list<TaskIdType>::iterator it;

        bool inProgress;
        bool loaded;
        bool done;
        bool loadInProgress;

        TaskValue() = delete;
        ~TaskValue() = default;

        TaskValue(std::vector<std::size_t> const& pageIds, TaskLayerModuleConfigListPtr const& configs,
            std::list<TaskIdType>::iterator it, bool inProgress, bool loaded, bool done, bool loadInProgress = false)
            : pageIds(pageIds)
            , configs(configs)
            , it(it)
            , inProgress(inProgress)
            , loaded(loaded)
            , done(done)
            , loadInProgress(loadInProgress)
        {
        }

        TaskValue(TaskValue&& o) noexcept
        {
            std::swap(pageIds, o.pageIds);
            std::swap(configs, o.configs);
            std::swap(it, o.it);
            std::swap(inProgress, o.inProgress);
            std::swap(loaded, o.loaded);
            std::swap(done, o.done);
            std::swap(loadInProgress, o.loadInProgress);
        }

        TaskValue& operator=(TaskValue&& o)
        {
            std::swap(pageIds, o.pageIds);
            std::swap(configs, o.configs);
            std::swap(it, o.it);
            std::swap(inProgress, o.inProgress);
            std::swap(loaded, o.loaded);
            std::swap(done, o.done);
            std::swap(loadInProgress, o.loadInProgress);
            return *this;
        }
    };

    using TaskValuePtr = std::shared_ptr<TaskValue>;

    enum ValueStatus
    {
        kVALUE_STATUS_MISSING = 0,
        kVALUE_STATUS_PROCESSING = 1,
        kVALUE_STATUS_LOADED = 2,
    };

    LoraCachePageManagerConfig mPageManagerConfig;
    ModelConfig mModelConfig;
    WorldConfig mWorldConfig;

    mutable std::mutex mPagesMutex;
    std::unique_ptr<LoraCachePageManager> mCachePageManager;

    mutable std::mutex mCacheMutex;
    std::unordered_map<TaskIdType, TaskValuePtr> mCacheMap;
    std::list<TaskIdType> mInProgressTasks;
    std::list<TaskIdType> mDoneTasks;

    std::vector<std::unique_ptr<BufferManager>> mDeviceBufferManagers;
    std::unique_ptr<BufferManager> mBufferManager;

    std::unordered_map<SizeType32, LoraModule> mModuleIdToModule;

    template <typename T>
    static void splitTransposeCpuInner(ITensor& output, ITensor const& input, SizeType32 tpSize, SizeType32 tpRank);

    void loadWeights(TaskValue& cacheValue, TensorPtr weights, TensorPtr config);
    void bumpTaskInProgress(TaskIdType taskId);
    [[nodiscard]] ValueStatus getStatus(TaskIdType taskId) const;

    [[nodiscard]] std::vector<std::size_t> claimPagesWithEvict(SizeType32 numPages);

    std::map<size_t, std::pair<size_t, SizeType32>> copyTaskMapPages(TaskValue& targetTaskValue,
        TaskValue const& sourceTaskValue, std::vector<size_t> const& targetPageIds, LoraCache const& targetCache);
};

std::string to_string(LoraCache::TaskLayerModuleConfig const& v);

std::ostream& operator<<(std::ostream& os, LoraCache::TaskLayerModuleConfig const& v);

}
