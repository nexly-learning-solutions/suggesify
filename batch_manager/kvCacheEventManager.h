
#pragma once

#include "../executor/executor.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace sugesstify::batch_manager::kv_cache_manager
{

using SizeType32 = sugesstify::runtime::SizeType32;

class KVCacheBlock;
using BlockPtr = std::shared_ptr<KVCacheBlock>;

class KVCacheEventManager
{
public:
    explicit KVCacheEventManager(size_t maxKVEventEntries);

    ~KVCacheEventManager();
    KVCacheEventManager(KVCacheEventManager& other) = delete;
    KVCacheEventManager& operator=(KVCacheEventManager& other) = delete;
    KVCacheEventManager(KVCacheEventManager&& other) = delete;
    KVCacheEventManager& operator=(KVCacheEventManager&& other) = delete;

    void enqueueCreatedEvent(std::vector<SizeType32> const& numBlocksPerCacheLevel);

    void enqueueStoredEvent(std::vector<BlockPtr> const& blocks);

    void enqueueRemovedEvent(BlockPtr const& block);

    void enqueueUpdatedEvent(executor::KVCacheUpdatedData const& data);

    std::deque<executor::KVCacheEvent> getEvents(std::optional<std::chrono::milliseconds> timeout);

    void flush();

    void worker();

private:
    void enqueueEvent(executor::KVCacheEvent&& event);

    bool mRun;
    std::thread mWorkerThread;

    std::deque<executor::KVCacheEvent> mEvents;
    std::mutex mEventsMutex;
    std::condition_variable mEmptyCV;

    std::deque<std::deque<executor::KVCacheEvent>> mPendingEvents;
    std::mutex mPendingEventsMutex;
    std::condition_variable mPendingEmptyCV;

    std::deque<executor::KVCacheEvent> mEventQueue;

    size_t mMaxSize;
    size_t mEventId;
};

}
