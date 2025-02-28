
#pragma once

#include "../kvCacheManager.h"

#include <chrono>
#include <vector>

using namespace sugesstify::batch_manager::kv_cache_manager;

namespace sugesstify::batch_manager::eviction_policy
{

class BaseEvictionPolicy
{
public:
    virtual ~BaseEvictionPolicy() = default;

    virtual void initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> sizes,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority)
        = 0;

    virtual std::tuple<BlockPtr, bool> getFreeBlock(SizeType32 cacheLevel) = 0;
    virtual void releaseBlock(BlockPtr block) = 0;
    virtual void releaseBlock(BlockPtr block, bool toFront) = 0;
    virtual SizeType32 getNumFreeBlocks(SizeType32 cacheLevel) = 0;
    virtual void claimBlock(BlockPtr block) = 0;
    virtual void claimBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
        std::optional<std::chrono::milliseconds> durationMs)
        = 0;
    virtual void refresh() = 0;
};

struct ExpiringBlockComparator
{
    inline bool operator()(BlockPtr const& a, BlockPtr const& b) const
    {
        return a->getExpirationTime() != b->getExpirationTime() ? a->getExpirationTime() < b->getExpirationTime()
                                                                : a.get() < b.get();
    }
};

class LRUEvictionPolicy : public BaseEvictionPolicy
{
public:
    void initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> sizes,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority) override;
    std::tuple<BlockPtr, bool> getFreeBlock(SizeType32 cacheLevel) override;

    void releaseBlock(BlockPtr block) override;
    void releaseBlock(BlockPtr block, bool toFront) override;

    SizeType32 getNumFreeBlocks(SizeType32 cacheLevel) override;

    void claimBlock(BlockPtr block) override;
    void claimBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
        std::optional<std::chrono::milliseconds> durationMs) override;

    void refresh() override;

    [[nodiscard]] virtual std::chrono::steady_clock::time_point::duration getTime() const;

private:
    bool isReleasedLeafBlock(BlockPtr const& block);

    std::vector<std::vector<FreeBlocksQueue>> mFreeQueues;
    std::vector<std::unordered_set<SizeType32>> mReleasedBlocks;
    std::vector<std::optional<FreeBlocksQueue::iterator>> mFreeBlockIterators;
    std::vector<SizeType32> mNumFreeBlocksPerLevel;
    executor::RetentionPriority mSecondaryOffloadMinPriority;
    std::set<BlockPtr, ExpiringBlockComparator> mExpiringBlockHeap;
};

}
