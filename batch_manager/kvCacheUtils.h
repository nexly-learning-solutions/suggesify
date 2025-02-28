
#pragma once

#include "../batch_manager/kvCacheManager.h"

namespace sugesstify::batch_manager::kv_cache_manager
{

class BlockIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = runtime::ITensor;
    using pointer = runtime::ITensor::SharedPtr;
    using reference = value_type&;
    using SizeType32 = sugesstify::runtime::SizeType32;

    BlockIterator(runtime::ITensor::SharedPtr blockPoolPtr, std::vector<SizeType32> blockIds, size_t idx)
        : mPool{std::move(blockPoolPtr)}
        , mBlockIds{std::move(blockIds)}
        , mIdx{idx}
    {
        TLLM_CHECK(mPool);
        TLLM_CHECK(mIdx <= mBlockIds.size());
        update();
    }

    [[nodiscard]] pointer operator->()
    {
        return mCurrent;
    }

    [[nodiscard]] reference operator*()
    {
        return *mCurrent;
    }

    BlockIterator& operator++()
    {
        mIdx++;
        update();
        return *this;
    }

    BlockIterator operator++(int)
    {
        auto ret = *this;
        ret.update();
        mIdx++;
        return ret;
    }

    operator runtime::ITensor::SharedPtr()
    {
        return mCurrent;
    }

    [[nodiscard]] bool operator==(BlockIterator const& other) const
    {
        return mIdx == other.mIdx && mPool.get() == other.mPool.get();
    }

    [[nodiscard]] bool operator!=(BlockIterator const& other) const
    {
        return !(*this == other);
    }

private:
    void update()
    {
        if (mIdx < mBlockIds.size())
        {
            mCurrent = runtime::ITensor::slice(mPool, mBlockIds.at(mIdx), 1);
        }
    }

    runtime::ITensor::SharedPtr mPool;
    runtime::ITensor::SharedPtr mCurrent;
    const std::vector<SizeType32> mBlockIds;
    size_t mIdx;
};

[[nodiscard]] BlockIterator getBlockBeginIt(
    BaseKVCacheManager const& cacheManager, LlmRequest const& request, SizeType32 beam, SizeType32 poolIdx);

[[nodiscard]] BlockIterator getBlockEndIt(
    BaseKVCacheManager const& cacheManager, LlmRequest const& request, SizeType32 beam, SizeType32 poolIdx);

}
