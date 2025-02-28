
#include "../kvCacheManager.h"
#include "../runtime/bufferManager.h"
#include "../runtime/cudaEvent.h"

namespace tr = sugesstify::runtime;

#pragma once

namespace sugesstify::batch_manager::kv_cache_manager
{

class KVCacheTransferManager
{
public:
    explicit KVCacheTransferManager(tr::BufferManager const& bufferManager);

    void onboard(BlockPtr const& offloadBlock, BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools);

    void offload(BlockPtr const& block, BlockPtr const& offloadBlock, std::vector<KVCacheBlockPool> const& pools);

    void syncTransfers();

private:
    static tr::ITensor::SharedPtr computeBlockPointer(
        BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx);

    void copyBlock(
        BlockPtr const& src, BlockPtr const& dst, std::vector<KVCacheBlockPool> const& pools, bool isOffload);

    runtime::BufferManager mBufferManager;
    runtime::BufferManager mOnboardManager;
    runtime::BufferManager mOffloadManager;

    std::unordered_map<int32_t, tr::CudaEvent> mPendingOffloads;
};

}
