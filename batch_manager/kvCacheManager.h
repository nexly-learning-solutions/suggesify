
#pragma once

#include "../kvCacheConfig.h"
#include "../kvCacheEventManager.h"
#include "../llmRequest.h"
#include "../common/optionalRef.h"
#include "../src/kvCacheIndex.h"
#include "../runtime/bufferManager.h"
#include "../runtime/common.h"
#include "../runtime/cudaStream.h"
#include "../runtime/iBuffer.h"
#include "../runtime/iTensor.h"
#include "../runtime/modelConfig.h"
#include "../runtime/worldConfig.h"
#include <NvInferRuntime.h>

#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace sugesstify::batch_manager::eviction_policy
{
class BaseEvictionPolicy;
}

namespace sugesstify::batch_manager::kv_cache_manager
{

static constexpr SizeType32 kPrimaryLevel = 0;

static constexpr SizeType32 kSecondaryLevel = 1;

class KVCacheBlock;
class KVCacheManager;
class KVCacheTransferManager;

using SizeType32 = sugesstify::runtime::SizeType32;
using TokenIdType = sugesstify::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using BlockPtr = std::shared_ptr<KVCacheBlock>;
using FreeBlocksQueue = std::list<BlockPtr>;
using UniqueToken = sugesstify::runtime::UniqueToken;
using VecUniqueTokens = sugesstify::runtime::VecUniqueTokens;
using LoraTaskIdType = sugesstify::runtime::LoraTaskIdType;

template <typename T>
using OptionalRef = sugesstify::common::OptionalRef<T>;

struct BlockKey
{
    bool hasLora;
    LoraTaskIdType loraTaskId;
    VecUniqueTokens uniqueTokens;

    BlockKey() = default;

    explicit BlockKey(bool hasLora, LoraTaskIdType loraTaskId, VecUniqueTokens uniqueTokens)
        : hasLora{hasLora}
        , loraTaskId{loraTaskId}
        , uniqueTokens{std::move(uniqueTokens)}
    {
    }

    bool operator==(BlockKey const& other) const noexcept
    {
        return (hasLora == other.hasLora && loraTaskId == other.loraTaskId && uniqueTokens == other.uniqueTokens);
    }
};

struct BlockKeyHasher
{
    std::size_t operator()(BlockKey const& blockKey, std::size_t parentHash = 0) const noexcept
    {
        size_t seed = blockKey.uniqueTokens.size() ^ parentHash * UINT64_C(0xbf58476d1ce4e5b9);

        for (auto const& uniqueToken : blockKey.uniqueTokens)
        {
            uint32_t a = static_cast<uint32_t>(uniqueToken.tokenId);
            a = ((a >> 16) ^ a) * 0x45d9f3b;
            a = ((a >> 16) ^ a) * 0x45d9f3b;
            a = (a >> 16) ^ a;

            uint64_t b = uniqueToken.tokenExtraId;
            b = (b ^ (b >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            b = (b ^ (b >> 27)) * UINT64_C(0x94d049bb133111eb);
            b = b ^ (b >> 31);

            seed ^= a + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= b + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        uint64_t c = blockKey.loraTaskId;
        c = (c ^ (c >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        c = (c ^ (c >> 27)) * UINT64_C(0x94d049bb133111eb);
        c = c ^ (c >> 31);
        seed ^= c + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        uint32_t d = static_cast<uint32_t>(blockKey.hasLora);
        d = ((d >> 16) ^ d) * 0x45d9f3b;
        d = ((d >> 16) ^ d) * 0x45d9f3b;
        d = (d >> 16) ^ d;
        seed ^= d + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        return seed;
    }
};

using NextBlockMap = std::unordered_map<BlockKey, BlockPtr, BlockKeyHasher>;

struct KvCacheStats
{
    SizeType32 maxNumBlocks;
    SizeType32 freeNumBlocks;
    SizeType32 usedNumBlocks;
    SizeType32 toksPerBlock;
    SizeType32 allocTotalBlocks;
    SizeType32 allocNewBlocks;
    SizeType32 reusedBlocks;
    SizeType32 missedBlocks;
    float cacheHitRate;
};

class KVCacheBlock
{
public:
    using IdType = std::int32_t;

    explicit KVCacheBlock(IdType blockId, kernels::KVCacheIndex blockIdx);

    void startScheduling();

    [[nodiscard]] IdType getBlockId() const;

    [[nodiscard]] NextBlockMap getNextBlocks() const;

    [[nodiscard]] kernels::KVCacheIndex::UnderlyingType getMemoryPoolBlockIndex() const;

    [[nodiscard]] bool isPrimary() const;

    void swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock);

    void incRefCount();

    void decRefCount();

    void decSchedulingRefCount();

    [[nodiscard]] bool hasRefs() const;

    [[nodiscard]] bool hasSchedulingRefs() const;

    void setBlockKey(BlockKey const& blockKey, bool isFull);

    BlockKey getBlockKey();

    [[nodiscard]] VecUniqueTokens const& getUniqueTokens() const;

    BlockPtr getPrevBlock() const;

    void setPrevBlock(BlockPtr prevBlock);

    void addNextBlock(BlockKey const& blockKey, BlockPtr block);

    void removeNextBlock(BlockKey const& blockKey);

    [[nodiscard]] BlockPtr findMatchingBlock(BlockKey const& blockKey) const;

    void freeLeafBlock();

    [[nodiscard]] bool isFull() const;

    [[nodiscard]] bool isShared() const;

    void setPriority(executor::RetentionPriority priority);

    [[nodiscard]] executor::RetentionPriority getPriority() const;

    void setDurationMs(std::optional<std::chrono::milliseconds> durationMs);

    [[nodiscard]] std::optional<std::chrono::milliseconds> getDurationMs() const;

    void setExpirationTime(std::optional<std::chrono::steady_clock::time_point::duration> expirationTime);

    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point::duration> getExpirationTime() const;

    void setHash(size_t hash);

    size_t getHash() const;

private:
    IdType mBlockId;

    kernels::KVCacheIndex mMemoryPoolBlockIndex;

    SizeType32 mRefCount;

    SizeType32 mSchedulingRefCount;

    BlockKey mBlockKey;

    BlockPtr mPrevBlock;

    NextBlockMap mNextBlocks;

    std::optional<FreeBlocksQueue::iterator> mFreeBlockIterator;

    bool mIsFull;

    executor::RetentionPriority mPriority;
    std::optional<std::chrono::milliseconds> mDurationMs;
    std::optional<std::chrono::steady_clock::time_point::duration> mExpirationTime;
    size_t mHash;
};

class GenerationRequest
{
public:
    using SizeType32 = sugesstify::runtime::SizeType32;

    explicit GenerationRequest(LlmRequest::RequestIdType requestId, SizeType32 numTokens, SizeType32 beamWidth,
        SizeType32 maxBlocks, SizeType32 numPools = 1,
        executor::KvCacheRetentionConfig kvCacheRetentionConfig = executor::KvCacheRetentionConfig())
        : mRequestId(requestId)
        , mNumTokens(numTokens)
        , mBeamWidth(beamWidth)
        , mCacheBlockIds(beamWidth)
        , mCacheBlockIndices{runtime::BufferManager::cpu(
              runtime::ITensor::makeShape({numPools, beamWidth, 2, maxBlocks}),
              runtime::TRTDataType<sugesstify::kernels::KVCacheIndex>::value)}
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
    {
        auto cacheBlockIdsRange = runtime::BufferRange<sugesstify::kernels::KVCacheIndex>(*mCacheBlockIndices);
        std::fill(cacheBlockIdsRange.begin(), cacheBlockIdsRange.end(),
            sugesstify::kernels::KVCacheIndex{
                std::numeric_limits<sugesstify::kernels::KVCacheIndex::UnderlyingType>::max()});
    }

    void addNewTokens(SizeType32 n)
    {
        mNumTokens += n;
    }

    void removeTokens(SizeType32 n)
    {
        CHECK(n <= mNumTokens);
        CHECK(mNumTokens - n >= 0);
        mNumTokens -= n;
    }

    [[nodiscard]] LlmRequest::RequestIdType getRequestId() const
    {
        return mRequestId;
    }

    [[nodiscard]] SizeType32 getNumTokens() const
    {
        return mNumTokens;
    }

    [[nodiscard]] SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] std::vector<std::vector<SizeType32>> const& getCacheBlockIds() const
    {
        return mCacheBlockIds;
    }

    [[nodiscard]] runtime::ITensor& getCacheBlockIndices()
    {
        return *mCacheBlockIndices;
    }

    [[nodiscard]] runtime::ITensor const& getCacheBlockIndices() const
    {
        return *mCacheBlockIndices;
    }

    void addCacheBlock(SizeType32 beamIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(beamIdx).push_back(blockId);
    }

    void changeCacheBlock(SizeType32 beamIdx, SizeType32 pagedBlockIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(beamIdx).at(pagedBlockIdx) = blockId;
    }

    void clearCacheBlocks()
    {
        for (auto& beamBlockIds : mCacheBlockIds)
        {
            beamBlockIds.clear();
        }
    }

    void removeLastBlock()
    {
        for (auto& beamBlockIds : mCacheBlockIds)
        {
            beamBlockIds.pop_back();
        }
    }

    [[nodiscard]] executor::RetentionPriority getDecodeRetentionPriority() const
    {
        return mKvCacheRetentionConfig.getDecodeRetentionPriority();
    }

    [[nodiscard]] std::optional<std::chrono::milliseconds> getDecodeDurationMs() const
    {
        return mKvCacheRetentionConfig.getDecodeDurationMs();
    }

private:
    LlmRequest::RequestIdType mRequestId;
    SizeType32 mNumTokens;
    SizeType32 mBeamWidth;
    std::vector<std::vector<KVCacheBlock::IdType>> mCacheBlockIds;
    runtime::ITensor::SharedPtr mCacheBlockIndices;
    executor::KvCacheRetentionConfig mKvCacheRetentionConfig;
};

class KVCacheBlockPool
{
public:
    SizeType32 numKvHeads;
    SizeType32 numLayers;
    SizeType32 blockSize;

    runtime::ITensor::SharedPtr primaryPtr;
    runtime::ITensor::SharedPtr secondaryPtr;

    KVCacheBlockPool(SizeType32 numKvHeads, SizeType32 numLayers, SizeType32 blockSize,
        runtime::ITensor::SharedPtr primaryPtr = nullptr, runtime::ITensor::SharedPtr secondaryPtr = nullptr)
        : numKvHeads(numKvHeads)
        , numLayers(numLayers)
        , blockSize(blockSize)
        , primaryPtr(std::move(primaryPtr))
        , secondaryPtr(std::move(secondaryPtr))
    {
    }
};

class BlockManager
{
public:
    using SizeType32 = sugesstify::runtime::SizeType32;
    using CacheType = sugesstify::batch_manager::kv_cache_manager::CacheType;
    using BaseEvictionPolicy = sugesstify::batch_manager::eviction_policy::BaseEvictionPolicy;

    explicit BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
        SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream, bool onboardBlocks,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr);

    ~BlockManager();

    void allocatePools(nvinfer1::DataType dtype, bool useUvm);

    void startScheduling();

    void addSequence(
        GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest);

    void addSequence(GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx);

    void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams = false);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx);

    void releaseBlocks(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest = std::nullopt);

    void schedulingReleaseBlocks(GenerationRequest& sequence);

    void releaseLastBlock(GenerationRequest& sequence);

    [[nodiscard]] SizeType32 getNumFreeBlocks() const noexcept;

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return mAllocTotalBlocks;
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return mAllocNewBlocks;
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept
    {
        return mReusedBlocks;
    }

    [[nodiscard]] SizeType32 getNumAllocatedBlocks() const noexcept
    {
        return getMaxNumBlocks() - getNumFreeBlocks();
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const noexcept
    {
        return mMissedBlocks;
    }

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout) const;

    [[nodiscard]] bool hasFreeBlocks(SizeType32 numRequired = 1) const noexcept
    {
        return getNumFreeBlocks() >= numRequired;
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const noexcept
    {
        return mSchedulingNumFreeBlocks >= numRequired;
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const noexcept
    {
        return static_cast<SizeType32>(mAllBlocksById.size());
    }

    [[nodiscard]] SizeType32 getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    [[nodiscard]] SizeType32 getBlockSize(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).blockSize;
    }

    [[nodiscard]] SizeType32 getNumPools() const noexcept
    {
        return mPools.size();
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).primaryPtr;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getSecondaryPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).secondaryPtr;
    }

    [[nodiscard]] SizeType32 getNumLayers() const
    {
        return mNumLayers;
    }

    [[nodiscard]] SizeType32 getNumPrimaryBlocks() const
    {
        return mNumPrimaryBlocks;
    }

    [[nodiscard]] SizeType32 getNumSecondaryBlocks() const
    {
        return mNumSecondaryBlocks;
    }

    [[nodiscard]] CacheType getCacheType() const
    {
        return mCacheType;
    }

    [[nodiscard]] SizeType32 getLayerPoolIdx(SizeType32 layerIdx) const
    {
        return mLayerToPool.at(layerIdx);
    }

    [[nodiscard]] SizeType32 getPoolLayerIdx(SizeType32 layerIdx) const
    {
        return mLayerIndexToPoolLayerIndex.at(layerIdx);
    }

    [[nodiscard]] kernels::KVCacheIndex getKOrVBlockIndex(
        KVCacheBlock::IdType blockId, SizeType32 fieldIdx, SizeType32 poolIdx) const;

    void onboardBlock(BlockPtr offloadBlock);

    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const;

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

    void refreshBlocks();

    void flushIterationEvents()
    {
        if (mEventManager)
        {
            mEventManager->flush();
        }
    }

    [[nodiscard]] static bool blockInRadixTree(BlockPtr const& block);

private:
    void addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType32 beamIdx);

    void addBlockToAllBeams(BlockPtr& block, GenerationRequest& sequence);

    void storeBlocks(std::vector<BlockKey> blockKeys, std::vector<KVCacheBlock::IdType> const& blockIds);

    SizeType32 loadOrAllocateBlocks(std::vector<BlockKey> const& blockKeys, SizeType32 numContextBlocks,
        GenerationRequest& sequence, std::vector<executor::RetentionPriorityAndDuration> const& perBlockRetentions);

    [[nodiscard]] BlockPtr getFreeBlock(
        executor::RetentionPriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority,
        std::optional<std::chrono::milliseconds> durationMs = std::nullopt);

    void claimLeafBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority = std::nullopt,
        std::optional<std::chrono::milliseconds> durationMs = std::nullopt);

private:
    SizeType32 mNumPrimaryBlocks;
    SizeType32 mNumSecondaryBlocks;

    std::unordered_map<LlmRequest::RequestIdType, std::vector<BlockPtr>> mAllocatedBlocksPerSeq;

    std::vector<KVCacheBlockPool> mPools;
    std::vector<SizeType32> mLayerToPool;
    std::vector<SizeType32> mLayerIndexToPoolLayerIndex;

    bool mOnboardBlocks;
    runtime::BufferManager mBufferManager;

    SizeType32 mSizePerHead;
    SizeType32 mNumLayers;
    SizeType32 mSchedulingNumFreeBlocks;
    SizeType32 mTokensPerBlock;
    std::vector<BlockPtr> mAllBlocksById;
    BlockPtr mCachedBlocksRoot;
    CacheType mCacheType;
    std::shared_ptr<BaseEvictionPolicy> mEvictionPolicy;
    std::shared_ptr<KVCacheEventManager> mEventManager;
    std::shared_ptr<KVCacheTransferManager> mTransferManager;

    SizeType32 mAllocTotalBlocks;
    SizeType32 mAllocNewBlocks;
    SizeType32 mReusedBlocks;
    SizeType32 mReusedUniqueBlocks;
    SizeType32 mMissedBlocks;
    std::set<KVCacheBlock::IdType> reusedBlockIds;

private:
    friend class KVCacheManager;
};

class BaseKVCacheManager
{
public:
    using SizeType32 = sugesstify::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = sugesstify::batch_manager::kv_cache_manager::CacheType;

    virtual ~BaseKVCacheManager() {}

    virtual void allocatePools(nvinfer1::DataType dtype, bool useUvm = false) = 0;

    virtual void startScheduling() = 0;

    [[nodiscard]] virtual SizeType32 getTokensPerBlock() const = 0;

    [[nodiscard]] virtual SizeType32 getMaxNumBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getUsedNumBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getNumFreeBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getNumPools() const = 0;

    [[nodiscard]] virtual SizeType32 getNumReusedBlocks() const noexcept = 0;
    [[nodiscard]] virtual KvCacheStats getKvCacheStats() const = 0;

    [[nodiscard]] virtual SizeType32 getMaxBlocksPerSeq() const = 0;

    [[nodiscard]] virtual std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const
        = 0;

    [[nodiscard]] virtual BlockManager const& getBlockManager() const = 0;

    [[nodiscard]] virtual SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const = 0;

    [[nodiscard]] virtual SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req) const = 0;

    virtual void addToken(LlmRequest::RequestIdType requestId) = 0;

    virtual void addSequence(LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        OptionalRef<LlmRequest> llmRequest = std::nullopt)
        = 0;

    virtual void removeSequence(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest = std::nullopt)
        = 0;

    virtual void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getBlockPoolPointers() const = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getLayerToPoolMapping() const = 0;

    virtual void getBlockOffsetsOfBatch(
        runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
        = 0;

    virtual SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual bool isEnableBlockReuse() const = 0;

    [[nodiscard]] virtual bool isUseOneMoreBlock() const = 0;

    virtual void rewindKVCache(LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) = 0;

    [[nodiscard]] virtual GenerationRequest const& getSequence(LlmRequest::RequestIdType requestId) const = 0;

    [[nodiscard]] virtual bool isCrossKv() const = 0;

    [[nodiscard]] virtual std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
        = 0;

    virtual void storeContextBlocks(LlmRequest const& llmRequest) = 0;

    virtual bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const = 0;

    virtual std::vector<std::vector<SizeType32>> const& getCacheBlockIds(LlmRequest::RequestIdType requestId) const = 0;

    virtual std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds) const
        = 0;

    virtual runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const = 0;
    virtual SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const = 0;

    virtual void refreshBlocks() = 0;
    virtual void flushIterationEvents() = 0;

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    [[nodiscard]] static SizeType32 calculateCacheSizePerToken(sugesstify::runtime::ModelConfig const& modelConfig,
        sugesstify::runtime::WorldConfig const& worldConfig, bool isCrossAttention = false)
    {
        return modelConfig.getSumLocalKvHeads(
                   worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank(), isCrossAttention)
            * 2 * modelConfig.getSizePerHead();
    }

    [[nodiscard]] static std::tuple<SizeType32, SizeType32> const calculateMaxNumBlocks(KvCacheConfig const& config,
        nvinfer1::DataType dtype, sugesstify::runtime::ModelConfig const& modelConfig,
        sugesstify::runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager);

    [[nodiscard]] virtual SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const = 0;
};

class KVCacheManager : public BaseKVCacheManager
{
public:
    friend class KVCacheManagerBindings;

    using SizeType32 = sugesstify::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = sugesstify::batch_manager::kv_cache_manager::CacheType;

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 temporaryAttentionWindow,
        SizeType32 sinkTokenLength, CudaStreamPtr stream, std::optional<SizeType32> maxSequenceLength,
        bool enableBlockReuse = false, bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr);

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 temporaryAttentionWindow,
        SizeType32 sinkTokenLength, int64_t stream, std::optional<SizeType32> maxSequenceLength,
        bool enableBlockReuse = false, bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF);

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 temporaryAttentionWindow,
        SizeType32 sinkTokenLength, CudaStreamPtr stream, std::optional<SizeType32> maxSequenceLength,
        bool enableBlockReuse = true, bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr);

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 temporaryAttentionWindow,
        SizeType32 sinkTokenLength, int64_t stream, std::optional<SizeType32> maxSequenceLength,
        bool enableBlockReuse = false, bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF);

    ~KVCacheManager() {}

    void allocatePools(nvinfer1::DataType dtype, bool useUvm = false) override;

    void startScheduling() override;

    [[nodiscard]] SizeType32 getTokensPerBlock() const override
    {
        return mBlockManager.getTokensPerBlock();
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const override
    {
        return mBlockManager.getMaxNumBlocks();
    }

    [[nodiscard]] SizeType32 getUsedNumBlocks() const override
    {
        return mBlockManager.getNumAllocatedBlocks();
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const override
    {
        return mBlockManager.getNumFreeBlocks();
    }

    [[nodiscard]] virtual SizeType32 getNumPools() const override
    {
        return mBlockManager.getNumPools();
    }

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return mBlockManager.getNumAllocTotalBlocks();
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return mBlockManager.getNumAllocNewBlocks();
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept
    {
        return mBlockManager.getNumReusedBlocks();
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const noexcept
    {
        return mBlockManager.getNumMissedBlocks();
    }

    [[nodiscard]] KvCacheStats getKvCacheStats() const
    {
        KvCacheStats kvCacheStats;
        kvCacheStats.maxNumBlocks = getMaxNumBlocks();
        kvCacheStats.freeNumBlocks = getNumFreeBlocks();
        kvCacheStats.usedNumBlocks = getUsedNumBlocks();
        kvCacheStats.toksPerBlock = getTokensPerBlock();
        kvCacheStats.allocTotalBlocks = getNumAllocTotalBlocks();
        kvCacheStats.allocNewBlocks = getNumAllocNewBlocks();
        kvCacheStats.reusedBlocks = getNumReusedBlocks();
        kvCacheStats.missedBlocks = getNumMissedBlocks();
        kvCacheStats.cacheHitRate = kvCacheStats.reusedBlocks == 0 ? 0
                                                                   : static_cast<float>(kvCacheStats.reusedBlocks)
                / static_cast<float>(kvCacheStats.reusedBlocks + kvCacheStats.missedBlocks);
        return kvCacheStats;
    }

    [[nodiscard]] SizeType32 getMaxBlocksPerSeq() const override
    {
        return mMaxBlocksPerSeq;
    }

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const
    {
        return mBlockManager.getLatestEvents(timeout);
    }

    [[nodiscard]] BlockManager const& getBlockManager() const
    {
        return mBlockManager;
    }

    [[nodiscard]] SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const override;

    [[nodiscard]] SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req) const override;

    void addContextTokens(LlmRequest::RequestIdType requestId, SizeType32 numTokens);

    void addToken(LlmRequest::RequestIdType requestId) override;

    void addSequence(LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        OptionalRef<LlmRequest> llmRequest = std::nullopt) override;

    void removeSequence(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest = std::nullopt) override;

    void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) override;

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        return mBlockPoolPointers;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        return mLayerToPoolMapping;
    }

    void getBlockOffsetsOfBatch(runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
        SizeType32 beamWidth) const override;

    SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const override;

    [[nodiscard]] bool isEnableBlockReuse() const override
    {
        return mEnableBlockReuse;
    }

    [[nodiscard]] bool isUseOneMoreBlock() const override
    {
        return mUseOneMoreBlock;
    }

    void removeToken(LlmRequest::RequestIdType requestId);
    void rewindKVCache(LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) override;

    [[nodiscard]] GenerationRequest const& getSequence(LlmRequest::RequestIdType requestId) const override;

    [[nodiscard]] bool isCrossKv() const override
    {
        return mBlockManager.getCacheType() == CacheType::kCROSS;
    }

    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const override;

    void storeContextBlocks(LlmRequest const& llmRequest) override;

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override;

    [[nodiscard]] static SizeType32 calculateMaxBlockRequirements(SizeType32 inputLength, SizeType32 outputLength,
        SizeType32 sinkTokenLength, SizeType32 maxAttentionWindow, SizeType32 beamWidth, SizeType32 tokensPerBlock);

    [[nodiscard]] static SizeType32 calculateMaxBlockRequirementsPerBeam(SizeType32 sequenceLength,
        SizeType32 sinkTokenLength, SizeType32 maxAttentionWindow, SizeType32 tokensPerBlock);

    bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const override;

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(LlmRequest::RequestIdType requestId) const override;

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds) const override;

    runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const override;

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override
    {
        return mBlockManager.getPoolLayerIdx(layer_idx);
    }

    void refreshBlocks() override
    {
        mBlockManager.refreshBlocks();
    }

    void flushIterationEvents() override
    {
        mBlockManager.flushIterationEvents();
    }

    [[nodiscard]] static SizeType32 calculateMaxAttentionWindow(SizeType32 inputLength, SizeType32 outputLength,
        SizeType32 sinkTokenLength, SizeType32 blockCapacity, SizeType32 beamWidth, SizeType32 tokensPerBlock);

private:
    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
        SizeType32 blockIdx, KVCacheBlock::IdType blockId) const;

    void cacheBlockOffsets(GenerationRequest& seq);
    void cacheNewBlockOffsets(GenerationRequest& seq);
    void updateNewBlockPointer(GenerationRequest& seq, SizeType32 blockIdx);
    void updateToken(GenerationRequest& sequence, bool addToken);

private:
    SizeType32 mMaxNumSequences;
    SizeType32 mMaxBeamWidth;
    SizeType32 mMaxBlocksPerSeq;
    SizeType32 mMaxAttentionWindow;
    SizeType32 mTemporaryAttentionWindow;
    SizeType32 mTokensPerBlock;
    SizeType32 mSinkBubbleLength;
    SizeType32 mMaxTokenNum;
    SizeType32 mSinkBlockTokenLength;
    BlockManager mBlockManager;
    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;
    bool mEnableBlockReuse;
    bool mUseOneMoreBlock;
    runtime::ITensor::SharedPtr mBlockPoolPointers;
    runtime::ITensor::SharedPtr mLayerToPoolMapping;
};

}
