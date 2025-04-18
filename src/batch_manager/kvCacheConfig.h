
#pragma once

#include "../executor/executor.h"
#include "../runtime/common.h"

#include <optional>

namespace sugesstify::batch_manager::kv_cache_manager
{

enum class CacheType
{
    kSELF = 0,
    kCROSS = 1,
};

class KvCacheConfig
{
public:
    using SizeType32 = sugesstify::runtime::SizeType32;

    explicit KvCacheConfig(std::optional<SizeType32> maxTokens = std::nullopt,
        std::optional<std::vector<SizeType32>> maxAttentionWindowVec = std::nullopt,
        std::optional<SizeType32> sinkTokenLength = std::nullopt,
        std::optional<float> freeGpuMemoryFraction = std::nullopt, bool enableBlockReuse = false, bool useUvm = false,
        std::optional<size_t> hostCacheSize = std::nullopt, bool onboardBlocks = true,
        std::optional<float> crossKvCacheFraction = std::nullopt,
        std::optional<SizeType32> secondaryOffloadMinPriority = std::nullopt, size_t eventBufferMaxSize = 0)
        : maxTokens{maxTokens}
        , maxAttentionWindowVec{std::move(maxAttentionWindowVec)}
        , sinkTokenLength{sinkTokenLength}
        , freeGpuMemoryFraction{freeGpuMemoryFraction}
        , enableBlockReuse(enableBlockReuse)
        , useUvm(useUvm)
        , hostCacheSize(hostCacheSize)
        , onboardBlocks(onboardBlocks)
        , crossKvCacheFraction{crossKvCacheFraction}
        , secondaryOffloadMinPriority(secondaryOffloadMinPriority)
        , eventBufferMaxSize(eventBufferMaxSize)
    {
    }

    explicit KvCacheConfig(executor::KvCacheConfig const& kvCacheConfig)
        : KvCacheConfig(kvCacheConfig.getMaxTokens(), kvCacheConfig.getMaxAttentionWindowVec(),
            kvCacheConfig.getSinkTokenLength(), kvCacheConfig.getFreeGpuMemoryFraction(),
            kvCacheConfig.getEnableBlockReuse(), false, kvCacheConfig.getHostCacheSize(),
            kvCacheConfig.getOnboardBlocks(), kvCacheConfig.getCrossKvCacheFraction(),
            kvCacheConfig.getSecondaryOffloadMinPriority(), kvCacheConfig.getEventBufferMaxSize())
    {
    }

    bool operator==(KvCacheConfig const& other) const
    {
        return maxTokens == other.maxTokens && maxAttentionWindowVec == other.maxAttentionWindowVec
            && sinkTokenLength == other.sinkTokenLength && freeGpuMemoryFraction == other.freeGpuMemoryFraction
            && enableBlockReuse == other.enableBlockReuse && useUvm == other.useUvm
            && hostCacheSize == other.hostCacheSize && onboardBlocks == other.onboardBlocks
            && crossKvCacheFraction == other.crossKvCacheFraction
            && secondaryOffloadMinPriority == other.secondaryOffloadMinPriority
            && eventBufferMaxSize == other.eventBufferMaxSize;
    }

    friend std::ostream& operator<<(std::ostream& os, KvCacheConfig const& self);

    std::optional<SizeType32> maxTokens;
    std::optional<std::vector<SizeType32>> maxAttentionWindowVec;
    std::optional<SizeType32> sinkTokenLength;
    std::optional<float> freeGpuMemoryFraction;
    bool enableBlockReuse;
    static constexpr auto kDefaultGpuMemFraction = 0.9F;
    bool useUvm;
    std::optional<size_t> hostCacheSize;
    bool onboardBlocks;
    std::optional<float> crossKvCacheFraction;
    std::optional<SizeType32> secondaryOffloadMinPriority;
    size_t eventBufferMaxSize;
};
}
