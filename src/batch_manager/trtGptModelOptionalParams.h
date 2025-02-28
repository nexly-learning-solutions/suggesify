
#pragma once

#include "../kvCacheConfig.h"
#include "../peftCacheManagerConfig.h"
#include "../executor/executor.h"
#include "../runtime/common.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace sugesstify::batch_manager
{

class TrtGptModelOptionalParams
{
    using KvCacheConfig = kv_cache_manager::KvCacheConfig;

public:
    using SizeType32 = sugesstify::runtime::SizeType32;

    explicit TrtGptModelOptionalParams(KvCacheConfig kvCacheConfig = KvCacheConfig{}, bool enableTrtOverlap = false,
        std::optional<std::vector<SizeType32>> deviceIds = std::nullopt, bool normalizeLogProbs = true,
        bool enableChunkedContext = true,
        PeftCacheManagerConfig const& peftCacheManagerConfig = PeftCacheManagerConfig{},
        executor::DecodingConfig decodingConfig = executor::DecodingConfig{}, float gpuWeightsPercent = 1,
        std::optional<SizeType32> maxBeamWidth = std::nullopt, std::optional<SizeType32> maxBatchSize = std::nullopt,
        std::optional<SizeType32> maxNumTokens = std::nullopt,
        executor::SchedulerConfig schedulerConfig = executor::SchedulerConfig{},
        executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig
        = executor::ExtendedRuntimePerfKnobConfig{},
        std::optional<executor::DebugConfig> debugConfig = std::nullopt,
        uint64_t maxSeqIdleMicroseconds = executor::ExecutorConfig::kDefaultMaxSeqIdleMicroseconds,
        std::optional<executor::SpeculativeDecodingConfig> specDecConfig = std::nullopt,
        std::optional<executor::GuidedDecodingConfig> guidedDecodingConfig = std::nullopt,
        bool isLeaderInOrchMode = false, std::optional<std::vector<std::string>> additionalOutputNames = std::nullopt)
        : kvCacheConfig{std::move(kvCacheConfig)}
        , enableTrtOverlap{enableTrtOverlap}
        , deviceIds(std::move(deviceIds))
        , normalizeLogProbs{normalizeLogProbs}
        , enableChunkedContext{enableChunkedContext}
        , peftCacheManagerConfig(peftCacheManagerConfig)
        , decodingConfig(std::move(decodingConfig))
        , gpuWeightsPercent(gpuWeightsPercent)
        , maxBeamWidth(maxBeamWidth)
        , maxBatchSize(maxBatchSize)
        , maxNumTokens(maxNumTokens)
        , schedulerConfig{std::move(schedulerConfig)}
        , extendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig)
        , debugConfig{std::move(debugConfig)}
        , maxSeqIdleMicroseconds{maxSeqIdleMicroseconds}
        , speculativeDecodingConfig{specDecConfig}
        , guidedDecodingConfig{std::move(guidedDecodingConfig)}
        , isLeaderInOrchMode{isLeaderInOrchMode}
        , additionalOutputNames{std::move(additionalOutputNames)}
    {
        if (guidedDecodingConfig)
        {
            guidedDecodingConfig->validate();
        }
    }

    explicit TrtGptModelOptionalParams(executor::ExecutorConfig const& executorConfig, bool isLeaderInOrchMode)
        : TrtGptModelOptionalParams(KvCacheConfig(executorConfig.getKvCacheConfig()), false,
            executorConfig.getParallelConfig().value_or(executor::ParallelConfig()).getDeviceIds(),
            executorConfig.getNormalizeLogProbs(), executorConfig.getEnableChunkedContext(),
            PeftCacheManagerConfig(executorConfig.getPeftCacheConfig().value_or(executor::PeftCacheConfig())),
            executorConfig.getDecodingConfig().value_or(executor::DecodingConfig{}),
            executorConfig.getGpuWeightsPercent(), executorConfig.getMaxBeamWidth(), executorConfig.getMaxBatchSize(),
            executorConfig.getMaxNumTokens(), executorConfig.getSchedulerConfig(),
            executorConfig.getExtendedRuntimePerfKnobConfig(), executorConfig.getDebugConfig(),
            executorConfig.getMaxSeqIdleMicroseconds(), executorConfig.getSpecDecConfig(),
            executorConfig.getGuidedDecodingConfig(), isLeaderInOrchMode, executorConfig.getAdditionalOutputNames())
    {
    }

    bool operator==(TrtGptModelOptionalParams const& other) const
    {
        return kvCacheConfig == other.kvCacheConfig
            && enableTrtOverlap == other.enableTrtOverlap
            && deviceIds == other.deviceIds
            && normalizeLogProbs == other.normalizeLogProbs
            && enableChunkedContext == other.enableChunkedContext
            && decodingConfig == other.decodingConfig
            && gpuWeightsPercent == other.gpuWeightsPercent
            && maxBeamWidth == other.maxBeamWidth
            && maxBatchSize == other.maxBatchSize
            && maxNumTokens == other.maxNumTokens
            && schedulerConfig == other.schedulerConfig
            && extendedRuntimePerfKnobConfig == other.extendedRuntimePerfKnobConfig
            && debugConfig == other.debugConfig
            && maxSeqIdleMicroseconds == other.maxSeqIdleMicroseconds
            && speculativeDecodingConfig == other.speculativeDecodingConfig
            && guidedDecodingConfig == other.guidedDecodingConfig
            && isLeaderInOrchMode == other.isLeaderInOrchMode
            && additionalOutputNames == other.additionalOutputNames
            ;
    }

    friend std::ostream& operator<<(std::ostream& os, TrtGptModelOptionalParams const& self);

    KvCacheConfig kvCacheConfig;

    bool enableTrtOverlap;
    std::optional<std::vector<SizeType32>> deviceIds;
    bool normalizeLogProbs;
    bool enableChunkedContext;
    PeftCacheManagerConfig peftCacheManagerConfig;
    executor::DecodingConfig decodingConfig;
    float gpuWeightsPercent;
    std::optional<SizeType32> maxBeamWidth;
    std::optional<SizeType32> maxBatchSize;
    std::optional<SizeType32> maxNumTokens;
    executor::SchedulerConfig schedulerConfig;
    executor::ExtendedRuntimePerfKnobConfig extendedRuntimePerfKnobConfig;
    std::optional<executor::DebugConfig> debugConfig;
    uint64_t maxSeqIdleMicroseconds;
    std::optional<executor::SpeculativeDecodingConfig> speculativeDecodingConfig;
    std::optional<executor::GuidedDecodingConfig> guidedDecodingConfig;
    bool isLeaderInOrchMode;
    std::optional<std::vector<std::string>> additionalOutputNames;
};

}
