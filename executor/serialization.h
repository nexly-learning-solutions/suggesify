
#pragma once

#include "../executor/executor.h"
#include "../executor/tensor.h"
#include "../executor/types.h"
#include <istream>
#include <ostream>

namespace suggestify::executor
{

namespace kv_cache
{
class CommState;
class CacheState;
class SocketState;
}

class Serialization
{
public:
    [[nodiscard]] static RequestPerfMetrics::TimePoint deserializeTimePoint(std::istream& is);
    static void serialize(RequestPerfMetrics::TimePoint const& tp, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestPerfMetrics::TimePoint const&);

    [[nodiscard]] static RequestPerfMetrics deserializeRequestPerfMetrics(std::istream& is);
    static void serialize(RequestPerfMetrics const& metrics, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestPerfMetrics const& metrics);

    [[nodiscard]] static SamplingConfig deserializeSamplingConfig(std::istream& is);
    static void serialize(SamplingConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(SamplingConfig const& config);

    [[nodiscard]] static OutputConfig deserializeOutputConfig(std::istream& is);
    static void serialize(OutputConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(OutputConfig const& config);

    [[nodiscard]] static OutputConfig::AdditionalModelOutput deserializeAdditionalModelOutput(std::istream& is);
    static void serialize(OutputConfig::AdditionalModelOutput const& additionalModelOutput, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(OutputConfig::AdditionalModelOutput const& additionalModelOutput);

    [[nodiscard]] static ExternalDraftTokensConfig deserializeExternalDraftTokensConfig(std::istream& is);
    static void serialize(ExternalDraftTokensConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(ExternalDraftTokensConfig const& config);

    [[nodiscard]] static PromptTuningConfig deserializePromptTuningConfig(std::istream& is);
    static void serialize(PromptTuningConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(PromptTuningConfig const& config);

    [[nodiscard]] static MropeConfig deserializeMropeConfig(std::istream& is);
    static void serialize(MropeConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(MropeConfig const& config);

    [[nodiscard]] static LoraConfig deserializeLoraConfig(std::istream& is);
    static void serialize(LoraConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(LoraConfig const& config);

    [[nodiscard]] static kv_cache::CommState deserializeCommState(std::istream& is);
    static void serialize(kv_cache::CommState const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(kv_cache::CommState const& state);

    [[nodiscard]] static kv_cache::SocketState deserializeSocketState(std::istream& is);
    static void serialize(kv_cache::SocketState const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(kv_cache::SocketState const& state);

    [[nodiscard]] static kv_cache::CacheState deserializeCacheState(std::istream& is);
    static void serialize(kv_cache::CacheState const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(kv_cache::CacheState const& state);

    [[nodiscard]] static DataTransceiverState deserializeDataTransceiverState(std::istream& is);
    static void serialize(DataTransceiverState const& dataTransceiverState, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(DataTransceiverState const& dataTransceiverState);

    [[nodiscard]] static ContextPhaseParams deserializeContextPhaseParams(std::istream& is);
    static void serialize(ContextPhaseParams const& contextPhaseParams, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(ContextPhaseParams const& contextPhaseParams);

    [[nodiscard]] static Request deserializeRequest(std::istream& is);
    static void serialize(Request const& request, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Request const& request);

    [[nodiscard]] static Tensor deserializeTensor(std::istream& is);
    static void serialize(Tensor const& tensor, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Tensor const& tensor);

    [[nodiscard]] static SpeculativeDecodingFastLogitsInfo deserializeSpecDecFastLogitsInfo(std::istream& is);
    static void serialize(SpeculativeDecodingFastLogitsInfo const& info, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(SpeculativeDecodingFastLogitsInfo const& info);

    [[nodiscard]] static Result deserializeResult(std::istream& is);
    static void serialize(Result const& result, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Result const& result);

    [[nodiscard]] static AdditionalOutput deserializeAdditionalOutput(std::istream& is);
    static void serialize(AdditionalOutput const& additionalOutput, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(AdditionalOutput const& additionalOutput);

    [[nodiscard]] static Response deserializeResponse(std::istream& is);
    static void serialize(Response const& response, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Response const& response);

    static std::vector<Response> deserializeResponses(std::vector<char>& buffer);
    static std::vector<char> serialize(std::vector<Response> const& responses);

    static KvCacheConfig deserializeKvCacheConfig(std::istream& is);
    static void serialize(KvCacheConfig const& kvCacheConfig, std::ostream& os);
    static size_t serializedSize(KvCacheConfig const& kvCacheConfig);

    static DynamicBatchConfig deserializeDynamicBatchConfig(std::istream& is);
    static void serialize(DynamicBatchConfig const& dynamicBatchConfig, std::ostream& os);
    static size_t serializedSize(DynamicBatchConfig const& dynamicBatchConfig);

    static SchedulerConfig deserializeSchedulerConfig(std::istream& is);
    static void serialize(SchedulerConfig const& schedulerConfig, std::ostream& os);
    static size_t serializedSize(SchedulerConfig const& schedulerConfig);

    static ExtendedRuntimePerfKnobConfig deserializeExtendedRuntimePerfKnobConfig(std::istream& is);
    static void serialize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig, std::ostream& os);
    static size_t serializedSize(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig);

    static ParallelConfig deserializeParallelConfig(std::istream& is);
    static void serialize(ParallelConfig const& parallelConfig, std::ostream& os);
    static size_t serializedSize(ParallelConfig const& parallelConfig);

    static PeftCacheConfig deserializePeftCacheConfig(std::istream& is);
    static void serialize(PeftCacheConfig const& peftCacheConfig, std::ostream& os);
    static size_t serializedSize(PeftCacheConfig const& peftCacheConfig);

    static OrchestratorConfig deserializeOrchestratorConfig(std::istream& is);
    static void serialize(OrchestratorConfig const& orchestratorConfig, std::ostream& os);
    static size_t serializedSize(OrchestratorConfig const& orchestratorConfig);

    static DecodingMode deserializeDecodingMode(std::istream& is);
    static void serialize(DecodingMode const& decodingMode, std::ostream& os);
    static size_t serializedSize(DecodingMode const& decodingMode);

    static LookaheadDecodingConfig deserializeLookaheadDecodingConfig(std::istream& is);
    static void serialize(LookaheadDecodingConfig const& lookaheadDecodingConfig, std::ostream& os);
    static size_t serializedSize(LookaheadDecodingConfig const& lookaheadDecodingConfig);

    static EagleConfig deserializeEagleConfig(std::istream& is);
    static void serialize(EagleConfig const& eagleConfig, std::ostream& os);
    static size_t serializedSize(EagleConfig const& eagleConfig);

    static SpeculativeDecodingConfig deserializeSpeculativeDecodingConfig(std::istream& is);
    static void serialize(SpeculativeDecodingConfig const& specDecConfig, std::ostream& os);
    static size_t serializedSize(SpeculativeDecodingConfig const& specDecConfig);

    static GuidedDecodingConfig deserializeGuidedDecodingConfig(std::istream& is);
    static void serialize(GuidedDecodingConfig const& guidedDecodingConfig, std::ostream& os);
    static size_t serializedSize(GuidedDecodingConfig const& guidedDecodingConfig);

    static GuidedDecodingParams deserializeGuidedDecodingParams(std::istream& is);
    static void serialize(GuidedDecodingParams const& guidedDecodingParams, std::ostream& os);
    static size_t serializedSize(GuidedDecodingParams const& guidedDecodingParams);

    static KvCacheRetentionConfig deserializeKvCacheRetentionConfig(std::istream& is);
    static void serialize(KvCacheRetentionConfig const& kvCacheRetentionConfig, std::ostream& os);
    static size_t serializedSize(KvCacheRetentionConfig const& kvCacheRetentionConfig);

    static KvCacheRetentionConfig::TokenRangeRetentionConfig deserializeTokenRangeRetentionConfig(std::istream& is);
    static void serialize(
        KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig, std::ostream& os);
    static size_t serializedSize(KvCacheRetentionConfig::TokenRangeRetentionConfig const& tokenRangeRetentionConfig);

    static DecodingConfig deserializeDecodingConfig(std::istream& is);
    static void serialize(DecodingConfig const& decodingConfig, std::ostream& os);
    static size_t serializedSize(DecodingConfig const& decodingConfig);

    static DebugConfig deserializeDebugConfig(std::istream& is);
    static void serialize(DebugConfig const& debugConfig, std::ostream& os);
    static size_t serializedSize(DebugConfig const& debugConfig);

    static ExecutorConfig deserializeExecutorConfig(std::istream& is);
    static void serialize(ExecutorConfig const& executorConfig, std::ostream& os);
    static size_t serializedSize(ExecutorConfig const& executorConfig);

    static KvCacheStats deserializeKvCacheStats(std::istream& is);
    static void serialize(KvCacheStats const& kvCacheStats, std::ostream& os);
    static size_t serializedSize(KvCacheStats const& kvCacheStats);

    static StaticBatchingStats deserializeStaticBatchingStats(std::istream& is);
    static void serialize(StaticBatchingStats const& staticBatchingStats, std::ostream& os);
    static size_t serializedSize(StaticBatchingStats const& staticBatchingStats);

    static InflightBatchingStats deserializeInflightBatchingStats(std::istream& is);
    static void serialize(InflightBatchingStats const& inflightBatchingStats, std::ostream& os);
    static size_t serializedSize(InflightBatchingStats const& inflightBatchingStats);

    static IterationStats deserializeIterationStats(std::vector<char>& buffer);
    static IterationStats deserializeIterationStats(std::istream& is);
    static void serialize(IterationStats const& iterStats, std::ostream& os);
    static std::vector<char> serialize(IterationStats const& iterStats);
    static size_t serializedSize(IterationStats const& iterStats);
    static std::vector<char> serialize(std::vector<IterationStats> const& iterStatsVec);
    static std::vector<IterationStats> deserializeIterationStatsVec(std::vector<char>& buffer);

    [[nodiscard]] static DisServingRequestStats deserializeDisServingRequestStats(std::istream& is);
    static void serialize(DisServingRequestStats const& stats, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(DisServingRequestStats const& disServingRequestStats);

    [[nodiscard]] static RequestStage deserializeRequestStage(std::istream& is);
    static void serialize(RequestStage const& requestStage, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestStage const& requestStage);

    [[nodiscard]] static RequestStats deserializeRequestStats(std::istream& is);
    static void serialize(RequestStats const& state, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(RequestStats const& state);

    [[nodiscard]] static RequestStatsPerIteration deserializeRequestStatsPerIteration(std::istream& is);
    [[nodiscard]] static RequestStatsPerIteration deserializeRequestStatsPerIteration(std::vector<char>& buffer);
    static void serialize(RequestStatsPerIteration const& state, std::ostream& os);
    [[nodiscard]] static std::vector<char> serialize(RequestStatsPerIteration const& state);
    [[nodiscard]] static size_t serializedSize(RequestStatsPerIteration const& state);
    [[nodiscard]] static std::vector<char> serialize(std::vector<RequestStatsPerIteration> const& requestStatsVec);
    [[nodiscard]] static std::vector<RequestStatsPerIteration> deserializeRequestStatsPerIterationVec(
        std::vector<char>& buffer);

    static std::string deserializeString(std::istream& is);

    static bool deserializeBool(std::istream& is);

    static ModelType deserializeModelType(std::istream& is);
};

}
