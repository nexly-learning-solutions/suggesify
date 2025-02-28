
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace suggestify::runtime
{
class CudaStream;
}

namespace suggestify::executor
{

class Request;
class Tensor;

using TensorPtr = std::shared_ptr<Tensor>;
using SizeType32 = std::int32_t;
using FloatType = float;
using TokenIdType = std::int32_t;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using IdType = std::uint64_t;
using VecTokenExtraIds = std::vector<IdType>;
using IterationType = std::uint64_t;
using RandomSeedType = std::uint64_t;
using VecLogProbs = std::vector<FloatType>;
using StreamPtr = std::shared_ptr<suggestify::runtime::CudaStream>;
using MillisecondsType = std::chrono::milliseconds;
using LogitsPostProcessor
    = std::function<void(IdType, Tensor&, BeamTokens const&, StreamPtr const&, std::optional<IdType>)>;
using LogitsPostProcessorMap = std::unordered_map<std::string, LogitsPostProcessor>;
using LogitsPostProcessorBatched = std::function<void(std::vector<IdType> const&, std::vector<Tensor>&,
    std::vector<std::reference_wrapper<BeamTokens const>> const&, StreamPtr const&,
    std::vector<std::optional<IdType>> const&)>;
using MedusaChoices = std::vector<std::vector<SizeType32>>;
using EagleChoices = std::vector<std::vector<SizeType32>>;
using PriorityType = float;
using BufferView = std::basic_string_view<uint8_t>;

enum class DataType
{
    kBOOL,
    kUINT8,
    kINT8,
    kINT32,
    kINT64,
    kBF16,
    kFP8,
    kFP16,
    kFP32,
    kUNKNOWN
};

enum class RequestType
{
    REQUEST_TYPE_CONTEXT_AND_GENERATION = 0,
    REQUEST_TYPE_CONTEXT_ONLY = 1,
    REQUEST_TYPE_GENERATION_ONLY = 2
};

template <typename T, bool = false>
struct TypeTraits
{
};

template <>
struct TypeTraits<float>
{
    static constexpr auto value = DataType::kFP32;
};

template <>
struct TypeTraits<half>
{
    static constexpr auto value = DataType::kFP16;
};

template <>
struct TypeTraits<std::int8_t>
{
    static constexpr auto value = DataType::kINT8;
};

template <>
struct TypeTraits<std::int32_t>
{
    static constexpr auto value = DataType::kINT32;
};

template <>
struct TypeTraits<std::int64_t>
{
    static constexpr auto value = DataType::kINT64;
};

template <>
struct TypeTraits<bool>
{
    static constexpr auto value = DataType::kBOOL;
};

template <>
struct TypeTraits<std::uint8_t>
{
    static constexpr auto value = DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <>
struct TypeTraits<__nv_bfloat16>
{
    static constexpr auto value = DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TypeTraits<__nv_fp8_e4m3>
{
    static constexpr auto value = DataType::kFP8;
};
#endif

template <typename T>
struct TypeTraits<T*>
{
    static constexpr auto value = DataType::kINT64;
};

enum class MemoryType
{
    kCPU,
    kCPU_PINNED,
    kCPU_PINNEDPOOL,
    kGPU,
    kUVM,
    kUNKNOWN
};

enum class ModelType
{
    kDECODER_ONLY = 0,
    kENCODER_ONLY = 1,
    kENCODER_DECODER = 2,
};

enum class BatchingType
{
    kSTATIC = 0,

    kINFLIGHT = 1,
};

enum class CapacitySchedulerPolicy
{
    kMAX_UTILIZATION = 0,

    kGUARANTEED_NO_EVICT = 1,

    kSTATIC_BATCH = 2
};

std::ostream& operator<<(std::ostream& os, CapacitySchedulerPolicy policy);

enum class ContextChunkingPolicy
{
    kFIRST_COME_FIRST_SERVED = 0,

    kEQUAL_PROGRESS = 1,
};

std::ostream& operator<<(std::ostream& os, ContextChunkingPolicy policy);

enum class CommunicationType
{
    kMPI = 0
};

enum class CommunicationMode
{
    kLEADER,
    kORCHESTRATOR,
};

struct KvCacheStats
{
    SizeType32 maxNumBlocks;
    SizeType32 freeNumBlocks;
    SizeType32 usedNumBlocks;
    SizeType32 tokensPerBlock;
    SizeType32 allocTotalBlocks;
    SizeType32 allocNewBlocks;
    SizeType32 reusedBlocks;
    SizeType32 missedBlocks;
    float cacheHitRate;
};

struct StaticBatchingStats
{
    SizeType32 numScheduledRequests;
    SizeType32 numContextRequests;
    SizeType32 numCtxTokens;
    SizeType32 numGenTokens;
    SizeType32 emptyGenSlots;
};

struct InflightBatchingStats
{
    SizeType32 numScheduledRequests;
    SizeType32 numContextRequests;
    SizeType32 numGenRequests;
    SizeType32 numPausedRequests;
    SizeType32 numCtxTokens;
    SizeType32 microBatchId;
    float avgNumDecodedTokensPerIter;
};

struct IterationStats
{
    std::string timestamp;
    IterationType iter;
    double iterLatencyMS;
    double newActiveRequestsQueueLatencyMS;
    SizeType32 numNewActiveRequests;
    SizeType32 numActiveRequests;
    SizeType32 numQueuedRequests;
    SizeType32 numCompletedRequests;
    SizeType32 maxNumActiveRequests;
    SizeType32 maxBatchSizeStatic;
    SizeType32 maxBatchSizeTunerRecommended;
    SizeType32 maxBatchSizeRuntime;
    SizeType32 maxNumTokensStatic;
    SizeType32 maxNumTokensTunerRecommended;
    SizeType32 maxNumTokensRuntime;
    size_t gpuMemUsage;
    size_t cpuMemUsage;
    size_t pinnedMemUsage;
    std::optional<KvCacheStats> kvCacheStats;
    std::optional<KvCacheStats> crossKvCacheStats;
    std::optional<StaticBatchingStats> staticBatchingStats;
    std::optional<InflightBatchingStats> inflightBatchingStats;
};

enum class RequestStage
{
    kQUEUED,
    kENCODER_IN_PROGRESS,
    kCONTEXT_IN_PROGRESS,
    kGENERATION_IN_PROGRESS,
    kGENERATION_COMPLETE,
};

struct DisServingRequestStats
{
    double kvCacheTransferMS;
};

struct RequestStats
{
    IdType id;
    RequestStage stage;
    SizeType32 contextPrefillPosition;
    SizeType32 numGeneratedTokens;
    float avgNumDecodedTokensPerIter;
    bool scheduled;
    bool paused;
    std::optional<DisServingRequestStats> disServingStats;
    SizeType32 allocTotalBlocksPerRequest;
    SizeType32 allocNewBlocksPerRequest;
    SizeType32 reusedBlocksPerRequest;
    SizeType32 missedBlocksPerRequest;
    SizeType32 kvCacheHitRatePerRequest;
};

struct RequestStatsPerIteration
{
    IterationType iter;
    std::vector<RequestStats> requestStats;
};

struct RequestPerfMetrics
{
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

    struct TimingMetrics
    {
        TimePoint arrivalTime;
        TimePoint firstScheduledTime;
        TimePoint firstTokenTime;
        TimePoint lastTokenTime;
        TimePoint kvCacheTransferStart;
        TimePoint kvCacheTransferEnd;
    };

    struct KvCacheMetrics
    {
        SizeType32 numTotalAllocatedBlocks{0};
        SizeType32 numNewAllocatedBlocks{0};
        SizeType32 numReusedBlocks{0};
        SizeType32 numMissedBlocks{0};
        SizeType32 kvCacheHitRate{0};
    };

    TimingMetrics timingMetrics;
    KvCacheMetrics kvCacheMetrics;

    std::optional<IterationType> firstIter;
    std::optional<IterationType> lastIter;
    std::optional<IterationType> iter;
};

struct DebugTensorsPerIteration
{
    IterationType iter;
    std::map<std::string, Tensor> debugTensors;
};

enum class FinishReason
{
    kNOT_FINISHED = 0,

    kEND_ID = 1,

    kSTOP_WORDS = 2,

    kLENGTH = 3,

    kTIMED_OUT = 4,

    kCANCELLED = 5
};

class DecodingMode
{
public:
    static auto constexpr Auto()
    {
        return DecodingMode{kAuto};
    }

    static auto constexpr TopK()
    {
        return DecodingMode{kTopK | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr TopP()
    {
        return DecodingMode{kTopP | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr TopKTopP()
    {
        return DecodingMode{kTopKTopP | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr BeamSearch()
    {
        return DecodingMode{kBeamSearch | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr Medusa()
    {
        return DecodingMode{kMedusa | kUseMinLength | kStandardStopCriteria | kUseExplicitEosStop};
    }

    static auto constexpr Lookahead()
    {
        return DecodingMode{kLookahead | kUseMinLength | kStandardStopCriteria | kUseExplicitEosStop};
    }

    static auto constexpr ExplicitDraftTokens()
    {
        return DecodingMode{kExplicitDraftTokens | kStandardStopCriteria | kUseExplicitEosStop};
    }

    static auto constexpr ExternalDraftTokens()
    {
        return DecodingMode{kExternalDraftTokens | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr Eagle()
    {
        return DecodingMode{kEagle | kStandardStopCriteria | kUseExplicitEosStop};
    }

    auto constexpr useTemperature(bool useTemp)
    {
        mState = setBitTo(kUseTemperature, useTemp);
        return *this;
    }

    auto constexpr useOccurrencePenalties(bool usePenalty)
    {
        mState = setBitTo(kUseOccurrencePenalties, usePenalty);
        return *this;
    }

    auto constexpr usePresencePenalty(bool usePenalty)
    {
        mState = setBitTo(kUsePresencePenalties, usePenalty);
        return *this;
    }

    auto constexpr useRepetitionPenalty(bool usePenalty)
    {
        mState = setBitTo(kUseRepetitionPenalties, usePenalty);
        return *this;
    }

    auto constexpr useFrequencyPenalty(bool usePenalty)
    {
        mState = setBitTo(kUseFrequencyPenalties, usePenalty);
        return *this;
    }

    auto constexpr useMinLength(bool useMinLen)
    {
        mState = setBitTo(kUseMinLength, useMinLen);
        return *this;
    }

    auto constexpr useBanTokens(bool banTokens)
    {
        mState = setBitTo(kUseBanTokens, banTokens);
        return *this;
    }

    auto constexpr useBanWords(bool banWords)
    {
        mState = setBitTo(kUseBanWords, banWords);
        return *this;
    }

    auto constexpr useNoRepeatNgramSize(bool noRepeatNgramSize)
    {
        mState = setBitTo(kUseNoRepeatNgramSize, noRepeatNgramSize);
        return *this;
    }

    auto constexpr useStopWords(bool stopWords)
    {
        mState = setBitTo(kUseStopWords, stopWords);
        return *this;
    }

    auto constexpr useMaxLengthStop(bool maxLengthStop)
    {
        mState = setBitTo(kUseMaxLengthStop, maxLengthStop);
        return *this;
    }

    auto constexpr useExplicitEosStop(bool explicitEosStop)
    {
        mState = setBitTo(kUseExplicitEosStop, explicitEosStop);
        return *this;
    }

    [[nodiscard]] bool constexpr isAuto() const
    {
        return anyBitSet(kAuto);
    }

    [[nodiscard]] bool constexpr isTopK() const
    {
        return anyBitSet(kTopK);
    }

    [[nodiscard]] bool constexpr isTopP() const
    {
        return anyBitSet(kTopP);
    }

    [[nodiscard]] bool constexpr isTopKorTopP() const
    {
        return anyBitSet(kTopKTopP);
    }

    [[nodiscard]] bool constexpr isTopKandTopP() const
    {
        return allBitSet(kTopKTopP);
    }

    [[nodiscard]] bool constexpr isBeamSearch() const
    {
        return anyBitSet(kBeamSearch);
    }

    [[nodiscard]] bool constexpr isMedusa() const
    {
        return anyBitSet(kMedusa);
    }

    [[nodiscard]] bool constexpr isLookahead() const
    {
        return anyBitSet(kLookahead);
    }

    [[nodiscard]] bool constexpr isExplicitDraftTokens() const
    {
        return anyBitSet(kExplicitDraftTokens);
    }

    [[nodiscard]] bool constexpr isExternalDraftTokens() const
    {
        return anyBitSet(kExternalDraftTokens);
    }

    [[nodiscard]] bool constexpr isEagle() const
    {
        return anyBitSet(kEagle);
    }

    [[nodiscard]] bool constexpr isUseTemperature() const
    {
        return anyBitSet(kUseTemperature);
    }

    [[nodiscard]] bool constexpr isUsePresencePenalty() const
    {
        return anyBitSet(kUsePresencePenalties);
    }

    [[nodiscard]] bool constexpr isUseFrequencyPenalty() const
    {
        return anyBitSet(kUseFrequencyPenalties);
    }

    [[nodiscard]] bool constexpr isUseRepetitionPenalty() const
    {
        return anyBitSet(kUseRepetitionPenalties);
    }

    [[nodiscard]] bool constexpr isUseMinLength() const
    {
        return anyBitSet(kUseMinLength);
    }

    [[nodiscard]] bool constexpr isUseOccurrencePenalty() const
    {
        return anyBitSet(kUseOccurrencePenalties);
    }

    [[nodiscard]] bool constexpr isUsePenalty() const
    {
        return anyBitSet(kUsePenalties);
    }

    [[nodiscard]] bool constexpr isUseBanWords() const
    {
        return anyBitSet(kUseBanWords);
    }

    bool constexpr isUseNoRepeatNgramSize() const
    {
        return anyBitSet(kUseNoRepeatNgramSize);
    }

    bool constexpr isUseBanTokens() const
    {
        return anyBitSet(kUseBanTokens);
    }

    bool constexpr isUseStopWords() const
    {
        return anyBitSet(kUseStopWords);
    }

    bool constexpr isUseMaxLengthStop() const
    {
        return anyBitSet(kUseMaxLengthStop);
    }

    bool constexpr isUseExplicitEosStop() const
    {
        return anyBitSet(kUseExplicitEosStop);
    }

    bool constexpr isUseStopCriteria() const
    {
        return anyBitSet(kStandardStopCriteria | kUseExplicitEosStop);
    }

    using UnderlyingType = uint32_t;

    bool operator==(DecodingMode const& other) const
    {
        return mState == other.mState;
    }

    explicit constexpr DecodingMode(UnderlyingType state)
        : mState(state)
    {
    }

    [[nodiscard]] constexpr UnderlyingType getState() const
    {
        return mState;
    }

private:
    static UnderlyingType constexpr kUseRepetitionPenalties{1u << 0};
    static UnderlyingType constexpr kUseFrequencyPenalties{1u << 1};
    static UnderlyingType constexpr kUsePresencePenalties{1u << 2};
    static UnderlyingType constexpr kUseTemperature{1u << 3};
    static UnderlyingType constexpr kUseMinLength{1u << 4};
    static UnderlyingType constexpr kUseBanWords{1u << 5};
    static UnderlyingType constexpr kUseStopWords{1u << 6};
    static UnderlyingType constexpr kUseMaxLengthStop{1u << 7};
    static UnderlyingType constexpr kUseExplicitEosStop{1u << 8};
    static UnderlyingType constexpr kUseNoRepeatNgramSize{1u << 9};
    static UnderlyingType constexpr kStandardStopCriteria{kUseStopWords | kUseMaxLengthStop};
    static UnderlyingType constexpr kUseOccurrencePenalties{
        kUseRepetitionPenalties | kUseFrequencyPenalties | kUsePresencePenalties};
    static UnderlyingType constexpr kUsePenalties{kUseOccurrencePenalties | kUseTemperature | kUseMinLength};
    static UnderlyingType constexpr kUseBanTokens{kUseNoRepeatNgramSize | kUseBanWords};
    static SizeType32 constexpr kNumFlags{10};
    static UnderlyingType constexpr kAuto{1u << (kNumFlags + 0)};
    static UnderlyingType constexpr kTopK{1u << (kNumFlags + 1)};
    static UnderlyingType constexpr kTopP{1u << (kNumFlags + 2)};
    static UnderlyingType constexpr kBeamSearch{1u << (kNumFlags + 3)};
    static UnderlyingType constexpr kMedusa{1u << (kNumFlags + 4)};
    static UnderlyingType constexpr kLookahead{1u << (kNumFlags + 5)};
    static UnderlyingType constexpr kExplicitDraftTokens{1u << (kNumFlags + 6)};
    static UnderlyingType constexpr kExternalDraftTokens{1u << (kNumFlags + 7)};
    static UnderlyingType constexpr kEagle{1u << (kNumFlags + 8)};
    static UnderlyingType constexpr kTopKTopP{kTopK | kTopP};

    [[nodiscard]] bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    [[nodiscard]] bool constexpr allBitSet(UnderlyingType bits) const
    {
        return (mState & bits) == bits;
    }

    UnderlyingType constexpr setBitTo(UnderlyingType state, bool x)
    {
        return (mState & (~state)) | (state * static_cast<UnderlyingType>(x));
    }

    UnderlyingType mState{};
};

static_assert(DecodingMode::Auto().isAuto());
static_assert(!DecodingMode::Auto().isUseBanWords());
static_assert(!DecodingMode::Auto().isUseOccurrencePenalty());
static_assert(!DecodingMode::Auto().isUseStopCriteria());
static_assert(!DecodingMode::Auto().isTopK());
static_assert(!DecodingMode::Auto().isTopP());
static_assert(!DecodingMode::Auto().isBeamSearch());
static_assert(!DecodingMode::Auto().isMedusa());
static_assert(!DecodingMode::Auto().isLookahead());
static_assert(!DecodingMode::Auto().isExplicitDraftTokens());
static_assert(!DecodingMode::Auto().isExternalDraftTokens());
static_assert(!DecodingMode::Auto().isEagle());

static_assert(DecodingMode::TopK().isTopK());
static_assert(DecodingMode::TopK().isTopKorTopP());
static_assert(DecodingMode::TopK().isUseBanWords());
static_assert(DecodingMode::TopK().isUseOccurrencePenalty());
static_assert(DecodingMode::TopK().isUseStopCriteria());
static_assert(!DecodingMode::TopK().useRepetitionPenalty(false).isUseRepetitionPenalty());
static_assert(DecodingMode::TopK().useRepetitionPenalty(false).isUseOccurrencePenalty());
static_assert(!DecodingMode::TopK()
                   .useRepetitionPenalty(false)
                   .usePresencePenalty(false)
                   .useFrequencyPenalty(false)
                   .isUseOccurrencePenalty());
static_assert(!DecodingMode::TopK().isAuto());
static_assert(!DecodingMode::TopK().isTopKandTopP());
static_assert(!DecodingMode::TopK().isTopP());
static_assert(!DecodingMode::TopK().isBeamSearch());
static_assert(!DecodingMode::TopK().isMedusa());
static_assert(!DecodingMode::TopK().isLookahead());
static_assert(!DecodingMode::TopK().isExplicitDraftTokens());
static_assert(!DecodingMode::TopK().isExternalDraftTokens());
static_assert(!DecodingMode::TopK().isEagle());

static_assert(DecodingMode::TopP().isTopP());
static_assert(DecodingMode::TopP().isTopKorTopP());
static_assert(DecodingMode::TopP().isUseBanWords());
static_assert(DecodingMode::TopP().isUseOccurrencePenalty());
static_assert(DecodingMode::TopP().isUseStopCriteria());
static_assert(!DecodingMode::TopP().isAuto());
static_assert(!DecodingMode::TopP().isTopKandTopP());
static_assert(!DecodingMode::TopP().isTopK());
static_assert(!DecodingMode::TopP().isBeamSearch());
static_assert(!DecodingMode::TopP().isMedusa());
static_assert(!DecodingMode::TopP().isLookahead());
static_assert(!DecodingMode::TopP().isExplicitDraftTokens());
static_assert(!DecodingMode::TopP().isEagle());

static_assert(DecodingMode::TopKTopP().isTopK());
static_assert(DecodingMode::TopKTopP().isTopP());
static_assert(DecodingMode::TopKTopP().isTopKorTopP());
static_assert(DecodingMode::TopKTopP().isTopKandTopP());
static_assert(DecodingMode::TopKTopP().isUseBanWords());
static_assert(DecodingMode::TopKTopP().isUseOccurrencePenalty());
static_assert(DecodingMode::TopKTopP().isUseStopCriteria());
static_assert(!DecodingMode::TopKTopP().isAuto());
static_assert(!DecodingMode::TopKTopP().isBeamSearch());
static_assert(!DecodingMode::TopKTopP().isMedusa());
static_assert(!DecodingMode::TopKTopP().isLookahead());
static_assert(!DecodingMode::TopKTopP().isExplicitDraftTokens());
static_assert(!DecodingMode::TopKTopP().isExternalDraftTokens());
static_assert(!DecodingMode::TopKTopP().isEagle());

static_assert(DecodingMode::BeamSearch().isBeamSearch());
static_assert(DecodingMode::BeamSearch().isUseStopCriteria());
static_assert(!DecodingMode::BeamSearch().isAuto());
static_assert(!DecodingMode::BeamSearch().isTopKorTopP());
static_assert(!DecodingMode::BeamSearch().isMedusa());
static_assert(!DecodingMode::BeamSearch().isLookahead());
static_assert(!DecodingMode::BeamSearch().isExplicitDraftTokens());
static_assert(!DecodingMode::BeamSearch().isExternalDraftTokens());
static_assert(!DecodingMode::BeamSearch().isEagle());

static_assert(!DecodingMode::Medusa().isAuto());
static_assert(!DecodingMode::Medusa().isTopK());
static_assert(!DecodingMode::Medusa().isTopKorTopP());
static_assert(!DecodingMode::Medusa().isTopKandTopP());
static_assert(!DecodingMode::Medusa().isTopP());
static_assert(!DecodingMode::Medusa().isBeamSearch());
static_assert(!DecodingMode::Medusa().isLookahead());
static_assert(!DecodingMode::Medusa().isUseBanWords());
static_assert(!DecodingMode::Medusa().isUseOccurrencePenalty());
static_assert(!DecodingMode::Medusa().isExplicitDraftTokens());
static_assert(DecodingMode::Medusa().isUseStopCriteria());
static_assert(DecodingMode::Medusa().isUsePenalty());
static_assert(DecodingMode::Medusa().isUseMinLength());
static_assert(DecodingMode::Medusa().isMedusa());
static_assert(!DecodingMode::Medusa().isExternalDraftTokens());
static_assert(!DecodingMode::Medusa().isEagle());

static_assert(!DecodingMode::Lookahead().isAuto());
static_assert(!DecodingMode::Lookahead().isTopK());
static_assert(!DecodingMode::Lookahead().isTopKorTopP());
static_assert(!DecodingMode::Lookahead().isTopKandTopP());
static_assert(!DecodingMode::Lookahead().isTopP());
static_assert(!DecodingMode::Lookahead().isBeamSearch());
static_assert(!DecodingMode::Lookahead().isMedusa());
static_assert(!DecodingMode::Lookahead().isExplicitDraftTokens());
static_assert(DecodingMode::Lookahead().isUseStopCriteria());
static_assert(DecodingMode::Lookahead().isUseStopWords());
static_assert(DecodingMode::Lookahead().isUseExplicitEosStop());
static_assert(DecodingMode::Lookahead().isLookahead());
static_assert(!DecodingMode::Lookahead().isExternalDraftTokens());
static_assert(!DecodingMode::Lookahead().isEagle());

static_assert(!DecodingMode::ExplicitDraftTokens().isAuto());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopK());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopKorTopP());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopKandTopP());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopP());
static_assert(!DecodingMode::ExplicitDraftTokens().isBeamSearch());
static_assert(!DecodingMode::ExplicitDraftTokens().isMedusa());
static_assert(!DecodingMode::ExplicitDraftTokens().isLookahead());
static_assert(!DecodingMode::ExplicitDraftTokens().isUsePenalty());
static_assert(DecodingMode::ExplicitDraftTokens().isUseStopCriteria());
static_assert(!DecodingMode::ExplicitDraftTokens().isUseBanWords());
static_assert(DecodingMode::ExplicitDraftTokens().isExplicitDraftTokens());
static_assert(!DecodingMode::ExplicitDraftTokens().isExternalDraftTokens());
static_assert(!DecodingMode::ExplicitDraftTokens().isEagle());

static_assert(!DecodingMode::ExternalDraftTokens().isTopK());
static_assert(!DecodingMode::ExternalDraftTokens().isTopP());
static_assert(!DecodingMode::ExternalDraftTokens().isTopKorTopP());
static_assert(!DecodingMode::ExternalDraftTokens().isTopKandTopP());
static_assert(DecodingMode::ExternalDraftTokens().isUseBanWords());
static_assert(DecodingMode::ExternalDraftTokens().isUseOccurrencePenalty());
static_assert(DecodingMode::ExternalDraftTokens().isUseStopCriteria());
static_assert(!DecodingMode::ExternalDraftTokens().isAuto());
static_assert(!DecodingMode::ExternalDraftTokens().isBeamSearch());
static_assert(!DecodingMode::ExternalDraftTokens().isMedusa());
static_assert(!DecodingMode::ExternalDraftTokens().isLookahead());
static_assert(!DecodingMode::ExternalDraftTokens().isExplicitDraftTokens());
static_assert(!DecodingMode::ExternalDraftTokens().isEagle());
static_assert(DecodingMode::ExternalDraftTokens().isExternalDraftTokens());

static_assert(!DecodingMode::Eagle().isTopK());
static_assert(!DecodingMode::Eagle().isTopP());
static_assert(!DecodingMode::Eagle().isTopKorTopP());
static_assert(!DecodingMode::Eagle().isTopKandTopP());
static_assert(!DecodingMode::Eagle().isUseBanWords());
static_assert(!DecodingMode::Eagle().isUseOccurrencePenalty());
static_assert(DecodingMode::Eagle().isUseStopCriteria());
static_assert(!DecodingMode::Eagle().isAuto());
static_assert(!DecodingMode::Eagle().isBeamSearch());
static_assert(!DecodingMode::Eagle().isMedusa());
static_assert(!DecodingMode::Eagle().isLookahead());
static_assert(!DecodingMode::Eagle().isExplicitDraftTokens());
static_assert(!DecodingMode::Eagle().isExternalDraftTokens());
static_assert(DecodingMode::Eagle().isEagle());
}
