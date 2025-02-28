
#pragma once

#include "../executor/tensor.h"
#include "../executor/types.h"
#include "../runtime/common.h"
#include "../runtime/runtimeDefaults.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace suggestify::mpi
{
class MpiComm;
}

namespace suggestify::batch_manager::kv_cache_manager
{
class BaseKVCacheManager;
}

namespace suggestify::executor
{

char const* version() noexcept;

class Model;
class Serialization;
class DataTransceiverState;

class SamplingConfig
{
public:
    explicit SamplingConfig(SizeType32 beamWidth = 1, std::optional<SizeType32> const& topK = std::nullopt,
        std::optional<FloatType> const& topP = std::nullopt, std::optional<FloatType> const& topPMin = std::nullopt,
        std::optional<TokenIdType> const& topPResetIds = std::nullopt,
        std::optional<FloatType> const& topPDecay = std::nullopt,
        std::optional<RandomSeedType> const& seed = std::nullopt,
        std::optional<FloatType> const& temperature = std::nullopt,
        std::optional<SizeType32> const& minTokens = std::nullopt,
        std::optional<FloatType> const& beamSearchDiversityRate = std::nullopt,
        std::optional<FloatType> const& repetitionPenalty = std::nullopt,
        std::optional<FloatType> const& presencePenalty = std::nullopt,
        std::optional<FloatType> const& frequencyPenalty = std::nullopt,
        std::optional<FloatType> const& lengthPenalty = std::nullopt,
        std::optional<SizeType32> const& earlyStopping = std::nullopt,
        std::optional<SizeType32> const& noRepeatNgramSize = std::nullopt,
        std::optional<SizeType32> const& numReturnSequences = std::nullopt);

    bool operator==(SamplingConfig const& other) const;

    [[nodiscard]] SizeType32 getBeamWidth() const;
    [[nodiscard]] SizeType32 getNumReturnBeams() const;
    [[nodiscard]] std::optional<SizeType32> getTopK() const;
    [[nodiscard]] std::optional<FloatType> getTopP() const;
    [[nodiscard]] std::optional<FloatType> getTopPMin() const;
    [[nodiscard]] std::optional<SizeType32> getTopPResetIds() const;
    [[nodiscard]] std::optional<FloatType> getTopPDecay() const;
    [[nodiscard]] std::optional<RandomSeedType> getSeed() const;
    [[nodiscard]] std::optional<RandomSeedType> getRandomSeed() const;
    [[nodiscard]] std::optional<FloatType> getTemperature() const;
    [[nodiscard]] std::optional<SizeType32> getMinTokens() const;
    [[nodiscard]] std::optional<SizeType32> getMinLength() const;
    [[nodiscard]] std::optional<FloatType> getBeamSearchDiversityRate() const;
    [[nodiscard]] std::optional<FloatType> getRepetitionPenalty() const;
    [[nodiscard]] std::optional<FloatType> getPresencePenalty() const;
    [[nodiscard]] std::optional<FloatType> getFrequencyPenalty() const;
    [[nodiscard]] std::optional<FloatType> getLengthPenalty() const;
    [[nodiscard]] std::optional<SizeType32> getEarlyStopping() const;
    [[nodiscard]] std::optional<SizeType32> getNoRepeatNgramSize() const;
    [[nodiscard]] std::optional<SizeType32> getNumReturnSequences() const;

    void setBeamWidth(SizeType32 beamWidth);
    void setTopK(std::optional<SizeType32> const& topK);
    void setTopP(std::optional<FloatType> const& topP);
    void setTopPMin(std::optional<FloatType> const& topPMin);
    void setTopPResetIds(std::optional<TokenIdType> const& topPResetIds);
    void setTopPDecay(std::optional<FloatType> const& topPDecay);
    void setSeed(std::optional<RandomSeedType> const& seed);
    void setRandomSeed(std::optional<RandomSeedType> const& randomSeed);
    void setTemperature(std::optional<FloatType> const& temperature);
    void setMinTokens(std::optional<SizeType32> const& minTokens);
    void setMinLength(std::optional<SizeType32> const& minLength);
    void setBeamSearchDiversityRate(std::optional<FloatType> const& beamSearchDiversityRate);
    void setRepetitionPenalty(std::optional<FloatType> const& repetitionPenalty);
    void setPresencePenalty(std::optional<FloatType> const& presencePenalty);
    void setFrequencyPenalty(std::optional<FloatType> const& frequencyPenalty);
    void setLengthPenalty(std::optional<FloatType> const& lengthPenalty);
    void setEarlyStopping(std::optional<SizeType32> const& earlyStopping);
    void setNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize);
    void setNumReturnSequences(std::optional<SizeType32> const& numReturnSequences);

private:
    static SizeType32 checkBeamWidth(SizeType32 beamWidth);
    static std::optional<FloatType> const& checkTopK(std::optional<FloatType> const& topK);
    static std::optional<FloatType> const& checkTopP(std::optional<FloatType> const& topP);
    static std::optional<FloatType> const& checkTopPMin(std::optional<FloatType> const& topPMin);
    static std::optional<TokenIdType> const& checkTopPResetIds(std::optional<TokenIdType> const& topPResetIds);
    static std::optional<FloatType> const& checkTopPDecay(std::optional<FloatType> const& topPDecay);
    static std::optional<FloatType> const& checkTemperature(std::optional<FloatType> const& temperature);
    static std::optional<FloatType> const& checkRepetitionPenalty(std::optional<FloatType> const& penalty);
    static std::optional<SizeType32> const& checkMinTokens(std::optional<SizeType32> const& minTokens);
    static std::optional<SizeType32> const& checkNoRepeatNgramSize(std::optional<SizeType32> const& noRepeatNgramSize);
    static std::optional<FloatType> const& checkBeamSearchDiversityRate(
        std::optional<FloatType> const& beamSearchDiversityRate);
    static std::optional<SizeType32> const& checkNumReturnSequences(
        std::optional<SizeType32> const& numReturnSequences, SizeType32 beamWidth);

    void updateNumReturnBeams();

    friend class Serialization;

    SizeType32 mBeamWidth;
    std::optional<SizeType32> mTopK;
    std::optional<FloatType> mTopP;
    std::optional<FloatType> mTopPMin;
    std::optional<TokenIdType> mTopPResetIds;
    std::optional<FloatType> mTopPDecay;
    std::optional<RandomSeedType> mSeed;
    std::optional<FloatType> mTemperature;
    std::optional<SizeType32> mMinTokens;
    std::optional<FloatType> mBeamSearchDiversityRate;
    std::optional<FloatType> mRepetitionPenalty;
    std::optional<FloatType> mPresencePenalty;
    std::optional<FloatType> mFrequencyPenalty;
    std::optional<FloatType> mLengthPenalty;
    std::optional<SizeType32> mEarlyStopping;
    std::optional<SizeType32> mNoRepeatNgramSize;
    std::optional<SizeType32> mNumReturnSequences;
    SizeType32 mNumReturnBeams;
};

class OutputConfig
{
public:
    class AdditionalModelOutput
    {
    public:
        explicit AdditionalModelOutput(std::string name, bool gatherContext = false);

        std::string name;
        bool gatherContext{false};
    };

    explicit OutputConfig(bool returnLogProbs = false, bool returnContextLogits = false,
        bool returnGenerationLogits = false, bool excludeInputFromOutput = false, bool returnEncoderOutput = false,
        bool returnPerfMetrics = false,
        std::optional<std::vector<AdditionalModelOutput>> additionalModelOutputs = std::nullopt);

    bool returnLogProbs;
    bool returnContextLogits;
    bool returnGenerationLogits;
    bool excludeInputFromOutput;
    bool returnEncoderOutput;
    bool returnPerfMetrics;

    std::optional<std::vector<AdditionalModelOutput>> additionalModelOutputs;
};

class ExternalDraftTokensConfig
{
public:
    explicit ExternalDraftTokensConfig(VecTokens tokens, std::optional<Tensor> logits = std::nullopt,
        std::optional<FloatType> const& acceptanceThreshold = std::nullopt,
        std::optional<bool> const& fastLogits = std::nullopt);

    [[nodiscard]] VecTokens getTokens() const;
    [[nodiscard]] std::optional<Tensor> getLogits() const;
    [[nodiscard]] std::optional<FloatType> getAcceptanceThreshold() const;
    [[nodiscard]] std::optional<bool> getFastLogits() const;

private:
    friend class Serialization;
    VecTokens mTokens;
    std::optional<Tensor> mLogits;
    std::optional<FloatType> mAcceptanceThreshold;
    std::optional<bool> mFastLogits;
};

class PromptTuningConfig
{
public:
    explicit PromptTuningConfig(
        Tensor embeddingTable, std::optional<VecTokenExtraIds> inputTokenExtraIds = std::nullopt);

    [[nodiscard]] Tensor getEmbeddingTable() const;

    [[nodiscard]] std::optional<VecTokenExtraIds> getInputTokenExtraIds() const;

private:
    friend class Serialization;
    Tensor mEmbeddingTable;

    std::optional<VecTokenExtraIds> mInputTokenExtraIds;
};

class MropeConfig
{
public:
    explicit MropeConfig(Tensor mropeRoratySinCos, SizeType32 mropePositionDeltas);

    [[nodiscard]] Tensor getMRopeRotaryCosSin() const;
    [[nodiscard]] SizeType32 getMRopePositionDeltas() const;

private:
    friend class Serialization;
    Tensor mMRopeRotaryCosSin;
    SizeType32 mMRopePositionDeltas;
};

class LoraConfig
{
public:
    explicit LoraConfig(
        IdType taskId, std::optional<Tensor> weights = std::nullopt, std::optional<Tensor> config = std::nullopt);

    [[nodiscard]] IdType getTaskId() const;
    [[nodiscard]] std::optional<Tensor> getWeights() const;
    [[nodiscard]] std::optional<Tensor> getConfig() const;

private:
    friend class Serialization;

    IdType mTaskId;
    std::optional<Tensor> mWeights;
    std::optional<Tensor> mConfig;
};

struct LookaheadDecodingConfig
{
    LookaheadDecodingConfig(SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize);

    explicit LookaheadDecodingConfig()
        : LookaheadDecodingConfig(
            kDefaultLookaheadDecodingWindow, kDefaultLookaheadDecodingNgram, kDefaultLookaheadDecodingVerificationSet)
    {
    }

    bool operator==(LookaheadDecodingConfig const& other) const;
    [[nodiscard]] std::tuple<SizeType32 const, SizeType32 const, SizeType32 const> get() const;
    [[nodiscard]] SizeType32 getWindowSize() const;
    [[nodiscard]] SizeType32 getNgramSize() const;
    [[nodiscard]] SizeType32 getVerificationSetSize() const;

    [[nodiscard]] std::tuple<SizeType32, SizeType32, SizeType32, SizeType32> calculateSpeculativeResource() const;

    [[nodiscard]] bool isLE(LookaheadDecodingConfig const& that) const;

    static bool isLegal(SizeType32 windowSize, SizeType32 ngramSize, SizeType32 verificationSetSize) noexcept;

private:
    friend class Serialization;

    static constexpr SizeType32 kDefaultLookaheadDecodingWindow = 4;
    static constexpr SizeType32 kDefaultLookaheadDecodingNgram = 3;
    static constexpr SizeType32 kDefaultLookaheadDecodingVerificationSet = 4;

    SizeType32 mWindowSize;
    SizeType32 mNgramSize;
    SizeType32 mVerificationSetSize;
};

struct EagleConfig
{
    explicit EagleConfig(std::optional<EagleChoices> eagleChoices = std::nullopt, bool greedySampling = true,
        std::optional<float> posteriorThreshold = std::nullopt);

    bool operator==(EagleConfig const& other) const;
    [[nodiscard]] std::optional<EagleChoices> getEagleChoices() const;
    [[nodiscard]] std::optional<float> getPosteriorThreshold() const;
    [[nodiscard]] bool isGreedySampling() const;

private:
    std::optional<float> const& checkPosteriorValue(std::optional<float> const& value);

private:
    friend class Serialization;

    std::optional<EagleChoices> mEagleChoices;

    bool mGreedySampling;
    std::optional<float> mPosteriorThreshold;
};

class ContextPhaseParams
{
public:
    using RequestIdType = std::uint64_t;

    explicit ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId);
    ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId, void* state);

    ContextPhaseParams(ContextPhaseParams const&);
    ContextPhaseParams(ContextPhaseParams&&) noexcept;
    ContextPhaseParams& operator=(ContextPhaseParams const&);
    ContextPhaseParams& operator=(ContextPhaseParams&&) noexcept;
    ~ContextPhaseParams();

    [[nodiscard]] bool operator==(ContextPhaseParams const&) const noexcept;

    [[nodiscard]] VecTokens const& getFirstGenTokens() const& noexcept;
    [[nodiscard]] VecTokens popFirstGenTokens() && noexcept;
    [[nodiscard]] RequestIdType getReqId() const noexcept;

    [[nodiscard]] void const* getState() const noexcept;
    [[nodiscard]] void* getState() noexcept;
    [[nodiscard]] void* releaseState() noexcept;

private:
    friend class Serialization;
    static void deleter(void const* data);
    using StatePtr = std::unique_ptr<void, decltype(&deleter)>;

    RequestIdType mReqId{0};

    VecTokens mFirstGenTokens;

    StatePtr mState{nullptr, deleter};
};

class SpeculativeDecodingConfig
{
public:
    explicit SpeculativeDecodingConfig(bool fastLogits = false);

    bool operator==(SpeculativeDecodingConfig const& other) const;

    bool fastLogits;
};

class GuidedDecodingParams
{
public:
    enum class GuideType
    {
        kJSON = 0,

        kJSON_SCHEMA = 1,

        kREGEX = 2,

        kEBNF_GRAMMAR = 3,
    };

    explicit GuidedDecodingParams(GuideType guideType, std::optional<std::string> guide = std::nullopt);

    bool operator==(GuidedDecodingParams const& other) const;
    [[nodiscard]] GuideType getGuideType() const;
    [[nodiscard]] std::optional<std::string> getGuide() const;

private:
    friend class Serialization;

    GuideType mGuideType;
    std::optional<std::string> mGuide;
};

using RetentionPriority = SizeType32;

struct RetentionPriorityAndDuration
{

    RetentionPriorityAndDuration(std::optional<RetentionPriority> const& retentionPriority,
        std::optional<std::chrono::milliseconds> const& durationMs)
        : retentionPriority{retentionPriority}
        , durationMs{durationMs}
    {
    }

    std::optional<RetentionPriority> retentionPriority;
    std::optional<std::chrono::milliseconds> durationMs;
};

class KvCacheRetentionConfig
{

public:
    static constexpr RetentionPriority kMinRetentionPriority = 0;
    static constexpr RetentionPriority kMaxRetentionPriority = 100;
    static constexpr RetentionPriority kDefaultRetentionPriority = 35;

    struct TokenRangeRetentionConfig
    {
    public:
        explicit TokenRangeRetentionConfig(SizeType32 tokenStart, std::optional<SizeType32> tokenEnd = std::nullopt,
            RetentionPriority priority = KvCacheRetentionConfig::kDefaultRetentionPriority,
            std::optional<std::chrono::milliseconds> durationMs = std::nullopt)
            : tokenStart{tokenStart}
            , tokenEnd{tokenEnd}
            , priority{priority}
            , durationMs{durationMs}
        {
            CHECK_WITH_INFO(priority >= KvCacheRetentionConfig::kMinRetentionPriority
                    && priority <= KvCacheRetentionConfig::kMaxRetentionPriority,
                "Invalid priority value. Must be between %d and %d", KvCacheRetentionConfig::kMinRetentionPriority,
                KvCacheRetentionConfig::kMaxRetentionPriority);
        };

        SizeType32 tokenStart;
        std::optional<SizeType32> tokenEnd;
        RetentionPriority priority;
        std::optional<std::chrono::milliseconds> durationMs;

        bool operator==(TokenRangeRetentionConfig const& other) const
        {
            return tokenStart == other.tokenStart && tokenEnd == other.tokenEnd && priority == other.priority
                && durationMs == other.durationMs;
        }
    };

    explicit KvCacheRetentionConfig()
        : KvCacheRetentionConfig({}, kDefaultRetentionPriority)
    {
    }

    explicit KvCacheRetentionConfig(std::vector<TokenRangeRetentionConfig> const& tokenRangeRetentionPriorities,
        RetentionPriority decodeRetentionPriority = kDefaultRetentionPriority,
        std::optional<std::chrono::milliseconds> decodeDurationMs = std::nullopt);

    [[nodiscard]] std::vector<TokenRangeRetentionConfig> getTokenRangeRetentionConfigs() const;
    [[nodiscard]] RetentionPriority getDecodeRetentionPriority() const;
    [[nodiscard]] std::optional<std::chrono::milliseconds> getDecodeDurationMs() const;

    [[nodiscard]] std::vector<RetentionPriorityAndDuration> getPerBlockRetentionPriorityDuration(
        SizeType32 blockSize, SizeType32 seqLen) const;

private:
    std::vector<TokenRangeRetentionConfig> mTokenRangeRetentionConfigs;
    RetentionPriority mDecodeRetentionPriority;
    std::optional<std::chrono::milliseconds> mDecodeDurationMs;
};

class Request
{
public:
    static constexpr PriorityType kDefaultPriority = 0.5;


    Request(VecTokens inputTokenIds, SizeType32 maxTokens, bool streaming = false,
        SamplingConfig const& samplingConfig = SamplingConfig(), OutputConfig const& outputConfig = OutputConfig(),
        std::optional<SizeType32> const& endId = std::nullopt, std::optional<SizeType32> const& padId = std::nullopt,
        std::optional<std::vector<SizeType32>> positionIds = std::nullopt,
        std::optional<std::list<VecTokens>> badWords = std::nullopt,
        std::optional<std::list<VecTokens>> stopWords = std::nullopt,
        std::optional<Tensor> embeddingBias = std::nullopt,
        std::optional<ExternalDraftTokensConfig> externalDraftTokensConfig = std::nullopt,
        std::optional<PromptTuningConfig> pTuningConfig = std::nullopt,
        std::optional<MropeConfig> mRopeConfig = std::nullopt, std::optional<LoraConfig> loraConfig = std::nullopt,
        std::optional<LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        std::optional<std::string> logitsPostProcessorName = std::nullopt,
        std::optional<VecTokens> encoderInputTokenIds = std::nullopt, std::optional<IdType> clientId = std::nullopt,
        bool returnAllGeneratedTokens = false, PriorityType priority = kDefaultPriority,
        RequestType type = RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<ContextPhaseParams> contextPhaseParams = std::nullopt,
        std::optional<Tensor> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<Tensor> crossAttentionMask = std::nullopt, SizeType32 numReturnSequences = 1,
        std::optional<EagleConfig> eagleConfig = std::nullopt, std::optional<Tensor> skipCrossAttnBlocks = std::nullopt,
        std::optional<GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt);

    static auto constexpr kBatchedPostProcessorName = "batched";

    Request(Request const& other);
    Request(Request&& other) noexcept;
    Request& operator=(Request const& other);
    Request& operator=(Request&& other) noexcept;
    ~Request();

    [[nodiscard]] VecTokens getInputTokenIds() const;
    [[nodiscard]] SizeType32 getMaxTokens() const;
    [[nodiscard]] SizeType32 getMaxNewTokens() const;
    [[nodiscard]] bool getStreaming() const;
    [[nodiscard]] SamplingConfig getSamplingConfig() const;
    [[nodiscard]] OutputConfig getOutputConfig() const;
    [[nodiscard]] std::optional<SizeType32> getEndId() const;
    [[nodiscard]] std::optional<SizeType32> getPadId() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getPositionIds() const;
    [[nodiscard]] std::optional<std::list<VecTokens>> getBadWords() const;
    [[nodiscard]] std::optional<std::list<VecTokens>> getStopWords() const;
    [[nodiscard]] std::optional<Tensor> getEmbeddingBias() const;
    [[nodiscard]] std::optional<ExternalDraftTokensConfig> getExternalDraftTokensConfig() const;
    [[nodiscard]] std::optional<PromptTuningConfig> getPromptTuningConfig() const;
    [[nodiscard]] std::optional<MropeConfig> getMropeConfig() const;
    [[nodiscard]] std::optional<LoraConfig> getLoraConfig() const;
    [[nodiscard]] std::optional<LookaheadDecodingConfig> getLookaheadConfig() const;
    [[nodiscard]] std::optional<KvCacheRetentionConfig> getKvCacheRetentionConfig() const;
    [[nodiscard]] std::optional<std::string> getLogitsPostProcessorName() const;
    [[nodiscard]] std::optional<VecTokens> getEncoderInputTokenIds() const;
    [[nodiscard]] std::optional<IdType> getClientId() const;
    [[nodiscard]] PriorityType getPriority() const;
    [[nodiscard]] bool getReturnAllGeneratedTokens() const;
    [[nodiscard]] std::optional<ContextPhaseParams> const& getContextPhaseParams() const;
    [[nodiscard]] std::optional<Tensor> getEncoderInputFeatures() const;
    [[nodiscard]] std::optional<SizeType32> getEncoderOutputLength() const;
    [[nodiscard]] std::optional<Tensor> getCrossAttentionMask() const;
    [[nodiscard]] RequestType getRequestType() const;
    [[nodiscard]] SizeType32 getNumReturnSequences() const;
    [[nodiscard]] std::optional<EagleConfig> getEagleConfig() const;
    [[nodiscard]] std::optional<Tensor> getSkipCrossAttnBlocks() const;
    [[nodiscard]] std::optional<GuidedDecodingParams> getGuidedDecodingParams() const;
    [[nodiscard]] std::optional<MillisecondsType> getAllottedTimeMs() const;
    [[nodiscard]] std::optional<std::vector<std::string>> getAdditionalOutputNames() const;

    void setStreaming(bool streaming);
    void setSamplingConfig(SamplingConfig const& config);
    void setOutputConfig(OutputConfig const& outputConfig);
    void setEndId(SizeType32 endId);
    void setPadId(SizeType32 padId);
    void setPositionIds(std::vector<SizeType32> const& positionIds);
    void setBadWords(std::list<VecTokens> const& badWords);
    void setStopWords(std::list<VecTokens> const& stopWords);
    void setEmbeddingBias(Tensor const& embeddingBias);
    void setExternalDraftTokensConfig(ExternalDraftTokensConfig const& externalDraftTokensConfig);
    void setPromptTuningConfig(PromptTuningConfig const& pTuningConfig);
    void setMropeConfig(MropeConfig const& mRopeConfig);
    void setLoraConfig(LoraConfig const& loraConfig);
    void setLookaheadConfig(LookaheadDecodingConfig const& lookaheadConfig);
    void setKvCacheRetentionConfig(KvCacheRetentionConfig const& kvCacheRetentionConfig);
    void setLogitsPostProcessorName(std::string const& logitsPostProcessorName);
    void setEncoderInputTokenIds(VecTokens const& encoderInputTokenIds);
    void setClientId(IdType clientId);
    void setPriority(PriorityType priority);
    void setReturnAllGeneratedTokens(bool returnAllGeneratedTokens);
    void setRequestType(RequestType const& requestType);
    void setContextPhaseParams(ContextPhaseParams contextPhaseParams);
    void setEncoderInputFeatures(Tensor encoderInputFeatures);
    void setEncoderOutputLength(SizeType32 encoderOutputLength);
    void setCrossAttentionMask(Tensor crossAttentionMask);
    void setNumReturnSequences(SizeType32 numReturnSequences);
    void setEagleConfig(std::optional<EagleConfig> const& eagleConfig);
    void setSkipCrossAttnBlocks(Tensor skipCrossAttnBlocks);
    void setGuidedDecodingParams(GuidedDecodingParams const& guidedDecodingParams);
    void setAllottedTimeMs(MillisecondsType allottedTimeMs);
    void setAdditionalOutputNames(std::optional<std::vector<std::string>> additionalOutputNames);

private:
    friend class Serialization;
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

struct SpeculativeDecodingFastLogitsInfo
{
    uint64_t draftRequestId;

    int32_t draftParticipantId;

    [[nodiscard]] Tensor toTensor() const;
};

struct AdditionalOutput
{
    AdditionalOutput(std::string name, Tensor output)
        : name(std::move(name))
        , output(std::move(output))
    {
    }

    std::string name;
    Tensor output;
};

struct Result
{
    bool isFinal;

    BeamTokens outputTokenIds;

    std::optional<VecLogProbs> cumLogProbs;

    std::optional<std::vector<VecLogProbs>> logProbs;

    std::optional<Tensor> contextLogits;

    std::optional<Tensor> generationLogits;

    std::optional<SpeculativeDecodingFastLogitsInfo> specDecFastLogitsInfo;

    std::optional<Tensor> encoderOutput;

    std::vector<FinishReason> finishReasons;

    std::optional<ContextPhaseParams> contextPhaseParams;

    SizeType32 decodingIter{0};

    SizeType32 sequenceIndex{0};

    bool isSequenceFinal;

    std::optional<RequestPerfMetrics> requestPerfMetrics;

    std::vector<AdditionalOutput> additionalOutputs;
};

class Response
{
public:
    Response(IdType requestId, std::string errorMsg, std::optional<IdType> clientId = std::nullopt);
    Response(IdType requestId, Result Result, std::optional<IdType> clientId = std::nullopt);

    ~Response();
    Response(Response const& other);
    Response(Response&& other) noexcept;
    Response& operator=(Response const& other);
    Response& operator=(Response&& other) noexcept;

    [[nodiscard]] IdType getRequestId() const;

    [[nodiscard]] std::optional<IdType> getClientId() const;

    [[nodiscard]] bool hasError() const;

    [[nodiscard]] std::string const& getErrorMsg() const;

    [[nodiscard]] Result const& getResult() const;

private:
    friend class Serialization;
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

class DynamicBatchConfig
{
public:
    static SizeType32 const kDefaultDynamicBatchMovingAverageWindow = 128;

    explicit DynamicBatchConfig(bool enableBatchSizeTuning = false, bool enableMaxNumTokensTuning = false,
        SizeType32 dynamicBatchMovingAverageWindow = kDefaultDynamicBatchMovingAverageWindow,
        std::vector<std::pair<SizeType32, SizeType32>> batchSizeTable = kDefaultBatchSizeTable);

    [[nodiscard]] SizeType32 getDynamicBatchMovingAverageWindow() const;

    [[nodiscard]] bool getEnableBatchSizeTuning() const;

    [[nodiscard]] bool getEnableMaxNumTokensTuning() const;

    [[nodiscard]] std::vector<std::pair<SizeType32, SizeType32>> getBatchSizeTable() const;

    static std::vector<std::pair<SizeType32, SizeType32>> const kDefaultBatchSizeTable;

private:
    friend class Serialization;

    bool mEnableBatchSizeTuning;

    bool mEnableMaxNumTokensTuning;

    SizeType32 mDynamicBatchMovingAverageWindow;

    std::vector<std::pair<SizeType32, SizeType32>> mBatchSizeTable;
};

class SchedulerConfig
{
public:
    explicit SchedulerConfig(
        CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        std::optional<ContextChunkingPolicy> contextChunkingPolicy = std::nullopt,
        std::optional<DynamicBatchConfig> dynamicBatchConfig = std::nullopt);

    bool operator==(SchedulerConfig const& other) const;

    [[nodiscard]] CapacitySchedulerPolicy getCapacitySchedulerPolicy() const;

    [[nodiscard]] std::optional<ContextChunkingPolicy> getContextChunkingPolicy() const;

    [[nodiscard]] std::optional<DynamicBatchConfig> getDynamicBatchConfig() const;

private:
    friend class Serialization;

    CapacitySchedulerPolicy mCapacitySchedulerPolicy;

    std::optional<ContextChunkingPolicy> mContextChunkingPolicy;

    std::optional<DynamicBatchConfig> mDynamicBatchConfig;
};

class KvCacheConfig
{
public:
    explicit KvCacheConfig(bool enableBlockReuse = false, std::optional<SizeType32> const& maxTokens = std::nullopt,
        std::optional<std::vector<SizeType32>> const& maxAttentionWindowVec = std::nullopt,
        std::optional<SizeType32> const& sinkTokenLength = std::nullopt,
        std::optional<FloatType> const& freeGpuMemoryFraction = std::nullopt,
        std::optional<size_t> const& hostCacheSize = std::nullopt, bool onboardBlocks = true,
        std::optional<FloatType> const& crossKvCacheFraction = std::nullopt,
        std::optional<RetentionPriority> secondaryOffloadMinPriority = std::nullopt, size_t eventBufferMaxSize = 0,
        std::optional<suggestify::runtime::RuntimeDefaults> const& runtimeDefaults = std::nullopt);

    [[nodiscard]] bool getEnableBlockReuse() const;
    [[nodiscard]] std::optional<SizeType32> getMaxTokens() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getMaxAttentionWindowVec() const;
    [[nodiscard]] std::optional<SizeType32> getSinkTokenLength() const;
    [[nodiscard]] std::optional<FloatType> getFreeGpuMemoryFraction() const;
    [[nodiscard]] std::optional<FloatType> getCrossKvCacheFraction() const;
    [[nodiscard]] std::optional<size_t> getHostCacheSize() const;
    [[nodiscard]] bool getOnboardBlocks() const;
    [[nodiscard]] std::optional<RetentionPriority> getSecondaryOffloadMinPriority() const;
    [[nodiscard]] size_t getEventBufferMaxSize() const;

    void setEnableBlockReuse(bool enableBlockReuse);
    void setMaxTokens(SizeType32 maxTokens);
    void setMaxAttentionWindowVec(std::vector<SizeType32> maxAttentionWindowVec);
    void setSinkTokenLength(SizeType32 sinkTokenLength);
    void setFreeGpuMemoryFraction(FloatType freeGpuMemoryFraction);
    void setCrossKvCacheFraction(FloatType crossKvCacheFraction);
    void setHostCacheSize(size_t hostCacheSize);
    void setOnboardBlocks(bool onboardBlocks);
    void setSecondaryOffloadMinPriority(std::optional<RetentionPriority> secondaryOffloadMinPriority);
    void setEventBufferMaxSize(size_t eventBufferMaxSize);
    void fillEmptyFieldsFromRuntimeDefaults(suggestify::runtime::RuntimeDefaults runtimeDefaults);

private:
    friend class Serialization;

    bool mEnableBlockReuse;

    std::optional<SizeType32> mMaxTokens;

    std::optional<std::vector<SizeType32>> mMaxAttentionWindowVec;

    std::optional<SizeType32> mSinkTokenLength;

    std::optional<FloatType> mFreeGpuMemoryFraction;

    std::optional<FloatType> mCrossKvCacheFraction;

    std::optional<size_t> mHostCacheSize;

    bool mOnboardBlocks;

    std::optional<RetentionPriority> mSecondaryOffloadMinPriority;

    size_t mEventBufferMaxSize;
};

class ExtendedRuntimePerfKnobConfig
{
public:
    explicit ExtendedRuntimePerfKnobConfig(bool multiBlockMode = true, bool enableContextFMHAFP32Acc = false,
        bool cudaGraphMode = false, SizeType32 cudaGraphCacheSize = 0);

    bool operator==(ExtendedRuntimePerfKnobConfig const& other) const
    {
        return mMultiBlockMode == other.mMultiBlockMode && mEnableContextFMHAFP32Acc == other.mEnableContextFMHAFP32Acc
            && mCudaGraphMode == other.mCudaGraphMode && mCudaGraphCacheSize == other.mCudaGraphCacheSize;
    }

    [[nodiscard]] bool getMultiBlockMode() const;
    [[nodiscard]] bool getEnableContextFMHAFP32Acc() const;
    [[nodiscard]] bool getCudaGraphMode() const;
    [[nodiscard]] SizeType32 getCudaGraphCacheSize() const;

    void setMultiBlockMode(bool multiBlockMode);
    void setEnableContextFMHAFP32Acc(bool enableContextFMHAFP32Acc);
    void setCudaGraphMode(bool cudaGraphMode);
    void setCudaGraphCacheSize(SizeType32 cacheSize);

private:
    friend class Serialization;

    bool mMultiBlockMode;

    bool mEnableContextFMHAFP32Acc;

    bool mCudaGraphMode;

    SizeType32 mCudaGraphCacheSize;
};

class DebugConfig
{
    using StringVec = std::vector<std::string>;

public:
    explicit DebugConfig(bool debugInputTensors = false, bool debugOutputTensors = false,
        StringVec debugTensorNames = {}, SizeType32 debugTensorsMaxIterations = 0);

    bool operator==(DebugConfig const& other) const;

    [[nodiscard]] bool getDebugInputTensors() const;
    [[nodiscard]] bool getDebugOutputTensors() const;
    [[nodiscard]] StringVec const& getDebugTensorNames() const;
    [[nodiscard]] SizeType32 getDebugTensorsMaxIterations() const;

    void setDebugInputTensors(bool debugInputTensors);
    void setDebugOutputTensors(bool debugOutputTensors);
    void setDebugTensorNames(StringVec const& debugTensorNames);
    void setDebugTensorsMaxIterations(SizeType32 debugTensorsMaxIterations);

private:
    friend class Serialization;

    bool mDebugInputTensors;
    bool mDebugOutputTensors;
    StringVec mDebugTensorNames;
    SizeType32 mDebugTensorsMaxIterations;
};

class OrchestratorConfig
{
public:
    explicit OrchestratorConfig(bool isOrchestrator = true, std::string workerExecutablePath = "",
        std::shared_ptr<mpi::MpiComm> orchLeaderComm = nullptr, bool spawnProcesses = true);

    [[nodiscard]] bool getIsOrchestrator() const;
    [[nodiscard]] std::string getWorkerExecutablePath() const;
    [[nodiscard]] std::shared_ptr<mpi::MpiComm> getOrchLeaderComm() const;
    [[nodiscard]] bool getSpawnProcesses() const;

    void setIsOrchestrator(bool isOrchestrator);
    void setWorkerExecutablePath(std::string const& workerExecutablePath);
    void setOrchLeaderComm(std::shared_ptr<mpi::MpiComm> const& orchLeaderComm);
    void setSpawnProcesses(bool spawnProcesses);

private:
    bool mIsOrchestrator;
    std::string mWorkerExecutablePath;
    std::shared_ptr<mpi::MpiComm> mOrchLeaderComm;
    bool mSpawnProcesses;
};

class ParallelConfig
{
public:
    explicit ParallelConfig(CommunicationType commType = CommunicationType::kMPI,
        CommunicationMode commMode = CommunicationMode::kLEADER,
        std::optional<std::vector<SizeType32>> deviceIds = std::nullopt,
        std::optional<std::vector<SizeType32>> participantIds = std::nullopt,
        std::optional<OrchestratorConfig> const& orchestratorConfig = std::nullopt);

    [[nodiscard]] CommunicationType getCommunicationType() const;
    [[nodiscard]] CommunicationMode getCommunicationMode() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getDeviceIds() const;
    [[nodiscard]] std::optional<std::vector<SizeType32>> getParticipantIds() const;
    [[nodiscard]] std::optional<OrchestratorConfig> getOrchestratorConfig() const;

    void setCommunicationType(CommunicationType type);
    void setCommunicationMode(CommunicationMode mode);
    void setDeviceIds(std::vector<SizeType32> const& deviceIds);
    void setParticipantIds(std::vector<SizeType32> const& participantIds);
    void setOrchestratorConfig(OrchestratorConfig const& orchestratorConfig);

private:
    friend class Serialization;

    CommunicationType mCommType;

    CommunicationMode mCommMode;

    std::optional<std::vector<SizeType32>> mDeviceIds;

    std::optional<std::vector<SizeType32>> mParticipantIds;

    std::optional<OrchestratorConfig> mOrchestratorConfig;
};

class PeftCacheConfig
{
public:
    static constexpr SizeType32 kDefaultOptimalAdapterSize = 8;
    static constexpr SizeType32 kDefaultMaxAdapterSize = 64;
    static constexpr SizeType32 kDefaultMaxPagesPerBlockHost = 24;
    static constexpr SizeType32 kDefaultMaxPagesPerBlockDevice = 8;

    explicit PeftCacheConfig(SizeType32 numHostModuleLayer = 0, SizeType32 numDeviceModuleLayer = 0,
        SizeType32 optimalAdapterSize = kDefaultOptimalAdapterSize, SizeType32 maxAdapterSize = kDefaultMaxAdapterSize,
        SizeType32 numPutWorkers = 1, SizeType32 numEnsureWorkers = 1, SizeType32 numCopyStreams = 1,
        SizeType32 maxPagesPerBlockHost = kDefaultMaxPagesPerBlockHost,
        SizeType32 maxPagesPerBlockDevice = kDefaultMaxPagesPerBlockDevice,
        std::optional<float> const& deviceCachePercent = std::nullopt,
        std::optional<size_t> const& hostCacheSize = std::nullopt);

    bool operator==(PeftCacheConfig const& other) const;

    [[nodiscard]] SizeType32 getNumHostModuleLayer() const;
    [[nodiscard]] SizeType32 getNumDeviceModuleLayer() const;
    [[nodiscard]] SizeType32 getOptimalAdapterSize() const;
    [[nodiscard]] SizeType32 getMaxAdapterSize() const;
    [[nodiscard]] SizeType32 getNumPutWorkers() const;
    [[nodiscard]] SizeType32 getNumEnsureWorkers() const;
    [[nodiscard]] SizeType32 getNumCopyStreams() const;
    [[nodiscard]] SizeType32 getMaxPagesPerBlockHost() const;
    [[nodiscard]] SizeType32 getMaxPagesPerBlockDevice() const;
    [[nodiscard]] std::optional<float> getDeviceCachePercent() const;
    [[nodiscard]] std::optional<size_t> getHostCacheSize() const;

private:
    friend class Serialization;

    SizeType32 mNumHostModuleLayer;
    SizeType32 mNumDeviceModuleLayer;
    SizeType32 mOptimalAdapterSize;
    SizeType32 mMaxAdapterSize;
    SizeType32 mNumPutWorkers;
    SizeType32 mNumEnsureWorkers;
    SizeType32 mNumCopyStreams;
    SizeType32 mMaxPagesPerBlockHost;
    SizeType32 mMaxPagesPerBlockDevice;
    std::optional<FloatType> mDeviceCachePercent;
    std::optional<size_t> mHostCacheSize;
};

class DecodingConfig
{
public:
    explicit DecodingConfig(std::optional<DecodingMode> decodingMode = std::nullopt,
        std::optional<LookaheadDecodingConfig> lookaheadDecodingConfig = std::nullopt,
        std::optional<MedusaChoices> medusaChoices = std::nullopt,
        std::optional<EagleConfig> eagleConfig = std::nullopt);

    bool operator==(DecodingConfig const& other) const;

    void setDecodingMode(DecodingMode const&);
    [[nodiscard]] std::optional<DecodingMode> getDecodingMode() const;

    void setLookaheadDecoding(LookaheadDecodingConfig const& lookaheadDecodingConfig);
    void enableSeamlessLookaheadDecoding();
    [[nodiscard]] std::optional<LookaheadDecodingConfig> getLookaheadDecodingConfig() const;
    [[nodiscard]] SizeType32 getLookaheadDecodingMaxNumRequest() const;

    void setMedusaChoices(MedusaChoices const&);
    [[nodiscard]] std::optional<MedusaChoices> getMedusaChoices() const;

    void setEagleConfig(EagleConfig const&);
    [[nodiscard]] std::optional<EagleConfig> getEagleConfig() const;

private:
    friend class Serialization;

    std::optional<DecodingMode> mDecodingMode;
    std::optional<LookaheadDecodingConfig> mLookaheadDecodingConfig;
    std::optional<MedusaChoices> mMedusaChoices;
    std::optional<EagleConfig> mEagleConfig;
    static constexpr SizeType32 mLookaheadDecodingMaxNumRequest = 8;
};

class GuidedDecodingConfig
{
public:
    enum class GuidedDecodingBackend
    {
        kXGRAMMAR = 0,
    };

    explicit GuidedDecodingConfig(GuidedDecodingBackend backend,
        std::optional<std::vector<std::string>> encodedVocab = std::nullopt,
        std::optional<std::string> tokenizerStr = std::nullopt,
        std::optional<std::vector<TokenIdType>> stopTokenIds = std::nullopt);

    bool operator==(GuidedDecodingConfig const& other) const;

    void setBackend(GuidedDecodingBackend const& backend);
    [[nodiscard]] GuidedDecodingBackend getBackend() const;

    void setEncodedVocab(std::vector<std::string> const& encodedVocab);
    [[nodiscard]] std::optional<std::vector<std::string>> getEncodedVocab() const;

    void setTokenizerStr(std::string const& tokenizerStr);
    [[nodiscard]] std::optional<std::string> getTokenizerStr() const;

    void setStopTokenIds(std::vector<TokenIdType> const& stopTokenIds);
    [[nodiscard]] std::optional<std::vector<TokenIdType>> getStopTokenIds() const;

    void validate() const;

private:
    friend class Serialization;

    GuidedDecodingBackend mBackend;
    std::optional<std::vector<std::string>> mEncodedVocab;
    std::optional<std::string> mTokenizerStr;
    std::optional<std::vector<TokenIdType>> mStopTokenIds;
};

class LogitsPostProcessorConfig
{
public:
    explicit LogitsPostProcessorConfig(std::optional<LogitsPostProcessorMap> processorMap = std::nullopt,
        std::optional<LogitsPostProcessorBatched> processorBatched = std::nullopt, bool replicate = true);

    [[nodiscard]] std::optional<LogitsPostProcessorMap> getProcessorMap() const;
    [[nodiscard]] std::optional<LogitsPostProcessorBatched> getProcessorBatched() const;
    [[nodiscard]] bool getReplicate() const;

    void setProcessorMap(LogitsPostProcessorMap const& processorMap);
    void setProcessorBatched(LogitsPostProcessorBatched const& processorBatched);
    void setReplicate(bool replicate);

private:
    std::optional<LogitsPostProcessorMap> mProcessorMap;
    std::optional<LogitsPostProcessorBatched> mProcessorBatched;
    bool mReplicate;
};

class ExecutorConfig
{
public:
    static constexpr uint64_t kDefaultMaxSeqIdleMicroseconds = 180000000;

    static constexpr SizeType32 kDefaultIterStatsMaxIterations = 1000;

    static constexpr SizeType32 kDefaultRequestStatsMaxIterations = 0;

    explicit ExecutorConfig(SizeType32 maxBeamWidth = 1, SchedulerConfig schedulerConfig = SchedulerConfig(),
        KvCacheConfig kvCacheConfig = KvCacheConfig(), bool enableChunkedContext = true, bool normalizeLogProbs = true,
        SizeType32 iterStatsMaxIterations = kDefaultIterStatsMaxIterations,
        SizeType32 requestStatsMaxIterations = kDefaultRequestStatsMaxIterations,
        BatchingType batchingType = BatchingType::kINFLIGHT, std::optional<SizeType32> maxBatchSize = std::nullopt,
        std::optional<SizeType32> maxNumTokens = std::nullopt,
        std::optional<ParallelConfig> parallelConfig = std::nullopt,
        std::optional<PeftCacheConfig> const& peftCacheConfig = std::nullopt,
        std::optional<LogitsPostProcessorConfig> logitsPostProcessorConfig = std::nullopt,
        std::optional<DecodingConfig> decodingConfig = std::nullopt, float gpuWeightsPercent = 1,
        std::optional<SizeType32> maxQueueSize = std::nullopt,
        ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig = ExtendedRuntimePerfKnobConfig(),
        std::optional<DebugConfig> debugConfig = std::nullopt, SizeType32 recvPollPeriodMs = 0,
        uint64_t maxSeqIdleMicroseconds = kDefaultMaxSeqIdleMicroseconds,
        std::optional<SpeculativeDecodingConfig> specDecConfig = std::nullopt,
        std::optional<GuidedDecodingConfig> guidedDecodingConfig = std::nullopt,
        std::optional<std::vector<std::string>> additionalOutputNames = std::nullopt);

    [[nodiscard]] SizeType32 getMaxBeamWidth() const;
    [[nodiscard]] SchedulerConfig getSchedulerConfig() const;
    [[nodiscard]] KvCacheConfig getKvCacheConfig() const;
    [[nodiscard]] SchedulerConfig& getSchedulerConfigRef();
    [[nodiscard]] KvCacheConfig& getKvCacheConfigRef();
    [[nodiscard]] bool getEnableChunkedContext() const;
    [[nodiscard]] bool getNormalizeLogProbs() const;
    [[nodiscard]] SizeType32 getIterStatsMaxIterations() const;
    [[nodiscard]] SizeType32 getRequestStatsMaxIterations() const;
    [[nodiscard]] BatchingType getBatchingType() const;
    [[nodiscard]] std::optional<SizeType32> getMaxBatchSize() const;
    [[nodiscard]] std::optional<SizeType32> getMaxNumTokens() const;
    [[nodiscard]] std::optional<ParallelConfig> getParallelConfig() const;
    [[nodiscard]] std::optional<PeftCacheConfig> getPeftCacheConfig() const;
    [[nodiscard]] std::optional<LogitsPostProcessorConfig> getLogitsPostProcessorConfig() const;
    [[nodiscard]] std::optional<DecodingConfig> getDecodingConfig() const;
    [[nodiscard]] float getGpuWeightsPercent() const;
    [[nodiscard]] std::optional<SizeType32> getMaxQueueSize() const;
    [[nodiscard]] ExtendedRuntimePerfKnobConfig getExtendedRuntimePerfKnobConfig() const;
    [[nodiscard]] std::optional<DebugConfig> getDebugConfig() const;
    [[nodiscard]] SizeType32 getRecvPollPeriodMs() const;
    [[nodiscard]] uint64_t getMaxSeqIdleMicroseconds() const;
    [[nodiscard]] std::optional<SpeculativeDecodingConfig> getSpecDecConfig() const;
    [[nodiscard]] std::optional<GuidedDecodingConfig> getGuidedDecodingConfig() const;
    [[nodiscard]] std::optional<std::vector<std::string>> getAdditionalOutputNames() const;

    void setMaxBeamWidth(SizeType32 maxBeamWidth);
    void setMaxBatchSize(SizeType32 maxBatchSize);
    void setMaxNumTokens(SizeType32 maxNumTokens);
    void setSchedulerConfig(SchedulerConfig const& schedulerConfig);
    void setKvCacheConfig(KvCacheConfig const& kvCacheConfig);
    void setEnableChunkedContext(bool enableChunkedContext);
    void setNormalizeLogProbs(bool normalizeLogProbs);
    void setIterStatsMaxIterations(SizeType32 iterStatsMaxIterations);
    void setRequestStatsMaxIterations(SizeType32 requestStatsMaxIterations);
    void setBatchingType(BatchingType batchingType);
    void setParallelConfig(ParallelConfig const& parallelConfig);
    void setPeftCacheConfig(PeftCacheConfig const& peftCacheConfig);
    void setLogitsPostProcessorConfig(LogitsPostProcessorConfig const& logitsPostProcessorConfig);
    void setDecodingConfig(DecodingConfig const& decodingConfig);
    void setGpuWeightsPercent(float const& gpuWeightsPercent);
    void setMaxQueueSize(std::optional<SizeType32> const& maxQueueSize);
    void setExtendedRuntimePerfKnobConfig(ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig);
    void setDebugConfig(DebugConfig const& debugConfig);
    void setRecvPollPeriodMs(SizeType32 const& recvPollPeriodMs);
    void setMaxSeqIdleMicroseconds(uint64_t maxNumTokens);
    void setSpecDecConfig(SpeculativeDecodingConfig const& specDecConfig);
    void setGuidedDecodingConfig(GuidedDecodingConfig const& guidedDecodingConfig);
    void setAdditionalOutputNames(std::vector<std::string> const& additionalOutputNames);

private:
    friend class Serialization;

    SizeType32 mMaxBeamWidth;

    SchedulerConfig mSchedulerConfig;

    KvCacheConfig mKvCacheConfig;

    bool mEnableChunkedContext;

    bool mNormalizeLogProbs;

    SizeType32 mIterStatsMaxIterations;

    SizeType32 mRequestStatsMaxIterations;

    BatchingType mBatchingType;

    std::optional<SizeType32> mMaxBatchSize;

    std::optional<SizeType32> mMaxNumTokens;

    std::optional<ParallelConfig> mParallelConfig;
    std::optional<PeftCacheConfig> mPeftCacheConfig;

    std::optional<LogitsPostProcessorConfig> mLogitsPostProcessorConfig;

    std::optional<DecodingConfig> mDecodingConfig;

    float mGpuWeightsPercent;

    std::optional<SizeType32> mMaxQueueSize;

    ExtendedRuntimePerfKnobConfig mExtendedRuntimePerfKnobConfig;

    std::optional<DebugConfig> mDebugConfig;

    SizeType32 mRecvPollPeriodMs;

    uint64_t mMaxSeqIdleMicroseconds;

    std::optional<SpeculativeDecodingConfig> mSpeculativeDecodingConfig;

    std::optional<GuidedDecodingConfig> mGuidedDecodingConfig;

    std::optional<std::vector<std::string>> mAdditionalOutputNames;
};

struct KVCacheCreatedData
{
    std::vector<SizeType32> numBlocksPerCacheLevel;
};

struct KVCacheStoredBlockData
{

    KVCacheStoredBlockData(IdType blockHash, suggestify::runtime::VecUniqueTokens tokens,
        suggestify::runtime::LoraTaskIdType loraId, SizeType32 cacheLevel, SizeType32 priority)
        : blockHash{blockHash}
        , tokens{std::move(tokens)}
        , loraId{loraId}
        , cacheLevel{cacheLevel}
        , priority{priority}
    {
    }

    IdType blockHash;
    suggestify::runtime::VecUniqueTokens tokens;
    suggestify::runtime::LoraTaskIdType loraId;
    SizeType32 cacheLevel;
    SizeType32 priority;
};

struct KVCacheStoredData
{
    std::optional<IdType> parentHash;
    std::vector<KVCacheStoredBlockData> blocks;
};

struct KVCacheRemovedData
{
    std::vector<IdType> blockHashes;
};

template <typename T>
struct KVCacheEventDiff
{
    T oldValue;
    T newValue;
};

struct KVCacheUpdatedData
{

    explicit KVCacheUpdatedData(IdType blockHash)
        : blockHash{blockHash} {};

    KVCacheUpdatedData& cacheLevelUpdated(SizeType32 oldValue, SizeType32 newValue)
    {
        cacheLevel = KVCacheEventDiff<SizeType32>{oldValue, newValue};
        return *this;
    }

    KVCacheUpdatedData& priorityUpdated(SizeType32 oldValue, SizeType32 newValue)
    {
        priority = KVCacheEventDiff<SizeType32>{oldValue, newValue};
        return *this;
    }

    IdType blockHash;
    std::optional<KVCacheEventDiff<SizeType32>> cacheLevel = std::nullopt;
    std::optional<KVCacheEventDiff<SizeType32>> priority = std::nullopt;
};

using KVCacheEventData = std::variant<KVCacheCreatedData, KVCacheStoredData, KVCacheRemovedData, KVCacheUpdatedData>;

struct KVCacheEvent
{

    KVCacheEvent(IdType eventId, KVCacheEventData data);

    IdType eventId;
    KVCacheEventData data;
};

class KVCacheEventManager
{
public:
    KVCacheEventManager(
        std::shared_ptr<suggestify::batch_manager::kv_cache_manager::BaseKVCacheManager> kvCacheManager);

    std::deque<KVCacheEvent> getLatestEvents(std::optional<std::chrono::milliseconds> timeout = std::nullopt);

private:
    std::shared_ptr<suggestify::batch_manager::kv_cache_manager::BaseKVCacheManager> kvCacheManager;
};

class Executor
{

public:
    Executor(std::filesystem::path const& modelPath, ModelType modelType, ExecutorConfig const& executorConfig);

    Executor(std::filesystem::path const& encoderModelPath, std::filesystem::path const& decoderModelPath,
        ModelType modelType, ExecutorConfig const& executorConfig);

    Executor(BufferView const& engineBuffer, std::string const& jsonConfigStr, ModelType modelType,
        ExecutorConfig const& executorConfig,
        std::optional<std::map<std::string, Tensor>> const& managedWeights = std::nullopt);

    Executor(BufferView const& encoderEngineBuffer, std::string const& encoderJsonConfigStr,
        BufferView const& decoderEngineBuffer, std::string const& decoderJsonConfigStr, ModelType modelType,
        ExecutorConfig const& executorConfig);

    Executor(std::shared_ptr<Model> model, ExecutorConfig const& executorConfig);

    Executor(
        std::shared_ptr<Model> encoderModel, std::shared_ptr<Model> decoderModel, ExecutorConfig const& executorConfig);

    ~Executor();
    Executor(Executor const& executor) = delete;
    Executor& operator=(Executor const& executor) = delete;
    Executor(Executor&&) = default;
    Executor& operator=(Executor&&) = default;

    [[nodiscard]] IdType enqueueRequest(Request const& request);

    [[nodiscard]] std::vector<IdType> enqueueRequests(std::vector<Request> const& requests);

    [[nodiscard]] std::vector<Response> awaitResponses(
        std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    [[nodiscard]] std::vector<Response> awaitResponses(
        IdType const& requestId, std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    [[nodiscard]] std::vector<std::vector<Response>> awaitResponses(
        std::vector<IdType> const& requestIds, std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    [[nodiscard]] SizeType32 getNumResponsesReady(std::optional<IdType> const& requestId = std::nullopt) const;

    void cancelRequest(IdType requestId);

    void shutdown();

    std::deque<IterationStats> getLatestIterationStats();

    std::deque<RequestStatsPerIteration> getLatestRequestStats();

    std::deque<DebugTensorsPerIteration> getLatestDebugTensors();

    [[nodiscard]] bool canEnqueueRequests() const;

    [[nodiscard]] bool isParticipant() const;

    std::optional<std::shared_ptr<KVCacheEventManager>> getKVCacheEventManager() const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

class JsonSerialization
{
public:
    [[nodiscard]] static std::string toJsonStr(IterationStats const& iterationStats);

    [[nodiscard]] static std::string toJsonStr(RequestStatsPerIteration const& requestStatsPerIter);

    [[nodiscard]] static std::string toJsonStr(RequestStats const& requestStats);
};

}
