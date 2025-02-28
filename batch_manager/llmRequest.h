
#pragma once

#include "../common/assert.h"
#include "../common/logger.h"
#include "../executor/executor.h"
#include "../runtime/bufferManager.h"
#include "../runtime/iBuffer.h"
#include "../runtime/iTensor.h"
#include "../runtime/samplingConfig.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace sugesstify::batch_manager
{

enum class LlmRequestState : int32_t
{
    kUNKNOWN = 0,
    kENCODER_INIT = 1,
    kCONTEXT_INIT = 2,
    KDISAGG_GENERATION_TRANS_COMPLETE = 3,
    kGENERATION_IN_PROGRESS = 4,
    kGENERATION_TO_COMPLETE = 5,
    kGENERATION_COMPLETE = 6,
    kDISAGG_GENERATION_INIT = 7,
    kDISAGG_CONTEXT_TRANS_IN_PROGRESS = 8,
    kDISAGG_CONTEXT_COMPLETE = 9,
    kDISAGG_GENERATION_TRANS_IN_PROGRESS = 10,
    kDISAGG_CONTEXT_INIT_AND_TRANS = 11,
};

enum LlmRequestType
{
    LLMREQUEST_TYPE_CONTEXT_AND_GENERATION = 0,
    LLMREQUEST_TYPE_CONTEXT_ONLY = 1,
    LLMREQUEST_TYPE_GENERATION_ONLY = 2
};

class ContextProgress;

template <typename TTensor, typename TStream = runtime::BufferManager::CudaStreamPtr>
class GenericLlmRequest
{
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

public:
    using SizeType32 = runtime::SizeType32;
    using TokenIdType = runtime::TokenIdType;
    using RequestIdType = std::uint64_t;
    using LoraTaskIdType = runtime::LoraTaskIdType;
    using VecTokens = std::vector<TokenIdType>;
    using TokenExtraIdType = runtime::TokenExtraIdType;
    using VecTokenExtraIds = runtime::VecTokenExtraIds;
    using VecLogProbs = std::vector<float>;
    using BeamTokens = std::vector<VecTokens>;
    using UniqueToken = runtime::UniqueToken;
    using VecUniqueTokens = runtime::VecUniqueTokens;
    using BeamUniqueTokens = std::vector<VecUniqueTokens>;
    using TensorPtr = TTensor;
    using LogitsPostProcessor = std::function<void(
        RequestIdType, TensorPtr&, BeamTokens const&, TStream const&, std::optional<RequestIdType>)>;
    using RequestPtr = std::shared_ptr<GenericLlmRequest>;
    using MillisecondsType = std::chrono::milliseconds;

    GenericLlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, std::shared_ptr<VecTokens> const& inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<TensorPtr> mropeRotaryCosSin = std::nullopt,
        std::optional<SizeType32> mropePositionDeltas = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<executor::KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<std::shared_ptr<VecTokens>> const& draftTokens = std::nullopt,
        std::optional<TensorPtr> draftLogits = std::nullopt, bool excludeInputFromOutput = false,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false,
        std::optional<std::shared_ptr<VecTokens>> encoderInputTokens = std::nullopt, bool returnEncoderOutput = false,
        std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority,
        std::optional<TensorPtr> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<TensorPtr> crossAttentionMask = std::nullopt,
        LlmRequestType llmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<std::shared_ptr<VecTokenExtraIds>> inputTokenExtraIds = std::nullopt,
        SizeType32 numReturnSequences = 1, std::optional<executor::EagleConfig> eagleConfig = std::nullopt,
        std::optional<TensorPtr> skipCrossAttnBlocks = std::nullopt, bool returnPerfMetrics = false,
        std::optional<executor::GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt)
        : mRequestId(requestId)
        , mPromptLen(inputTokens->size())
        , mMaxNewTokens(maxNewTokens)
        , mSamplingConfig(samplingConfig)
        , mState(LlmRequestState::kCONTEXT_INIT)
        , mEndId(endId)
        , mPadId(padId)
        , mLogitsPostProcessor(std::move(logitsPostProcessor))
        , mApplyLogitsPostProcessorBatched(applyLogitsPostProcessorBatched)
        , mClientId(clientId)
        , mIsStreaming(isStreaming)
        , mOrigPromptLen(mPromptLen)
        , mNumPreDecodedTokens(samplingConfig.beamWidth, 0)
        , mMaxSentTokenLen(mPromptLen)
        , mEmbeddingBias(std::move(embeddingBias))
        , mBadWordsList(std::move(badWordsList))
        , mStopWordsList(std::move(stopWordsList))
        , mPositionIds(std::move(positionIds))
        , mPromptEmbeddingTable(std::move(promptEmbeddingTable))
        , mPromptVocabSize(promptVocabSize)
        , mMropeRotaryCosSin(std::move(mropeRotaryCosSin))
        , mMropePositionDeltas(mropePositionDeltas)
        , mLoraTaskId(loraTaskId)
        , mLoraWeights(std::move(loraWeights))
        , mLoraConfig(std::move(loraConfig))
        , mLookaheadConfig(lookaheadConfig)
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mContextChunkSize{mPromptLen}
        , mLogProbs(samplingConfig.beamWidth)
        , mCumLogProbs(samplingConfig.beamWidth)
        , mDraftTokens(draftTokens.value_or(std::make_shared<VecTokens>()))
        , mDraftLogits(std::move(draftLogits))
        , mNumTokensPerIteration(1)
        , mReturnAllGeneratedTokens(isStreaming && (samplingConfig.beamWidth > 1))
        , mReturnContextLogits(returnContextLogits)
        , mReturnGenerationLogits(returnGenerationLogits)
        , mExcludeInputFromOutput(excludeInputFromOutput)
        , mEncoderTokens(std::move(encoderInputTokens))
        , mReturnEncoderOutput(returnEncoderOutput)
        , mDecodingIter(0)
        , mPriority(priority)
        , mFinishReasons(samplingConfig.beamWidth)
        , mEncoderInputFeatures(std::move(encoderInputFeatures))
        , mEncoderOutputLength(encoderOutputLength)
        , mCrossAttentionMask(std::move(crossAttentionMask))
        , mLlmRequestType(llmRequestType)
        , mInputTokenExtraIds(std::move(inputTokenExtraIds))
        , mNumReturnSequences(numReturnSequences)
        , mEagleConfig(std::move(eagleConfig))
        , mSequenceIndex(0)
        , mSkipCrossAttnBlocks(std::move(skipCrossAttnBlocks))
        , mReturnPerfMetrics(returnPerfMetrics)
        , mGuidedDecodingParams(std::move(guidedDecodingParams))
        , mAllottedTimeMs(allottedTimeMs)
    {
        if (mEncoderTokens.has_value() || encoderInputFeatures.has_value())
        {
            mState = LlmRequestState::kENCODER_INIT;
        }

        initialize(*inputTokens, returnLogProbs);
    }

    GenericLlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, VecTokens const& inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt, bool returnLogProbs = false,
        bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<VecTokens> draftTokens = std::nullopt, std::optional<TensorPtr> draftLogits = std::nullopt,
        bool excludeInputFromOutput = false, std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false, std::optional<VecTokens> encoderInputTokens = std::nullopt,
        bool returnEncoderOutput = false, std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority, SizeType32 numReturnSequences = 1)
        : mRequestId(requestId)
        , mPromptLen(inputTokens.size())
        , mMaxNewTokens(maxNewTokens)
        , mSamplingConfig(samplingConfig)
        , mState(LlmRequestState::kCONTEXT_INIT)
        , mEndId(endId)
        , mPadId(padId)
        , mLogitsPostProcessor(logitsPostProcessor)
        , mApplyLogitsPostProcessorBatched(applyLogitsPostProcessorBatched)
        , mClientId(clientId)
        , mIsStreaming(isStreaming)
        , mOrigPromptLen(mPromptLen)
        , mNumPreDecodedTokens(samplingConfig.beamWidth, 0)
        , mMaxSentTokenLen(mPromptLen)
        , mEmbeddingBias(std::move(embeddingBias))
        , mBadWordsList(std::move(badWordsList))
        , mStopWordsList(std::move(stopWordsList))
        , mPositionIds(std::move(positionIds))
        , mPromptEmbeddingTable(std::move(promptEmbeddingTable))
        , mPromptVocabSize(promptVocabSize)
        , mLoraTaskId(loraTaskId)
        , mLoraWeights(std::move(loraWeights))
        , mLoraConfig(std::move(loraConfig))
        , mLookaheadConfig(lookaheadConfig)
        , mContextChunkSize(mPromptLen)
        , mContextCurrentPosition(0)
        , mLogProbs(samplingConfig.beamWidth)
        , mCumLogProbs(samplingConfig.beamWidth)
        , mDraftLogits(draftLogits)
        , mNumTokensPerIteration(1)
        , mReturnAllGeneratedTokens(isStreaming && (samplingConfig.beamWidth > 1))
        , mReturnContextLogits(returnContextLogits)
        , mReturnGenerationLogits(returnGenerationLogits)
        , mExcludeInputFromOutput(excludeInputFromOutput)
        , mReturnEncoderOutput(returnEncoderOutput)
        , mDecodingIter(0)
        , mPriority(priority)
        , mFinishReasons(samplingConfig.beamWidth)
        , mNumReturnSequences(numReturnSequences)
        , mSequenceIndex(0)
    {
        if (draftTokens)
        {
            mDraftTokens = std::make_shared<VecTokens>(draftTokens.value());
        }
        else
        {
            mDraftTokens = std::make_shared<VecTokens>();
        }
        if (encoderInputTokens)
        {
            mEncoderTokens = std::make_shared<VecTokens>(encoderInputTokens.value());
        }
        else
        {
            mEncoderTokens = std::nullopt;
        }
        if (mEncoderTokens.has_value())
        {
            mState = LlmRequestState::kENCODER_INIT;
        }
        initialize(inputTokens, returnLogProbs);
    }

    GenericLlmRequest(RequestIdType requestId, executor::Request const& req)
        : mRequestId(requestId)
        , mPromptLen(req.getInputTokenIds().size())
        , mMaxNewTokens(req.getMaxTokens())
        , mSamplingConfig(req.getSamplingConfig(), req.getExternalDraftTokensConfig())
        , mState(LlmRequestState::kCONTEXT_INIT)
        , mEndId(req.getEndId())
        , mPadId(req.getPadId())
        , mClientId(req.getClientId())
        , mIsStreaming(req.getStreaming())
        , mOrigPromptLen(mPromptLen)
        , mNumPreDecodedTokens(mSamplingConfig.beamWidth, 0)
        , mMaxSentTokenLen(mPromptLen)
        , mEmbeddingBias(std::nullopt)
        , mBadWordsList(std::nullopt)
        , mStopWordsList(std::nullopt)
        , mPositionIds(std::nullopt)
        , mPromptEmbeddingTable(std::nullopt)
        , mPromptVocabSize(std::nullopt)
        , mMropeRotaryCosSin(std::nullopt)
        , mMropePositionDeltas(std::nullopt)
        , mLoraTaskId(std::nullopt)
        , mLoraWeights(std::nullopt)
        , mLoraConfig(std::nullopt)
        , mLookaheadConfig(std::nullopt)
        , mKvCacheRetentionConfig(std::nullopt)
        , mContextChunkSize{mPromptLen}
        , mLogProbs(mSamplingConfig.beamWidth)
        , mCumLogProbs(mSamplingConfig.beamWidth)
        , mDraftTokens(std::make_shared<VecTokens>())
        , mDraftLogits(std::nullopt)
        , mNumTokensPerIteration(1)
        , mReturnAllGeneratedTokens(req.getReturnAllGeneratedTokens())
        , mReturnContextLogits(req.getOutputConfig().returnContextLogits)
        , mReturnGenerationLogits(req.getOutputConfig().returnGenerationLogits)
        , mExcludeInputFromOutput(req.getOutputConfig().excludeInputFromOutput)
        , mEncoderTokens(std::nullopt)
        , mReturnEncoderOutput(req.getOutputConfig().returnEncoderOutput)
        , mDecodingIter(0)
        , mPriority(req.getPriority())
        , mFinishReasons(mSamplingConfig.beamWidth)
        , mEncoderInputFeatures(std::nullopt)
        , mEncoderOutputLength(req.getEncoderOutputLength())
        , mContextPhaseParams(req.getContextPhaseParams())
        , mInputTokenExtraIds(std::nullopt)
        , mNumReturnSequences(1)
        , mEagleConfig(req.getEagleConfig())
        , mSequenceIndex(0)
        , mReturnPerfMetrics(req.getOutputConfig().returnPerfMetrics)
        , mGuidedDecodingParams(req.getGuidedDecodingParams())
        , mAllottedTimeMs(req.getAllottedTimeMs())
    {
        if (req.getRequestType() == executor::RequestType::REQUEST_TYPE_GENERATION_ONLY)
        {
            mState = LlmRequestState::kDISAGG_GENERATION_INIT;
        }
        if (mIsStreaming && mSamplingConfig.beamWidth > 1 && !mReturnAllGeneratedTokens)
        {
            TLLM_LOG_WARNING(
                "Setting mReturnAllGeneratedTokens to True since streaming AND beam search are done simultaneously. "
                "Returning the full beams at each streaming step is needed because beam search + streaming can change "
                "previous outputs. Initialize request with mReturnAllGeneratedTokens = True to dismiss this error. "
                "WARNING: using this option may increase network usage significantly (quadratically w.r.t output "
                "length).");
            mReturnAllGeneratedTokens = true;
        }

        if (mIsStreaming && mSamplingConfig.beamWidth > 1 && mReturnGenerationLogits)
        {
            TLLM_LOG_WARNING(
                "Returning generation logits when streaming is enabled and beamWidth > 1 is not allowed. "
                "This is because the logits may appear in irrelevant order when the beams are gathered, "
                "since logits are not. Disabling returnGenerationLogits.");
            mReturnGenerationLogits = false;
        }

        if (req.getEncoderInputTokenIds().has_value() || req.getEncoderInputFeatures().has_value())
        {
            mState = LlmRequestState::kENCODER_INIT;
            if (req.getEncoderInputTokenIds().has_value())
            {
                mEncoderTokens = std::make_shared<VecTokens>(req.getEncoderInputTokenIds().value());
            }
        }

        if (req.getEmbeddingBias())
        {
            mEmbeddingBias
                = sugesstify::runtime::ITensor::view(executor::detail::toITensor(req.getEmbeddingBias().value()));
            mEmbeddingBias.value()->unsqueeze(0);
        }
        if (req.getBadWords())
        {
            mBadWordsList = createListTensor(req.getBadWords().value());
        }
        if (req.getStopWords())
        {
            mStopWordsList = createListTensor(req.getStopWords().value());
        }

        if (req.getPositionIds())
        {
            mPositionIds = std::make_shared<std::vector<SizeType32>>(req.getPositionIds().value());
        }

        auto pTuningConfig = req.getPromptTuningConfig();
        if (pTuningConfig)
        {
            mPromptEmbeddingTable = sugesstify::runtime::ITensor::view(
                executor::detail::toITensor(pTuningConfig.value().getEmbeddingTable()));
            TLLM_CHECK(mPromptEmbeddingTable.value()->getShape().nbDims == 2);
            mPromptVocabSize = mPromptEmbeddingTable.value()->getShape().d[0];
            mPromptEmbeddingTable.value()->unsqueeze(0);

            if (pTuningConfig->getInputTokenExtraIds())
            {
                mInputTokenExtraIds
                    = std::make_shared<VecTokenExtraIds>(pTuningConfig->getInputTokenExtraIds().value());
            }
        }
        auto mRopeConfig = req.getMropeConfig();
        if (mRopeConfig)
        {
            mMropeRotaryCosSin = executor::detail::toITensor(mRopeConfig.value().getMRopeRotaryCosSin());
            mMropePositionDeltas = mRopeConfig.value().getMRopePositionDeltas();
        }

        auto loraConfig = req.getLoraConfig();
        if (loraConfig)
        {
            mLoraTaskId = loraConfig->getTaskId();
            if (loraConfig.value().getWeights())
            {
                mLoraWeights = sugesstify::runtime::ITensor::view(
                    executor::detail::toITensor(loraConfig.value().getWeights().value()));
                mLoraWeights.value()->unsqueeze(0);
            }

            if (loraConfig.value().getConfig())
            {
                mLoraConfig = sugesstify::runtime::ITensor::view(
                    executor::detail::toITensor(loraConfig.value().getConfig().value()));
                mLoraConfig.value()->unsqueeze(0);
            }
        }

        auto externalDraftTokensConfig = req.getExternalDraftTokensConfig();
        if (externalDraftTokensConfig)
        {
            mDraftTokens = std::make_shared<VecTokens>(externalDraftTokensConfig.value().getTokens());

            if (externalDraftTokensConfig.value().getLogits())
            {
                mDraftLogits = executor::detail::toITensor(externalDraftTokensConfig.value().getLogits().value());
            }

        }

        if (req.getOutputConfig().additionalModelOutputs.has_value())
        {
            auto const additionalModelOutputs
                = req.getOutputConfig()
                      .additionalModelOutputs.value();
            for (auto const& modelOutput : additionalModelOutputs)
            {
                if (modelOutput.gatherContext)
                {
                    mAdditionalContextOutputTensors.emplace(modelOutput.name, TensorPtr{});
                }
                mAdditionalGenerationOutputTensors.emplace(modelOutput.name, TensorPtr{});
            }
        }

        auto const& encoderInputFeatures = req.getEncoderInputFeatures();
        if (encoderInputFeatures.has_value())
        {
            mEncoderInputFeatures = executor::detail::toITensor(encoderInputFeatures.value());
        }
        else
        {
            mEncoderInputFeatures = std::nullopt;
        }

        auto const& crossAttentionMask = req.getCrossAttentionMask();
        if (crossAttentionMask.has_value())
        {
            mCrossAttentionMask = executor::detail::toITensor(crossAttentionMask.value());
        }
        else
        {
            mCrossAttentionMask = std::nullopt;
        }

        auto const& skipCrossAttnBlocks = req.getSkipCrossAttnBlocks();
        if (skipCrossAttnBlocks.has_value())
        {
            mSkipCrossAttnBlocks = executor::detail::toITensor(skipCrossAttnBlocks.value());
        }
        else
        {
            mSkipCrossAttnBlocks = std::nullopt;
        }

        switch (req.getRequestType())
        {
        case executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION:
            mLlmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION;
            break;
        case executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY:
            mLlmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_ONLY;
            break;
        case executor::RequestType::REQUEST_TYPE_GENERATION_ONLY:
            mLlmRequestType = LlmRequestType::LLMREQUEST_TYPE_GENERATION_ONLY;
            break;
        default: throw std::runtime_error("Unsupported request type found.");
        }

        initialize(req.getInputTokenIds(), req.getOutputConfig().returnLogProbs);
    }

    void validate(SizeType32 maxInputLen, SizeType32 maxSequenceLen, SizeType32 maxDraftLen,
        std::optional<SizeType32> maxEncoderInputLen = std::nullopt, bool enableKVCacheReuse = false)
    {
        TLLM_CHECK_WITH_INFO(!(maxEncoderInputLen.has_value() && getEncoderInputLen() > maxEncoderInputLen.value()),
            "Encoder length (%d) exceeds maximum encoder input length (%d).", getEncoderInputLen(),
            maxEncoderInputLen.value());

        if (mPromptLen > maxInputLen)
        {
            TLLM_THROW(
                "Prompt length (%d) exceeds maximum input length (%d). Set log level to info and check "
                "TRTGptModel logs for how maximum input length is set",
                mPromptLen, maxInputLen);
        }

        auto draftLenPerEngineStep = maxDraftLen;
        auto const& draftTokens = getDraftTokens();
        if (draftTokens && !draftTokens->empty())
        {
            auto const inputDraftTokensLen = static_cast<SizeType32>(draftTokens->size());
            if (inputDraftTokensLen > maxDraftLen)
            {
                TLLM_THROW("Draft tokens length (%d) exceeds maximum draft tokens length (%d).", inputDraftTokensLen,
                    maxDraftLen);
            }
            draftLenPerEngineStep = inputDraftTokensLen;

            if (mPromptLen + draftLenPerEngineStep > maxInputLen)
            {
                auto const newDraftLenPerEngineStep = maxInputLen - mPromptLen;
                TLLM_LOG_WARNING(
                    "Prompt length + number of draft tokens (%d + %d) exceeds maximum input length (%d)."
                    "Number of draft tokens is changed to (%d)",
                    mPromptLen, draftLenPerEngineStep, maxInputLen, newDraftLenPerEngineStep);
                draftLenPerEngineStep = newDraftLenPerEngineStep;
                mDraftTokens->resize(draftLenPerEngineStep);
            }
        }

        if (mPromptLen + mMaxNewTokens + draftLenPerEngineStep > maxSequenceLen)
        {
            auto const maxNewTokens = maxSequenceLen - mPromptLen - draftLenPerEngineStep;
            TLLM_LOG_WARNING(
                "Prompt length + number of requested output tokens + draft tokens per step (%d + %d + %d) exceeds "
                "maximum sequence length (%d). "
                "Number of requested output tokens is changed to (%d).",
                mPromptLen, mMaxNewTokens, draftLenPerEngineStep, maxSequenceLen, maxNewTokens);
            mMaxNewTokens = maxNewTokens;
        }

        TLLM_CHECK_WITH_INFO(mSamplingConfig.validate(), "Incorrect sampling config");

        if (enableKVCacheReuse && mPromptEmbeddingTable.has_value() && mPromptVocabSize.has_value())
        {
            TLLM_CHECK_WITH_INFO(mInputTokenExtraIds.has_value() && mInputTokenExtraIds.value(),
                "Input token extra ids must be provided when enabling kv cache reuse with prompt table");
            TLLM_CHECK_WITH_INFO(mInputTokenExtraIds.value()->size() == static_cast<size_t>(mOrigPromptLen),
                "inputTokenExtraIds vector size (%lu) must be the same as input token vector size (%lu).",
                mInputTokenExtraIds.value()->size(), static_cast<size_t>(mOrigPromptLen));
        }
    }

    void setExcludeInputFromOutput(bool exclude)
    {
        mExcludeInputFromOutput = exclude;
    }

    [[nodiscard]] std::optional<executor::ContextPhaseParams> const& getContextPhaseParams() const noexcept
    {
        return mContextPhaseParams;
    }

    void setContextPhaseParams(executor::ContextPhaseParams contextPhaseParams)
    {
        mContextPhaseParams = std::move(contextPhaseParams);
    }

    [[nodiscard]] executor::DataTransceiverState const& getDataTransceiverState() const
    {
        TLLM_CHECK(mContextPhaseParams.has_value());
        return *static_cast<executor::DataTransceiverState const*>(mContextPhaseParams.value().getState());
    }

    [[nodiscard]] std::shared_ptr<ContextProgress> const& getContextProgress() const noexcept
    {
        return mContextProgress;
    }

    void setContextProgress(std::shared_ptr<ContextProgress> const& progress)
    {
        mContextProgress = progress;
    }

    [[nodiscard]] SizeType32 getNumTokens(SizeType32 beam) const
    {
        return mTokens.at(beam).size() - mNumPreDecodedTokens[beam];
    }

    [[nodiscard]] SizeType32 getNumReturnSequences() const
    {
        TLLM_LOG_WARNING(
            "mNumReturnSequences in the LlmRequest class is deprecated. Please use numReturnSequences in "
            "SamplingConfig directly.");
        return mNumReturnSequences;
    }

    [[nodiscard]] SizeType32 getNumSubRequests() const
    {
        return mSamplingConfig.beamWidth == 1 ? mSamplingConfig.numReturnSequences.value_or(1) : 1;
    }

    [[nodiscard]] std::vector<RequestPtr> const& getChildRequests() const
    {
        return mChildRequests;
    }

    [[nodiscard]] SizeType32 getMaxBeamNumTokens() const
    {
        SizeType32 maxTokens = 0;
        for (SizeType32 beam = 0; beam < mSamplingConfig.beamWidth; ++beam)
        {
            maxTokens = std::max(maxTokens, getNumTokens(beam));
        }
        return maxTokens;
    }

    [[nodiscard]] TokenIdType getToken(SizeType32 beam, SizeType32 pos) const
    {
        return mTokens.at(beam).at(pos);
    }

    [[nodiscard]] VecTokens const& getTokens(SizeType32 beam) const
    {
        return mTokens.at(beam);
    }

    [[nodiscard]] BeamTokens const& getTokens() const
    {
        return mTokens;
    }

    [[nodiscard]] VecUniqueTokens const& getUniqueTokens(SizeType32 beam) const
    {
        return mUniqueTokens.at(beam);
    }

    [[nodiscard]] BeamUniqueTokens const& getUniqueTokens() const
    {
        return mUniqueTokens;
    }

    [[nodiscard]] std::optional<std::shared_ptr<VecTokens>> const& getEncoderTokens() const
    {
        return mEncoderTokens;
    }

    [[nodiscard]] std::optional<std::shared_ptr<VecUniqueTokens>> const& getEncoderUniqueTokens() const
    {
        return mEncoderUniqueTokens;
    }

    [[nodiscard]] SizeType32 getEncoderInputLen() const
    {
        if (mEncoderInputFeatures.has_value())
        {
            return getEncoderInputFeatures()->getShape().d[0];
        }
        if (getEncoderTokens().has_value())
        {
            return getEncoderTokens().value()->size();
        }

        TLLM_THROW("GenericLlmRequest::getEncoderInputLen - Do not have encoder length!");
    }

    [[nodiscard]] SizeType32 getEncoderOutputLen() const
    {
        if (mEncoderOutputLength.has_value())
        {
            return mEncoderOutputLength.value();
        }

        return getEncoderInputLen();
    }

    [[nodiscard]] std::optional<std::shared_ptr<std::vector<SizeType32>>> getPositionIds() const
    {
        return mPositionIds;
    }

    [[nodiscard]] std::shared_ptr<VecTokens> const& getDraftTokens() const
    {
        return mDraftTokens;
    }

    [[nodiscard]] std::optional<TensorPtr> getDraftLogits() const
    {
        return mDraftLogits;
    }

    [[nodiscard]] bool hasDraftTokens() const
    {
        return mDraftTokens && !mDraftTokens->empty();
    }

    [[nodiscard]] SizeType32 getMaxNumGeneratedTokens() const
    {
        return getMaxBeamNumTokens() - mPromptLen;
    }

    [[nodiscard]] LlmRequestType getLlmRequestType() const
    {
        return mLlmRequestType;
    }

    SizeType32 addNewToken(TokenIdType token, SizeType32 beam)
    {
        mLastTokens[beam] = token;
        mTokens.at(beam).push_back(token);
        mUniqueTokens.at(beam).push_back({token, 0});
        return getNumTokens(beam);
    }

    void addNewTokens(VecTokens const& beamTokens)
    {
        assert(static_cast<size_t>(mSamplingConfig.beamWidth) == beamTokens.size());
        mLastTokens = beamTokens;
        for (std::size_t beam = 0; beam < beamTokens.size(); ++beam)
        {
            auto const outputId = beamTokens[beam];
            mTokens.at(beam).push_back(outputId);
            mUniqueTokens.at(beam).push_back({outputId, 0});
        }
    }

    void setNumPreDecodedTokens(SizeType32 num_tokens, SizeType32 beam)
    {
        mNumPreDecodedTokens[beam] = num_tokens;
    }

    void clearGeneratedTokens()
    {
        TLLM_LOG_DEBUG("emptying generated tokens for request %d with promptlen", mRequestId, mPromptLen);
        for (auto& beam : mTokens)
        {
            beam.resize(mPromptLen);
        }
    }

    void setGeneratedTokens(BeamTokens const& generatedBeamTokens)
    {
        TLLM_LOG_DEBUG("setting generated tokens for request %d", mRequestId);
        assert(generatedBeamTokens.size() == static_cast<size_t>(mSamplingConfig.beamWidth));

        for (size_t beamId = 0; beamId < generatedBeamTokens.size(); ++beamId)
        {
            auto& beamTokens = mTokens[beamId];
            beamTokens.resize(mPromptLen);
            beamTokens.insert(beamTokens.end(), generatedBeamTokens[beamId].begin(), generatedBeamTokens[beamId].end());
            auto& beamUniqueTokens = mUniqueTokens[beamId];
            beamUniqueTokens.resize(mPromptLen);
            for (auto const token : generatedBeamTokens[beamId])
            {
                beamUniqueTokens.push_back({token, 0});
            }
        }
    }

    void setNumReturnSequences(SizeType32 const& numReturnSequences)
    {
        TLLM_CHECK_WITH_INFO(!isChild(), "A child request cannot change numReturnSequences.");
        TLLM_CHECK_WITH_INFO(
            numReturnSequences > 0, "numReturnSequences should be a positive integer, got %d.", numReturnSequences);
        TLLM_CHECK_WITH_INFO(mChildRequests.size() <= static_cast<size_t>(numReturnSequences),
            "Cannot set numReturnSequences %d smaller than the number %ld of child requests that have already created.",
            numReturnSequences, mChildRequests.size());
        mSamplingConfig.numReturnSequences = numReturnSequences;
        mSequenceFinalVec->resize(numReturnSequences);
    }

    [[nodiscard]] bool constexpr isChild() const noexcept
    {
        return mSequenceIndex > 0;
    }

    [[nodiscard]] RequestIdType getParentRequestId() const
    {
        return mParentRequestId;
    }

    [[nodiscard]] VecTokens const& getLastTokens()
    {
        return mLastTokens;
    }

    [[nodiscard]] TokenIdType const& getLastTokens(SizeType32 beam)
    {
        return mLastTokens[beam];
    }

    void pause(SizeType32 maxInputLen)
    {
        if (mSamplingConfig.beamWidth > 1)
        {
            for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
            {
                auto& beamTokens = mTokens.at(beam);
                beamTokens.resize(mPromptLen);
                auto& beamUniqueTokens = mUniqueTokens.at(beam);
                beamUniqueTokens.resize(mPromptLen);
                if (returnLogProbs())
                {
                    mLogProbs.at(beam).clear();
                }
            }
        }
        else
        {
            SizeType32 newPromptLen = std::min(maxInputLen, mPromptLen + getMaxNumGeneratedTokens());
            for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
            {
                auto& beamTokens = mTokens.at(beam);
                beamTokens.resize(newPromptLen);
                auto& beamUniqueTokens = mUniqueTokens.at(beam);
                beamUniqueTokens.resize(newPromptLen);

                if (returnLogProbs())
                {
                    auto& logProb = mLogProbs.at(beam);
                    logProb.resize(newPromptLen - mPromptLen);
                }
            }
            mMaxNewTokens -= (newPromptLen - mPromptLen);
            mPromptLen = newPromptLen;
        }

        mState = mEncoderTokens.has_value() || mEncoderInputFeatures ? LlmRequestState::kENCODER_INIT
                                                                     : LlmRequestState::kCONTEXT_INIT;
        mContextCurrentPosition = 0;
        mContextChunkSize = mPromptLen;
        mSeqSlot.reset();
    }

    [[nodiscard]] SizeType32 getMaxSentTokenLen() const
    {
        return mMaxSentTokenLen;
    }

    void setMaxSentTokenLen(SizeType32 maxSentLength)
    {
        mMaxSentTokenLen = maxSentLength;
    }

    [[nodiscard]] std::optional<TensorPtr> getPromptEmbeddingTable() const
    {
        return mPromptEmbeddingTable;
    }

    [[nodiscard]] std::optional<SizeType32> getPromptVocabSize() const
    {
        return mPromptVocabSize;
    }

    [[nodiscard]] std::optional<TensorPtr> getMropeRotaryCosSin() const
    {
        return mMropeRotaryCosSin;
    }

    [[nodiscard]] std::optional<SizeType32> getMropePositionDeltas() const
    {
        return mMropePositionDeltas;
    }

    [[nodiscard]] std::optional<LoraTaskIdType> getLoraTaskId() const
    {
        return mLoraTaskId;
    }

    void setLoraTaskId(LoraTaskIdType taskId)
    {
        mLoraTaskId = taskId;
    }

    [[nodiscard]] std::optional<TensorPtr> getLoraWeights() const
    {
        return mLoraWeights;
    }

    void setLoraWeights(TensorPtr weights)
    {
        mLoraWeights = weights;
    }

    void clearLoraWeights()
    {
        mLoraWeights = std::nullopt;
    }

    [[nodiscard]] std::optional<TensorPtr> getLoraConfig() const
    {
        return mLoraConfig;
    }

    void setLoraConfig(TensorPtr config)
    {
        mLoraConfig = config;
    }

    void clearLoraConfig()
    {
        mLoraConfig = std::nullopt;
    }

    [[nodiscard]] std::optional<executor::LookaheadDecodingConfig> getLookaheadConfig() const
    {
        return mLookaheadConfig;
    }

    void setLookaheadConfig(executor::LookaheadDecodingConfig config)
    {
        mLookaheadConfig = config;
    }

    [[nodiscard]] std::optional<executor::KvCacheRetentionConfig> getKvCacheRetentionConfig() const
    {
        return mKvCacheRetentionConfig;
    }

    void setKvCacheRetentionConfig(executor::KvCacheRetentionConfig config)
    {
        mKvCacheRetentionConfig = config;
    }

    void clearLookaheadConfig()
    {
        mLookaheadConfig = std::nullopt;
    }

    [[nodiscard]] std::optional<executor::EagleConfig> getEagleConfig() const
    {
        return mEagleConfig;
    }

    void setEagleConfig(executor::EagleConfig config)
    {
        mEagleConfig = config;
    }

    [[nodiscard]] std::optional<executor::GuidedDecodingParams> getGuidedDecodingParams() const
    {
        return mGuidedDecodingParams;
    }

    void setGuidedDecodingParams(executor::GuidedDecodingParams guidedDecodingParams)
    {
        mGuidedDecodingParams = guidedDecodingParams;
    }

    [[nodiscard]] std::optional<TensorPtr> getEmbeddingBias() const
    {
        return mEmbeddingBias;
    }

    [[nodiscard]] std::optional<TensorPtr> getBadWordsList() const
    {
        return mBadWordsList;
    }

    [[nodiscard]] std::optional<TensorPtr> getStopWordsList() const
    {
        return mStopWordsList;
    }

    [[nodiscard]] bool returnLogProbs() const
    {
        return mSamplingConfig.outputLogProbs.has_value() ? mSamplingConfig.outputLogProbs->at(0) : false;
    }

    void setReturnLogProbs(bool returnLogProbs)
    {
        mSamplingConfig.outputLogProbs = {{returnLogProbs}};
        mSamplingConfig.cumLogProbs = {{returnLogProbs}};
    }

    [[nodiscard]] std::vector<VecLogProbs> const& getLogProbs() const
    {
        return mLogProbs;
    }

    [[nodiscard]] VecLogProbs const& getLogProbs(SizeType32 beam) const
    {
        return mLogProbs.at(beam);
    }

    void setLogProbs(VecLogProbs const& logProbs, SizeType32 beam)
    {
        mLogProbs.at(beam).resize(mPromptLen - mOrigPromptLen);
        mLogProbs.at(beam).insert(mLogProbs.at(beam).end(), logProbs.begin(), logProbs.end());
    }

    [[nodiscard]] VecLogProbs const& getCumLogProbs() const
    {
        return mCumLogProbs;
    }

    void setCumLogProb(float cumLogProb, SizeType32 beam)
    {
        mCumLogProbs.at(beam) = cumLogProb;
    }

    [[nodiscard]] SizeType32 getOrigPromptLen() const
    {
        return mOrigPromptLen;
    }

    [[nodiscard]] SizeType32 getPromptLen() const
    {
        return mPromptLen;
    }

    [[nodiscard]] SizeType32 getPrepopulatedPromptLen() const
    {
        return mPrepopulatedPromptLen;
    }

    void setPrepopulatedPromptLen(SizeType32 prepopulatedPromptLen, SizeType32 kvTokensPerBlock)
    {
        auto const promptLen = getPromptLen();
        TLLM_CHECK(prepopulatedPromptLen < promptLen);
        mPrepopulatedPromptLen = prepopulatedPromptLen;

        if (prepopulatedPromptLen > 0)
        {
            auto chunkSize = getContextChunkSize();
            if (prepopulatedPromptLen + chunkSize < promptLen)
            {
                auto const flooredEndPosition
                    = (prepopulatedPromptLen + chunkSize) / kvTokensPerBlock * kvTokensPerBlock;
                chunkSize = flooredEndPosition - prepopulatedPromptLen;
                TLLM_CHECK(chunkSize <= getContextChunkSize());
            }
            setContextCurrentPosition(prepopulatedPromptLen);
            setContextChunkSize(chunkSize);

            if (!isLastContextChunk())
            {
                TLLM_CHECK_WITH_INFO((getContextCurrentPosition() + getContextChunkSize()) % kvTokensPerBlock == 0,
                    "To prevent cache fragmentation, the context position after current chunk should be divisible "
                    "by the number of tokens per block, except for the last chunk.");
            }
        }
    }

    void setDraftTokens(std::shared_ptr<VecTokens> const& draftTokens)
    {
        mDraftTokens = draftTokens;
    }

    void setDraftLogits(std::optional<TensorPtr> const& draftLogits)
    {
        mDraftLogits = draftLogits;
    }

    [[nodiscard]] SizeType32 getNumDraftTokens() const
    {
        return mDraftTokens->size();
    }

    void discardDraftTokens(SizeType32 numTokensToDiscard)
    {
        TLLM_CHECK_WITH_INFO(
            numTokensToDiscard > 0, "Can only discard a positive amount of draft tokens, got %d", numTokensToDiscard);
        TLLM_CHECK_WITH_INFO(numTokensToDiscard <= getNumDraftTokens(),
            "Can't discard more draft tokens (%d) than exists (%d).", numTokensToDiscard, getNumDraftTokens());
        mDraftTokens->resize(getNumDraftTokens() - numTokensToDiscard);
    }

    void setNumTokensPerIteration(SizeType32 numTokensPerIteration)
    {
        mNumTokensPerIteration = std::max(1, numTokensPerIteration);
    }

    [[nodiscard]] SizeType32 getNumTokensPerIteration() const
    {
        return mNumTokensPerIteration;
    }

    void setReturnEncoderOutput(bool const returnEncoderOutput)
    {
        mReturnEncoderOutput = returnEncoderOutput;
    }

    [[nodiscard]] bool getReturnEncoderOutput() const
    {
        return mReturnEncoderOutput;
    }

    [[nodiscard]] TensorPtr const& getEncoderOutputHost() const
    {
        return mEncoderOutputHost;
    }

    [[nodiscard]] TensorPtr getEncoderInputFeatures() const
    {
        return mEncoderInputFeatures.value_or(nullptr);
    }

    void setEncoderOutputHost(TensorPtr encoderOutputHost)
    {
        mEncoderOutputHost = std::move(encoderOutputHost);
    }

    void setEncoderOutput(TensorPtr encoderOutput)
    {
        mEncoderOutput = std::move(encoderOutput);
    }

    void allocEncoderOutputHost(SizeType32 encoderHiddenSize, nvinfer1::DataType dataType)
    {
        mEncoderOutputHost = runtime::BufferManager::pinned(
            runtime::ITensor::makeShape({getEncoderOutputLen(), encoderHiddenSize}), dataType);
    }

    [[nodiscard]] TensorPtr const& getEncoderOutput() const noexcept
    {
        return mEncoderOutput;
    }

    [[nodiscard]] TensorPtr const& getEncoderHiddenStates() const noexcept
    {
        return mEncoderHiddenStates;
    }

    void allocEncoderOutput(runtime::BufferManager const& manager, nvinfer1::DataType dataType)
    {
        mEncoderOutput = std::move(manager.emptyTensor(runtime::MemoryType::kGPU, dataType));
    }

    void allocEncoderHiddenStates(runtime::BufferManager const& manager, nvinfer1::DataType dataType)
    {
        mEncoderHiddenStates = std::move(manager.emptyTensor(runtime::MemoryType::kGPU, dataType));
    }

    void freeEncoderOutputBuffers()
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

        TLLM_LOG_DEBUG(
            "Encoder output buffers use count: %u, %u", mEncoderOutput.use_count(), mEncoderHiddenStates.use_count());

        mEncoderOutput.reset();
        mEncoderHiddenStates.reset();

        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    }

    [[nodiscard]] TensorPtr getCrossAttentionMask() const
    {
        return mCrossAttentionMask.value_or(nullptr);
    }

    [[nodiscard]] TensorPtr getSkipCrossAttnBlocks() const
    {
        return mSkipCrossAttnBlocks.value_or(nullptr);
    }

    [[nodiscard]] bool constexpr getReturnPerfMetrics() const noexcept
    {
        return mReturnPerfMetrics;
    }

    void constexpr setReturnPerfMetrics(bool returnPerfMetrics) noexcept
    {
        mReturnPerfMetrics = returnPerfMetrics;
    }

    [[nodiscard]] executor::RequestPerfMetrics const& getPerfMetrics() const noexcept
    {
        return mPerfMetrics;
    }

    void setFirstScheduledTime(executor::RequestPerfMetrics::TimePoint const& time)
    {
        if (mPerfMetrics.timingMetrics.firstScheduledTime == executor::RequestPerfMetrics::TimePoint{})
        {
            mPerfMetrics.timingMetrics.firstScheduledTime = time;
        }
    }

    [[nodiscard]] bool constexpr isStreaming() const noexcept
    {
        return mIsStreaming;
    }

    void constexpr setStreaming(bool isStreaming) noexcept
    {
        mIsStreaming = isStreaming;
    }

    void setPriority(executor::PriorityType priority) noexcept
    {
        mPriority = priority;
    }

    void setReturnAllGeneratedTokens(bool const returnAllGeneratedTokens)
    {
        TLLM_CHECK_WITH_INFO(!mIsStreaming || mSamplingConfig.beamWidth == 1 || returnAllGeneratedTokens,
            "returnAllGeneratedTokens must be true if streaming AND beam search are used.");
        mReturnAllGeneratedTokens = returnAllGeneratedTokens;
    }

    [[nodiscard]] bool getReturnAllGeneratedTokens()
    {
        return mReturnAllGeneratedTokens;
    }

    void setAllottedTimeMs(MillisecondsType allottedTimeMs)
    {
        mAllottedTimeMs = allottedTimeMs;
    }

    void setReturnContextLogits(bool const returnContextLogits)
    {
        mReturnContextLogits = returnContextLogits;
    }

    [[nodiscard]] bool getReturnContextLogits() const
    {
        return mReturnContextLogits;
    }

    void setReturnGenerationLogits(bool const returnGenerationLogits)
    {
        TLLM_CHECK_WITH_INFO(!(mIsStreaming && mSamplingConfig.beamWidth > 1 && returnGenerationLogits),
            "returnGenerationLogits must be false if streaming AND beam search are used.");
        mReturnGenerationLogits = returnGenerationLogits;
    }

    [[nodiscard]] bool getReturnGenerationLogits() const
    {
        return mReturnGenerationLogits;
    }

    [[nodiscard]] TensorPtr const& getContextLogitsHost() const
    {
        return mContextLogitsHost;
    }

    void setContextLogitsHost(TensorPtr contextLogitsHost)
    {
        mContextLogitsHost = std::move(contextLogitsHost);
    }

    void allocContextLogitsHost(SizeType32 vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        mContextLogitsHost = runtime::BufferManager::pinnedPool(
            runtime::ITensor::makeShape({mPromptLen, vocabSizePadded}), logitsDataType);
    }

    [[nodiscard]] TensorPtr const& getGenerationLogitsHost() const
    {
        return mGenerationLogitsHost;
    }

    void setGenerationLogitsHost(TensorPtr generationLogitsHost)
    {
        mGenerationLogitsHost = std::move(generationLogitsHost);
    }

    void allocGenerationLogitsHost(SizeType32 vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        if (mIsStreaming)
        {
            mGenerationLogitsHost = runtime::BufferManager::pinnedPool(
                runtime::ITensor::makeShape({mMaxNewTokens, mSamplingConfig.beamWidth, vocabSizePadded}),
                logitsDataType);
        }
        else
        {
            mGenerationLogitsHost = runtime::BufferManager::pinnedPool(
                runtime::ITensor::makeShape({mSamplingConfig.beamWidth, mMaxNewTokens, vocabSizePadded}),
                logitsDataType);
        }
    }

    void allocTargetModelAcceptedTokenLogitsHost(SizeType32 vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        mGenerationLogitsHost = runtime::BufferManager::pinnedPool(
            runtime::ITensor::makeShape({1, getNumDraftTokens() + 1, vocabSizePadded}), logitsDataType);
    }

    [[nodiscard]] std::vector<TensorPtr> const& getGenerationLogitsFragments() const
    {
        return mGenerationLogitsFragments;
    }

    void addGenerationLogitsFragment(TensorPtr& genLogits)
    {
        mGenerationLogitsFragments.push_back(genLogits);
    }

    SizeType32 getGenerationLogitsFragmentsSize()
    {
        return mGenerationLogitsFragments.size();
    }

    void clearGenerationLogitsFragments()
    {
        mGenerationLogitsFragments.clear();
    }

    bool hasAdditionalOutputs()
    {
        return !mAdditionalContextOutputTensors.empty() || !mAdditionalGenerationOutputTensors.empty();
    }

    [[nodiscard]] TensorMap const& getAdditionalContextOutputs() const
    {
        return mAdditionalContextOutputTensors;
    }

    [[nodiscard]] TensorMap const& getAdditionalGenerationOutputs() const
    {
        return mAdditionalGenerationOutputTensors;
    }

    template <typename TypeFunc, typename ShapeFunc>
    void allocAdditionalOutputs(TypeFunc getTensorDataType, ShapeFunc getTensorShape)
    {
        for (auto& outputTensor : mAdditionalContextOutputTensors)
        {
            auto const& outputTensorName = outputTensor.first;
            auto const dataType = getTensorDataType(outputTensorName);
            auto shape = getTensorShape(outputTensorName);
            TLLM_CHECK_WITH_INFO(shape.d[0] == -1, "First dimension of additional output tensor '%s' must be dynamic",
                outputTensorName.c_str());
            shape.d[0] = mPromptLen;
            auto tensor = runtime::BufferManager::pinnedPool(shape, dataType);
            outputTensor.second = std::move(tensor);
        }
        for (auto& outputTensor : mAdditionalGenerationOutputTensors)
        {
            auto const& outputTensorName = outputTensor.first;
            auto const dataType = getTensorDataType(outputTensorName);
            auto shape = getTensorShape(outputTensorName);
            TLLM_CHECK_WITH_INFO(shape.d[0] == -1, "First dimension of additional output tensor '%s' must be dynamic",
                outputTensorName.c_str());
            shape.d[0] = mMaxNewTokens - 1;
            shape = runtime::ITensor::unsqueeze(shape, 0);
            shape.d[0] = mSamplingConfig.beamWidth;
            auto tensor = runtime::BufferManager::pinnedPool(shape, dataType);
            outputTensor.second = std::move(tensor);
        }
    }

    [[nodiscard]] bool hasReachedState(LlmRequestState state) const noexcept
    {
        return mState >= state;
    }

    [[nodiscard]] bool isEncoderInitState() const noexcept
    {
        return mState == LlmRequestState::kENCODER_INIT;
    }

    [[nodiscard]] bool isContextInitState() const noexcept
    {
        return mState == LlmRequestState::kCONTEXT_INIT || mState == LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS;
    }

    [[nodiscard]] bool isGenerationInProgressState() const noexcept
    {
        return mState == LlmRequestState::kGENERATION_IN_PROGRESS || mState == LlmRequestState::kGENERATION_TO_COMPLETE
            || mState == LlmRequestState::KDISAGG_GENERATION_TRANS_COMPLETE;
    }

    [[nodiscard]] bool isGenerationCompleteState() const noexcept
    {
        return mState == LlmRequestState::kGENERATION_COMPLETE;
    }

    [[nodiscard]] bool isDisaggGenerationInitState() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_GENERATION_INIT;
    }

    [[nodiscard]] bool isDisaggGenerationTransmissionComplete() const noexcept
    {
        return mState == LlmRequestState::KDISAGG_GENERATION_TRANS_COMPLETE;
    }

    [[nodiscard]] bool isDisaggGenerationTransmissionInProgress() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_GENERATION_TRANS_IN_PROGRESS;
    }

    [[nodiscard]] bool isDisaggContextTransmissionState() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS
            || mState == LlmRequestState::kDISAGG_CONTEXT_INIT_AND_TRANS;
    }

    [[nodiscard]] bool isDisaggContextCompleteState() const noexcept
    {
        return mState == LlmRequestState::kDISAGG_CONTEXT_COMPLETE;
    }

    [[nodiscard]] bool isFullContextRequest() const noexcept
    {
        return (isContextInitState() || isDisaggGenerationInitState() || isDisaggGenerationTransmissionComplete())
            && !mContextChunkSize;
    }

    [[nodiscard]] bool isContextOnlyRequest() const noexcept
    {
        return mLlmRequestType == LlmRequestType::LLMREQUEST_TYPE_CONTEXT_ONLY;
    }

    [[nodiscard]] bool isGenerationOnlyRequest() const noexcept
    {
        return mLlmRequestType == LlmRequestType::LLMREQUEST_TYPE_GENERATION_ONLY;
    }

    void setContextCurrentPosition(SizeType32 contextCurrentPosition)
    {
        mContextCurrentPosition = contextCurrentPosition;
    }

    [[nodiscard]] SizeType32 getContextCurrentPosition() const noexcept
    {
        return mContextCurrentPosition;
    }

    [[nodiscard]] SizeType32 getContextRemainingLength() const noexcept
    {
        return mPromptLen - getContextCurrentPosition();
    }

    [[nodiscard]] SizeType32 getContextChunkSize() const
    {
        TLLM_CHECK_WITH_INFO(
            isContextInitState() || isDisaggGenerationInitState() || isDisaggGenerationTransmissionComplete(),
            "getContextChunkSize is only possible during the context phase.");
        return mContextChunkSize;
    }

    void setContextChunkSize(SizeType32 size)
    {
        TLLM_CHECK_WITH_INFO(isContextInitState(), "setContextChunkSize is only possible during the context phase.");
        TLLM_CHECK_WITH_INFO(size >= 0, "The chunk size of context (%d) can't be negative.", size);
        mContextChunkSize = std::min(size, getContextRemainingLength());
    }

    [[nodiscard]] bool isLastContextChunk() const noexcept
    {
        return isDisaggGenerationInitState() || isDisaggGenerationTransmissionComplete()
            || getContextCurrentPosition() + getContextChunkSize() == mPromptLen;
    }

    [[nodiscard]] bool isFirstContextChunk() const noexcept
    {
        return getContextCurrentPosition() == 0;
    }

    void moveToNextContextChunk()
    {
        TLLM_CHECK_WITH_INFO(isContextInitState(), "Chunking is only possible during the context phase.");
        mContextCurrentPosition += getContextChunkSize();
        setContextChunkSize(0);
    }

    [[nodiscard]] executor::PriorityType priority() const noexcept
    {
        return mPriority;
    }

    SizeType32 getDecodingIter()
    {
        return mDecodingIter;
    }

    void advanceDecodingIter()
    {
        mDecodingIter++;
    }

    [[nodiscard]] float getAvgDecodedTokensPerIter() const noexcept
    {
        if (mDecodingIter == 0)
        {
            return 0.F;
        }
        return static_cast<float>(getMaxNumGeneratedTokens()) / mDecodingIter;
    }

    [[nodiscard]] bool isFinished() const noexcept
    {
        return isGenerationCompleteState() || mState == LlmRequestState::kDISAGG_CONTEXT_TRANS_IN_PROGRESS;
    }

    [[nodiscard]] bool isTimedOut() const
    {
        if (!mAllottedTimeMs.has_value())
        {
            return false;
        }
        auto const currentTime = std::chrono::steady_clock::now();
        auto const elapsed = (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - mStartTime));
        TLLM_LOG_DEBUG("Checked timeOut for request %d with allotted Time %d after time %d and got %d", mRequestId,
            mAllottedTimeMs->count(), elapsed.count(), (elapsed >= mAllottedTimeMs));

        return elapsed >= *mAllottedTimeMs;
    }

    std::optional<executor::Response> createResponse(bool useFastLogits = false, int32_t mpiWorldRank = 0)
    {
        TLLM_CHECK(!isDisaggContextCompleteState());
        if (!(isFinished() || (mIsStreaming && mState == LlmRequestState::kGENERATION_IN_PROGRESS)))
        {
            return std::nullopt;
        }

        TLLM_LOG_DEBUG("Creating response for request %lu", mRequestId);

        executor::Result result;
        result.sequenceIndex = mSequenceIndex;

        result.isSequenceFinal = isFinished();
        mSequenceFinalVec->at(mSequenceIndex) = result.isSequenceFinal;

        result.isFinal = std::all_of(
            mSequenceFinalVec->begin(), mSequenceFinalVec->end(), [](bool isSequenceFinal) { return isSequenceFinal; });

        auto const maxNbTokens = getMaxBeamNumTokens();

        if (isDisaggContextTransmissionState() && isContextOnlyRequest())
        {
            auto const reqBeamWidth = mSamplingConfig.beamWidth;
            std::vector<TokenIdType> firstGenTokens;
            for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
            {
                firstGenTokens.push_back(getTokens().at(beam).back());
            }
            result.contextPhaseParams = executor::ContextPhaseParams{
                std::move(firstGenTokens), mRequestId, mContextPhaseParams.value().releaseState()};
        }

        auto const calculateNbTokensOut = [this](SizeType32 maxNbTokens)
        {
            if (!mIsStreaming)
            {
                return maxNbTokens - (mExcludeInputFromOutput ? getOrigPromptLen() : 0);
            }
            return mReturnAllGeneratedTokens ? maxNbTokens - getOrigPromptLen() : maxNbTokens - getMaxSentTokenLen();
        };

        auto const maxNbTokensOut = calculateNbTokensOut(maxNbTokens);

        auto const nbBeams = mSamplingConfig.getNumReturnBeams();

        result.outputTokenIds.resize(nbBeams);

        auto const startTokenPos = maxNbTokens - maxNbTokensOut;

        auto const shouldSendResponse = isFinished() || (mIsStreaming && maxNbTokens > getMaxSentTokenLen());

        if (!shouldSendResponse)
        {
            return std::nullopt;
        }

        for (SizeType32 beam = 0; beam < nbBeams; ++beam)
        {
            auto const& tokens = getTokens(beam);
            auto const nbTokensOut = calculateNbTokensOut(tokens.size());

            if (nbTokensOut > 0)
            {
                auto const first = tokens.data() + startTokenPos;
                result.outputTokenIds.at(beam).assign(first, first + nbTokensOut);
            }
        }

        auto sliceBeams = [&nbBeams](auto beams)
        { return std::vector<typename decltype(beams)::value_type>(beams.begin(), beams.begin() + nbBeams); };

        if (returnLogProbs())
        {
            result.cumLogProbs = sliceBeams(getCumLogProbs());
            result.logProbs = sliceBeams(getLogProbs());
        }

        if (getReturnContextLogits())
        {
            result.contextLogits = executor::detail::ofITensor(getContextLogitsHost());
        }

        if (getReturnGenerationLogits())
        {
            bool hasDraftTokens = mDraftTokens && !mDraftTokens->empty();
            if (isStreaming() && !hasDraftTokens)
            {
                auto startGenTokenPos = startTokenPos - getOrigPromptLen();
                TensorPtr generationLogitsHostCurrentStep
                    = runtime::ITensor::slice(getGenerationLogitsHost(), startGenTokenPos, maxNbTokensOut);
                result.generationLogits = executor::detail::ofITensor(generationLogitsHostCurrentStep);
            }
            else if (useFastLogits)
            {
                result.specDecFastLogitsInfo = executor::SpeculativeDecodingFastLogitsInfo{mRequestId, mpiWorldRank};
            }
            else
            {
                result.generationLogits
                    = executor::detail::ofITensor(runtime::ITensor::slice(getGenerationLogitsHost(), 0, nbBeams));
            }
        }

        if (getReturnEncoderOutput())
        {
            result.encoderOutput = executor::detail::ofITensor(getEncoderOutputHost());
        }

        if (getReturnPerfMetrics())
        {
            result.requestPerfMetrics = mPerfMetrics;
        }

        result.finishReasons = sliceBeams(mFinishReasons);
        result.decodingIter = mDecodingIter;

        if (hasAdditionalOutputs())
        {
            std::string prefix = "context_";
            for (auto const& outputTensorMap : {mAdditionalContextOutputTensors, mAdditionalGenerationOutputTensors})
            {
                for (auto const& outputTensor : outputTensorMap)
                {
                    TLLM_LOG_DEBUG("Adding tensor %s with shape %s to result.", outputTensor.first.c_str(),
                        runtime::ITensor::toString(outputTensor.second->getShape()).c_str());
                    result.additionalOutputs.emplace_back(
                        prefix + outputTensor.first, executor::detail::ofITensor(outputTensor.second));
                }
                prefix = "generation_";
            }
        }

        setMaxSentTokenLen(maxNbTokens);

        auto requestId = isChild() ? mParentRequestId : mRequestId;
        auto response = executor::Response(requestId, std::move(result), mClientId);

        return response;
    }

    void setFinishedReason(executor::FinishReason reason, SizeType32 beam)
    {
        mFinishReasons.at(beam) = reason;
    }

    void setDecodingIter(SizeType32 iter)
    {
        mDecodingIter = iter;
    }

    void setKvCacheTransferStart(std::chrono::time_point<std::chrono::steady_clock> const& time)
    {
        mPerfMetrics.timingMetrics.kvCacheTransferStart = time;
    }

    void setKvCacheTransferEnd(std::chrono::time_point<std::chrono::steady_clock> const& time)
    {
        mPerfMetrics.timingMetrics.kvCacheTransferEnd = time;
    }

    [[nodiscard]] double getKvCacheTransferTimeMS() const
    {
        return std::max(0.0,
            std::chrono::duration<double, std::milli>(
                mPerfMetrics.timingMetrics.kvCacheTransferEnd - mPerfMetrics.timingMetrics.kvCacheTransferStart)
                .count());
    }

    void updateAllocTotalBlocksPerRequest(SizeType32 allocTotalBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numTotalAllocatedBlocks += allocTotalBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getAllocTotalBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numTotalAllocatedBlocks;
    }

    void updateAllocNewBlocksPerRequest(SizeType32 allocNewBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numNewAllocatedBlocks += allocNewBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getAllocNewBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numNewAllocatedBlocks;
    }

    void updateReusedBlocksPerRequest(SizeType32 reusedBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numReusedBlocks += reusedBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getReusedBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numReusedBlocks;
    }

    void finishByReason(executor::FinishReason finishReason)
    {
        if (finishReason == executor::FinishReason::kTIMED_OUT)
        {
            TLLM_LOG_DEBUG("Request %d finished by timeout after %f sec", mRequestId,
                std::chrono::duration<float>(std::chrono::steady_clock::now() - mStartTime).count());
        }
        if (finishReason == executor::FinishReason::kCANCELLED)
        {
            TLLM_LOG_DEBUG("Request %d finished by cancel", mRequestId);
        }

        for (int beam = 0; beam < mSamplingConfig.beamWidth; ++beam)
        {
            if (mFinishReasons.at(beam) == executor::FinishReason::kNOT_FINISHED)
            {
                setFinishedReason(finishReason, beam);
            }
        }
        mState = LlmRequestState::kGENERATION_COMPLETE;
    }

    void updateMissedBlocksPerRequest(SizeType32 missedBlocksPerRequest)
    {
        mPerfMetrics.kvCacheMetrics.numMissedBlocks += missedBlocksPerRequest;
    }

    [[nodiscard]] SizeType32 getMissedBlocksPerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numMissedBlocks;
    }

    [[nodiscard]] float getKVCacheHitRatePerRequest() const
    {
        return mPerfMetrics.kvCacheMetrics.numReusedBlocks == 0
            ? 0
            : static_cast<float>(mPerfMetrics.kvCacheMetrics.numReusedBlocks)
                / (static_cast<float>(
                    mPerfMetrics.kvCacheMetrics.numReusedBlocks + mPerfMetrics.kvCacheMetrics.numMissedBlocks));
    }

    void updatePerfMetrics(executor::IterationType iter)
    {
        if (!mPerfMetrics.firstIter)
        {
            mPerfMetrics.firstIter = iter;
            mPerfMetrics.timingMetrics.firstTokenTime = std::chrono::steady_clock::now();
        }

        mPerfMetrics.iter = iter;

        if (isFinished())
        {
            mPerfMetrics.lastIter = iter;
            mPerfMetrics.timingMetrics.lastTokenTime = std::chrono::steady_clock::now();
        }
    }

    RequestIdType mRequestId;
    SizeType32 mPromptLen;
    SizeType32 mMaxNewTokens;
    runtime::SamplingConfig mSamplingConfig;
    LlmRequestState mState;
    std::optional<TokenIdType> mEndId;
    std::optional<TokenIdType> mPadId;
    std::optional<SizeType32> mSeqSlot;
    std::optional<LogitsPostProcessor> mLogitsPostProcessor;
    bool mApplyLogitsPostProcessorBatched;
    std::optional<RequestIdType> mClientId;
    SizeType32 mMaskPosition{0};

protected:
    bool mIsStreaming;

    VecTokens mLastTokens;
    BeamTokens mTokens;
    SizeType32 mOrigPromptLen;
    std::vector<SizeType32> mNumPreDecodedTokens;
    SizeType32 mPrepopulatedPromptLen{0};
    SizeType32 mMaxSentTokenLen;

    std::optional<TensorPtr> mEmbeddingBias;
    std::optional<TensorPtr> mBadWordsList;
    std::optional<TensorPtr> mStopWordsList;

    std::optional<std::shared_ptr<std::vector<SizeType32>>> mPositionIds;

    std::optional<TensorPtr> mPromptEmbeddingTable;
    std::optional<SizeType32> mPromptVocabSize;
    std::optional<TensorPtr> mMropeRotaryCosSin;
    std::optional<SizeType32> mMropePositionDeltas;

    std::optional<LoraTaskIdType> mLoraTaskId;
    std::optional<TensorPtr> mLoraWeights;
    std::optional<TensorPtr> mLoraConfig;
    std::optional<executor::LookaheadDecodingConfig> mLookaheadConfig;

    std::optional<executor::KvCacheRetentionConfig> mKvCacheRetentionConfig;
    SizeType32 mContextChunkSize{0};
    SizeType32 mContextCurrentPosition{0};

    std::vector<VecLogProbs> mLogProbs;
    VecLogProbs mCumLogProbs;
    std::shared_ptr<VecTokens> mDraftTokens;
    std::optional<TensorPtr> mDraftLogits;
    SizeType32 mNumTokensPerIteration;

    bool mReturnAllGeneratedTokens;
    bool mReturnContextLogits;
    bool mReturnGenerationLogits;
    bool mReturnLogProbs;
    TensorPtr mContextLogitsHost;
    TensorPtr mGenerationLogitsHost;
    std::vector<TensorPtr> mGenerationLogitsFragments;

    bool mExcludeInputFromOutput;

    std::optional<std::shared_ptr<VecTokens>> mEncoderTokens;
    bool mReturnEncoderOutput;
    TensorPtr mEncoderOutput;
    TensorPtr mEncoderHiddenStates;
    TensorPtr mEncoderOutputHost;

    SizeType32 mDecodingIter;
    executor::PriorityType mPriority;
    std::vector<executor::FinishReason> mFinishReasons;
    std::optional<TensorPtr> mEncoderInputFeatures;
    std::optional<SizeType32>
        mEncoderOutputLength;
    std::optional<TensorPtr> mCrossAttentionMask;
    LlmRequestType mLlmRequestType;
    std::optional<executor::ContextPhaseParams> mContextPhaseParams;
    std::shared_ptr<ContextProgress> mContextProgress;

    std::optional<std::shared_ptr<VecTokenExtraIds>> mInputTokenExtraIds;
    BeamUniqueTokens mUniqueTokens;
    std::optional<std::shared_ptr<VecUniqueTokens>> mEncoderUniqueTokens;

    SizeType32 mNumReturnSequences;
    std::optional<executor::EagleConfig> mEagleConfig;
    SizeType32 mSequenceIndex;
    std::vector<RequestPtr> mChildRequests;
    RequestIdType mParentRequestId;
    std::shared_ptr<std::vector<bool>> mSequenceFinalVec;

    std::optional<TensorPtr> mSkipCrossAttnBlocks;

    bool mReturnPerfMetrics;
    executor::RequestPerfMetrics mPerfMetrics;

    std::optional<executor::GuidedDecodingParams> mGuidedDecodingParams;

    std::chrono::steady_clock::time_point mStartTime;
    std::optional<MillisecondsType> mAllottedTimeMs;

    TensorMap mAdditionalContextOutputTensors;
    TensorMap mAdditionalGenerationOutputTensors;

private:
    void initialize(VecTokens const& inputTokens, bool outputLogProbs)
    {
        mTokens = BeamTokens(mSamplingConfig.beamWidth, inputTokens);
        mLastTokens = VecTokens(mSamplingConfig.beamWidth);

        VecUniqueTokens uniqueTokens{inputTokens.size()};
        if (mInputTokenExtraIds.has_value() && mInputTokenExtraIds.value())
        {
            if (mInputTokenExtraIds.value()->size() != inputTokens.size())
            {
                TLLM_THROW("inputTokenExtraIds vector size (%lu) must be the same as input token vector size (%lu).",
                    mInputTokenExtraIds.value()->size(), inputTokens.size());
            }
            std::transform(inputTokens.cbegin(), inputTokens.cend(), mInputTokenExtraIds.value()->cbegin(),
                uniqueTokens.begin(),
                [](auto const inputToken, auto const tokenExtraId) {
                    return UniqueToken{inputToken, tokenExtraId};
                });
        }
        else
        {
            std::transform(inputTokens.cbegin(), inputTokens.cend(), uniqueTokens.begin(),
                [](auto const inputToken) {
                    return UniqueToken{inputToken, 0};
                });
        }
        mUniqueTokens = BeamUniqueTokens(mSamplingConfig.beamWidth, uniqueTokens);

        if (mEncoderTokens.has_value() && mEncoderTokens.value())
        {
            auto const& encoderTokens = *(mEncoderTokens.value());
            auto encoderUniqueTokens = std::make_shared<VecUniqueTokens>(encoderTokens.size());
            std::transform(encoderTokens.cbegin(), encoderTokens.cend(), encoderUniqueTokens->begin(),
                [](auto const encoderToken) {
                    return UniqueToken{encoderToken, 0};
                });
            mEncoderUniqueTokens = encoderUniqueTokens;
        }

        if ((mPromptEmbeddingTable.has_value() && !mPromptVocabSize.has_value())
            || (!mPromptEmbeddingTable.has_value() && mPromptVocabSize.has_value()))
        {
            std::string errStr
                = "Prompt embedding table and prompt vocab size tensors must both be provided for requests with "
                  "prompt "
                  "tuning enabled.";
            TLLM_THROW(errStr);
        }

        if (mDraftLogits.has_value() && mDraftTokens->empty())
        {
            TLLM_THROW("Draft tokens must be specified when draft logits are given.");
        }

        setReturnLogProbs(outputLogProbs);

        if (mNumReturnSequences > 1)
        {
            if (!mSamplingConfig.numReturnSequences)
            {
                TLLM_LOG_WARNING(
                    "In the Executor class, mNumReturnSequences is deprecated. Please set numReturnSequences in "
                    "SamplingConfig directly.");
            }
            else if (mSamplingConfig.numReturnSequences
                && mSamplingConfig.numReturnSequences.value() != mNumReturnSequences)
            {
                TLLM_THROW(
                    "In the Executor class, both mSamplingConfig.numReturnSequences (%d) and mNumReturnSequences (%d) "
                    "are provided but unmatched. Please use numReturnSequences in SamplingConfig directly.",
                    mSamplingConfig.numReturnSequences.value(), mNumReturnSequences);
            }
            mSamplingConfig.numReturnSequences = mNumReturnSequences;
        }

        if (!isChild())
        {
            mSequenceFinalVec = std::make_shared<std::vector<bool>>(getNumSubRequests(), false);
        }

        if (mReturnPerfMetrics)
        {
            mPerfMetrics.timingMetrics.arrivalTime = std::chrono::steady_clock::now();
        }
        mStartTime = std::chrono::steady_clock::now();
    }

    TensorPtr createListTensor(std::list<VecTokens> const& wordsList)
    {
        std::vector<SizeType32> offsets;
        VecTokens words;
        SizeType32 offsetCnt = 0;
        for (auto const& tokens : wordsList)
        {
            offsetCnt += tokens.size();
            offsets.push_back(offsetCnt);
            words.insert(words.end(), tokens.begin(), tokens.end());
        }
        offsets.resize(words.size(), -1);

        auto const numWords = static_cast<SizeType32>(words.size());
        auto const shape = runtime::ITensor::makeShape({2, numWords});
        auto tensor = runtime::BufferManager::pinnedPool(shape, nvinfer1::DataType::kINT32);
        auto* data = runtime::bufferCast<int32_t>(*tensor);
        std::memcpy(data, words.data(), numWords * sizeof(int32_t));
        std::memcpy(data + numWords, offsets.data(), numWords * sizeof(int32_t));

        tensor->unsqueeze(0);

        return tensor;
    }
};

class LlmRequest : public GenericLlmRequest<runtime::ITensor::SharedPtr>
{
    friend class LlmRequestBindings;

public:
    using Base = GenericLlmRequest<runtime::ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;
    using SizeType32 = Base::SizeType32;
    using TokenIdType = Base::TokenIdType;
    using RequestIdType = Base::RequestIdType;
    using VecLogProbs = Base::VecLogProbs;
    using BeamTokens = Base::BeamTokens;
    using VecTokens = Base::VecTokens;
    using LoraTaskIdType = Base::LoraTaskIdType;
    using TokenExtraIdType = Base::TokenExtraIdType;
    using VecTokenExtraIds = Base::VecTokenExtraIds;

    LlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, std::shared_ptr<VecTokens> inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<TensorPtr> mropeRotaryCosSin = std::nullopt,
        std::optional<SizeType32> mropePositionDeltas = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<executor::KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<std::shared_ptr<VecTokens>> const& draftTokens = std::nullopt,
        std::optional<TensorPtr> draftLogits = std::nullopt, bool excludeInputFromOutput = false,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false,
        std::optional<std::shared_ptr<VecTokens>> encoderInputTokens = std::nullopt, bool returnEncoderOutput = false,
        std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority,
        std::optional<TensorPtr> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<TensorPtr> crossAttentionMask = std::nullopt,
        LlmRequestType llmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<std::shared_ptr<VecTokenExtraIds>> inputTokenExtraIds = std::nullopt,
        SizeType32 numReturnSequences = 1, std::optional<executor::EagleConfig> eagleConfig = std::nullopt,
        std::optional<TensorPtr> skipCrossAttnBlocks = std::nullopt, bool returnPerfMetrics = false,
        std::optional<executor::GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt)
        : Base(requestId, maxNewTokens, std::move(inputTokens), samplingConfig, isStreaming, endId, padId,
            std::move(embeddingBias), std::move(badWordsList), std::move(stopWordsList), std::move(positionIds),
            std::move(promptEmbeddingTable), promptVocabSize, std::move(mropeRotaryCosSin), mropePositionDeltas,
            loraTaskId, std::move(loraWeights), std::move(loraConfig), std::move(lookaheadConfig),
            std::move(kvCacheRetentionConfig), returnLogProbs, returnContextLogits, returnGenerationLogits,
            std::move(draftTokens), std::move(draftLogits), excludeInputFromOutput, std::move(logitsPostProcessor),
            applyLogitsPostProcessorBatched, std::move(encoderInputTokens), returnEncoderOutput, clientId, priority,
            std::move(encoderInputFeatures), std::move(encoderOutputLength), std::move(crossAttentionMask),
            llmRequestType, std::move(inputTokenExtraIds), numReturnSequences, std::move(eagleConfig),
            std::move(skipCrossAttnBlocks), returnPerfMetrics, std::move(guidedDecodingParams), allottedTimeMs)
    {
    }

    LlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, std::vector<TokenIdType> inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::vector<SizeType32>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<TensorPtr> mropeRotaryCosSin = std::nullopt,
        std::optional<SizeType32> mropePositionDeltas = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt,
        std::optional<executor::KvCacheRetentionConfig> kvCacheRetentionConfig = std::nullopt,
        bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<VecTokens> draftTokens = std::nullopt, std::optional<TensorPtr> draftLogits = std::nullopt,
        bool excludeInputFromOutput = false, std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false, std::optional<VecTokens> encoderInputTokens = std::nullopt,
        bool returnEncoderOutput = false, std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority,
        std::optional<TensorPtr> encoderInputFeatures = std::nullopt,
        std::optional<SizeType32> encoderOutputLength = std::nullopt,
        std::optional<TensorPtr> crossAttentionMask = std::nullopt,
        LlmRequestType llmRequestType = LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        std::optional<VecTokenExtraIds> inputTokenExtraIds = std::nullopt, SizeType32 numReturnSequences = 1,
        std::optional<executor::EagleConfig> eagleConfig = std::nullopt,
        std::optional<TensorPtr> skipCrossAttnBlocks = std::nullopt, bool returnPerfMetrics = false,
        std::optional<executor::GuidedDecodingParams> guidedDecodingParams = std::nullopt,
        std::optional<MillisecondsType> allottedTimeMs = std::nullopt)
        : Base(requestId, maxNewTokens, std::make_shared<std::vector<TokenIdType>>(std::move(inputTokens)),
            samplingConfig, isStreaming, endId, padId, std::move(embeddingBias), std::move(badWordsList),
            std::move(stopWordsList),
            positionIds.has_value() ? std::make_shared<std::vector<SizeType32>>(std::move(positionIds.value()))
                                    : std::optional<std::shared_ptr<std::vector<SizeType32>>>(std::nullopt),
            std::move(promptEmbeddingTable), promptVocabSize, std::move(mropeRotaryCosSin), mropePositionDeltas,
            loraTaskId, std::move(loraWeights), std::move(loraConfig), lookaheadConfig,
            std::move(kvCacheRetentionConfig), returnLogProbs, returnContextLogits, returnGenerationLogits,
            draftTokens.has_value() ? std::make_shared<VecTokens>(std::move(draftTokens.value()))
                                    : std::make_shared<VecTokens>(),
            std::move(draftLogits), excludeInputFromOutput, std::move(logitsPostProcessor),
            applyLogitsPostProcessorBatched,
            encoderInputTokens ? std::make_optional(std::make_shared<VecTokens>(std::move(*encoderInputTokens)))
                               : std::optional<std::shared_ptr<VecTokens>>(std::nullopt),
            returnEncoderOutput, clientId, priority, std::move(encoderInputFeatures), encoderOutputLength,
            std::move(crossAttentionMask), llmRequestType,
            inputTokenExtraIds ? std::make_optional(std::make_shared<VecTokenExtraIds>(std::move(*inputTokenExtraIds)))
                               : std::optional<std::shared_ptr<VecTokenExtraIds>>(std::nullopt),
            numReturnSequences, std::move(eagleConfig), skipCrossAttnBlocks, returnPerfMetrics,
            std::move(guidedDecodingParams), allottedTimeMs)
    {
    }

    LlmRequest(RequestIdType requestId, SizeType32 maxNewTokens, VecTokens const& inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType32> endId = std::nullopt,
        std::optional<SizeType32> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<std::shared_ptr<std::vector<SizeType32>>> positionIds = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType32> promptVocabSize = std::nullopt,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt,
        std::optional<executor::LookaheadDecodingConfig> lookaheadConfig = std::nullopt, bool returnLogProbs = false,
        bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<VecTokens> draftTokens = std::nullopt, std::optional<TensorPtr> draftLogits = std::nullopt,
        bool excludeInputFromOutput = false, std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false, std::optional<VecTokens> encoderInputTokens = std::nullopt,
        bool returnEncoderOutput = false, std::optional<RequestIdType> clientId = std::nullopt,
        executor::PriorityType priority = executor::Request::kDefaultPriority, SizeType32 numReturnSequences = 1)
        : Base(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming, endId, padId,
            std::move(embeddingBias), std::move(badWordsList), std::move(stopWordsList), std::move(positionIds),
            std::move(promptEmbeddingTable), promptVocabSize, loraTaskId, std::move(loraWeights), std::move(loraConfig),
            lookaheadConfig, returnLogProbs, returnContextLogits, returnGenerationLogits, std::move(draftTokens),
            std::move(draftLogits), excludeInputFromOutput, std::move(logitsPostProcessor),
            applyLogitsPostProcessorBatched, std::move(encoderInputTokens), returnEncoderOutput, clientId, priority,
            numReturnSequences)
    {
    }

    LlmRequest(RequestIdType requestId, executor::Request const& request,
        std::optional<Base::LogitsPostProcessor> logitsPostProcessor = std::nullopt,
        bool applyLogitsPostProcessorBatched = false)
        : Base(requestId, request)
    {
        mLogitsPostProcessor = std::move(logitsPostProcessor);
        mApplyLogitsPostProcessorBatched = applyLogitsPostProcessorBatched;
        mLookaheadConfig = request.getLookaheadConfig();
        mKvCacheRetentionConfig = request.getKvCacheRetentionConfig();
    }

    std::shared_ptr<LlmRequest> createChildRequest(RequestIdType requestId)
    {
        TLLM_CHECK_WITH_INFO(!isChild(), "A child request cannot create its own child.");
        TLLM_CHECK_WITH_INFO(mChildRequests.size() + 1 < static_cast<size_t>(getNumSubRequests()),
            "Cannot create child requests more than the number of return sequences (%d)", getNumSubRequests());
        auto childReq = std::make_shared<LlmRequest>(*this);
        childReq->mRequestId = requestId;
        childReq->mSequenceIndex = mChildRequests.size() + 1;
        childReq->mParentRequestId = this->mRequestId;
        childReq->mSequenceFinalVec = this->mSequenceFinalVec;
        childReq->mSeqSlot.reset();

        using RandomSeedType = sugesstify::executor::RandomSeedType;
        if (childReq->mSamplingConfig.randomSeed.has_value())
        {
            childReq->mSamplingConfig.randomSeed->at(0) += static_cast<RandomSeedType>(childReq->mSequenceIndex);
        }
        else
        {
            RandomSeedType defaultSeed{0};
            mSamplingConfig.randomSeed = std::vector<RandomSeedType>(1, defaultSeed);
            childReq->mSamplingConfig.randomSeed
                = std::vector<RandomSeedType>(1, defaultSeed + static_cast<RandomSeedType>(childReq->mSequenceIndex));
        }

        mChildRequests.push_back(childReq);
        return childReq;
    }

    void movePromptEmbeddingTableToGpu(runtime::BufferManager const& manager)
    {
        if (!mPromptEmbeddingTable.has_value()
            || mPromptEmbeddingTable.value()->getMemoryType() == runtime::MemoryType::kGPU)
        {
            return;
        }

        TensorPtr gpuPromptEmbeddingTable = manager.copyFrom(*mPromptEmbeddingTable.value(), runtime::MemoryType::kGPU);
        mPromptEmbeddingTable = gpuPromptEmbeddingTable;
    }

    void moveLoraWeightsToGpu(runtime::BufferManager const& manager)
    {
        if (!mLoraWeights.has_value() || mLoraWeights.value()->getMemoryType() == runtime::MemoryType::kGPU)
        {
            return;
        }
        TensorPtr gpuLoraWeights = manager.copyFrom(*mLoraWeights.value(), runtime::MemoryType::kGPU);
        mLoraWeights = gpuLoraWeights;
    }
};

}
