
#pragma once

#include "bufferManager.h"
#include "cudaEvent.h"
#include "cudaStream.h"
#include "generationOutput.h"
#include "gptDecoder.h"
#include "iGptDecoderBatched.h"
#include "iTensor.h"

#include <memory>
#include <vector>

namespace suggestify::batch_manager
{
class LlmRequest;
}

namespace suggestify::runtime
{

class GptDecoderBatched : public IGptDecoderBatched
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using LlmRequestPtr = std::shared_ptr<suggestify::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using TensorPtr = ITensor::SharedPtr;
    using SharedConstPtr = ITensor::SharedConstPtr;

    enum class ForwardType
    {
        kASYNC,
        kSYNC
    };

    GptDecoderBatched(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream,
        SpeculativeDecodingMode const& speculativeDecodingMode, nvinfer1::DataType dtype);

    void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig) override;

    void setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers) override;

    void setupEagle(EagleBuffers::Inputs eagleBuffers) override;

    void setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers) override;

    void disableLookahead(SizeType32 maxBatchSize, RequestVector const& genRequests) override;

    void newBatch(GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig,
        ModelConfig const& modelConfig) override;

    void newRequests(std::vector<SizeType32> const& seqSlots, std::vector<decoder_batch::Request> const& requests,
        std::vector<SamplingConfig> const& samplingConfigs, ModelConfig const& modelConfig) override;

    DecoderFinishedEventPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) override;

    void forwardSync(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent) override;

    void forwardSync(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent, decoder_batch::Output& output,
        decoder_batch::Input const& input) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    void forwardSync() override;

    [[nodiscard]] std::vector<bool> getFinished() const override
    {
        return {mFinished.begin(), mFinished.begin() + mActualBatchSize};
    }

    [[nodiscard]] TensorPtr getFinishReasons() const override
    {
        return ITensor::slice(mJointDecodingOutput->finishReasons, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getIds(SizeType32 batchIdx) const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->ids, batchIdx, 1);
        tensor->squeeze(0);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    [[nodiscard]] TensorPtr getIds() const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->ids, 0, mActualBatchSize);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    [[nodiscard]] TensorPtr getGatheredIds(SizeType32 batchIdx) const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->gatheredIds, batchIdx, 1);
        tensor->squeeze(0);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    [[nodiscard]] TensorPtr getGatheredIds() const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->gatheredIds, 0, mActualBatchSize);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    [[nodiscard]] CudaEvent finalize(
        SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const override;

    void finalize(SamplingConfig const& samplingConfig) const override;

    [[nodiscard]] TensorPtr getParentIds() const override
    {
        return ITensor::slice(mJointDecodingOutput->parentIds, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getCumLogProbs() const override
    {
        return ITensor::slice(mJointDecodingOutput->cumLogProbs, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getCumLogProbs(SizeType32 batchIdx) const override
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->cumLogProbs, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    [[nodiscard]] TensorPtr getLogProbs() const override
    {
        return ITensor::slice(mJointDecodingOutput->logProbs, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getLogProbs(SizeType32 batchIdx) const override
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->logProbs, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    [[nodiscard]] TensorPtr getAllNewTokens() const override
    {
        return mJointDecodingOutput->newTokensSteps;
    }

    [[nodiscard]] TensorPtr getNewTokens(SizeType32 iter = 0) const override
    {
        TensorPtr newTokensView = ITensor::slice(mJointDecodingOutput->newTokensSteps, iter, 1);
        newTokensView->squeeze(0);
        return ITensor::slice(newTokensView, 0, mActualBatchSize);
    }

    [[nodiscard]] std::vector<SizeType32> getNbSteps() const override
    {
        return {mNbSteps.begin(), mNbSteps.begin() + mActualBatchSize};
    }

    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mFinishedSum;
    }

    [[nodiscard]] TensorPtr getNextDraftTokens() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokens;
    }

    [[nodiscard]] TensorPtr getPrevDraftTokensLengths() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->prevDraftTokensLen;
    }

    [[nodiscard]] TensorPtr getNextDraftTokensLengths() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokensLen;
    }

    [[nodiscard]] TensorPtr getAcceptedLengthsCumSum() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->acceptedLengthsCumSum;
    }

    [[nodiscard]] TensorPtr getAcceptedPackedPaths() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->pathsOffsets;
    }

    executor::DecodingMode getDecodingMode() const override
    {
        return mDecodingMode;
    }

private:
    [[nodiscard]] CudaEvent postProcessRequest(
        SizeType32 batchIdx, SamplingConfig const& samplingConfig, bool streaming) const;

    void newRequest(SizeType32 batchSlot, decoder_batch::Request const& request, SamplingConfig const& samplingConfig,
        ModelConfig const& modelConfig);

    void allocateSpeculativeDecodingBuffers(nvinfer1::DataType dtype);

    void setupSpeculativeDecoding(ModelConfig const& modelConfig);

    void setupLookahead(ModelConfig const& modelConfig);

    void newRequestSpeculativeDecoding(SizeType32 batchIdx, decoder_batch::Request const& request,
        SamplingConfig const& samplingConfig, ModelConfig const& modelConfig);

    void newRequestDraftTokensExternal(
        SizeType32 batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig);

    void newRequestMedusa(SizeType32 batchIdx, decoder_batch::Request const& request);

    void newRequestLookahead(SizeType32 batchIdx, decoder_batch::Request const& request);

    void newRequestExplicitDraftTokens(SizeType32 batchIdx, decoder_batch::Request const& request);

    void newRequestEagle(SizeType32 batchIdx, decoder_batch::Request const& request, ModelConfig const& modelConfig);

    void updateFinished(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent);

    void setExplicitDraftTokensInputs(decoder_batch::Input const& input);

    void setEagleInputs(decoder_batch::Input const& input);

    void forwardDispatch(decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType);

    void forwardDecoder(
        SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType);

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mRuntimeStream;
    CudaStreamPtr mDecoderStream;
    BufferManager mBufferManager;
    DecoderFinishedEventPtr mDecoderFinishEvent;
    CudaEvent mForwardEvent;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;

    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    DecodingInputPtr mJointDecodingInput;
    DecodingOutputPtr mJointDecodingOutput;

    std::vector<SizeType32> mNbSteps;
    std::vector<bool> mFinished;
    TensorPtr mFinishedSum;
    std::vector<SizeType32> mMaxNewTokens;
    std::vector<SizeType32> mBeamWidths;
    std::vector<SizeType32> mNumDecodingEngineTokens;

    TensorPtr mFinishedSteps;

    TensorPtr mBatchSlotsSetup;
    TensorPtr mBatchSlotsDecoder;
    SizeType32 mMaxSequenceLength{};
    SizeType32 mMaxAttentionWindow{};
    SizeType32 mSinkTokenLength{};
    SizeType32 mActualBatchSize{};
    SizeType32 mMaxDecodingDecoderTokens{};
    SizeType32 mMaxDecodingEngineTokens{};

    SpeculativeDecodingMode mSpeculativeDecodingMode;
    executor::DecodingMode mDecodingMode{executor::DecodingMode::Auto()};

    std::shared_ptr<DecodingOutput::BeamHypotheses> mOutputBeamHypotheses{nullptr};
    DecodingOutput::TensorPtr mCumLogProbsTmp;
    SizeType32 mNumSMs;
};
}
