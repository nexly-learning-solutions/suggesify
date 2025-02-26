
#pragma once

#include "suggestify/executor/types.h"
#include "bufferManager.h"
#include "cudaEvent.h"
#include "gptDecoder.h"
#include "iStatefulGptDecoder.h"
#include "iTensor.h"

#include <memory>

namespace suggestify::runtime
{

class StatefulGptDecoder : public IStatefulGptDecoder
{
public:
    StatefulGptDecoder(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream);

    void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig) override;

    void newBatch(GenerationInput const& input, GenerationOutput const& output, SamplingConfig const& samplingConfig,
        ModelConfig const& modelConfig) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    void forwardSync() override;

    void finalize(SamplingConfig const& samplingConfig) const override;

    [[nodiscard]] TensorPtr getIds() const override
    {
        return mDecodingOutput->ids;
    }

    [[nodiscard]] TensorPtr getGatheredIds() const override
    {
        return mDecodingOutput->ids;
    }

    [[nodiscard]] TensorPtr getCumLogProbs() const override
    {
        return mDecodingOutput->cumLogProbs;
    }

    [[nodiscard]] TensorPtr getLogProbs() const override
    {
        return mDecodingOutput->logProbs;
    }

    [[nodiscard]] TensorPtr getNewTokens(SizeType32 iter = 0) const override
    {
        TLLM_CHECK(iter == 0);
        return mDecodingOutput->newTokens;
    }

    [[nodiscard]] TensorPtr getAllNewTokens() const override
    {
        TensorPtr newTokens = ITensor::view(mDecodingOutput->newTokensSteps);
        newTokens->unsqueeze(0);
        return newTokens;
    }

    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mFinishedSum;
    }

private:
    void reshapeBuffers(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 mMaxAttentionWindow,
        SizeType32 mSinkTokenLength, SizeType32 maxSequenceLength);

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mStream;
    BufferManager mBufferManager;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;
    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    DecodingInputPtr mDecodingInput;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    DecodingOutputPtr mDecodingOutput;
    CudaEvent mDecodedEvent{};

    TensorPtr mFinishedSum;
    TensorPtr mSetupBatchSlots;

    SizeType32 mNbSteps;
    SizeType32 mMaxSequenceLength{};
    SizeType32 mMaxAttentionWindow{};
    SizeType32 mSinkTokenLength{};
};
}
