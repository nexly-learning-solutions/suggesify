
#pragma once

#include "../executor/types.h"
#include "cudaStream.h"
#include "generationInput.h"
#include "generationOutput.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "samplingConfig.h"

#include <memory>
#include <utility>

#include <NvInferRuntime.h>

namespace suggestify::batch_manager
{
struct DecoderBuffers;
}

namespace suggestify::runtime
{

namespace decoder
{

class Input
{
public:
    using TensorPtr = ITensor::SharedPtr;

    explicit Input(TensorPtr logits)
        : logits{std::move(logits)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
    }

    TensorPtr logits;

    TensorPtr cacheIndirection;
};

class Output
{
public:
    using TensorPtr = std::shared_ptr<ITensor>;

    Output() = default;

    TensorPtr cacheIndirection;
    TensorPtr sequenceLengths;
};
}

class IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    virtual void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig)
        = 0;

    virtual void newBatch(GenerationInput const& inputs, GenerationOutput const& outputs,
        SamplingConfig const& samplingConfig, ModelConfig const& modelConfig)
        = 0;

    virtual void forwardAsync(decoder::Output& output, decoder::Input const& input) = 0;

    virtual void forwardSync() = 0;

    virtual void forward(decoder::Output& output, decoder::Input const& input)
    {
        forwardAsync(output, input);
        return forwardSync();
    }

    virtual void finalize(SamplingConfig const& samplingConfig) const = 0;

    [[nodiscard]] virtual TensorPtr getIds() const = 0;

    [[nodiscard]] virtual TensorPtr getGatheredIds() const = 0;

    [[nodiscard]] virtual TensorPtr getCumLogProbs() const = 0;

    [[nodiscard]] virtual TensorPtr getLogProbs() const = 0;

    [[nodiscard]] virtual TensorPtr getNewTokens(SizeType32 iter = 0) const = 0;

    [[nodiscard]] virtual TensorPtr getAllNewTokens() const = 0;

    [[nodiscard]] virtual TensorPtr getNbFinished() const = 0;

    virtual ~IStatefulGptDecoder() = default;

protected:
    IStatefulGptDecoder() = default;
};

}
