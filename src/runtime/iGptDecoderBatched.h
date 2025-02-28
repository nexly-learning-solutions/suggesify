
#pragma once

#include "cudaEvent.h"
#include "cudaStream.h"
#include "eagleBuffers.h"
#include "explicitDraftTokensBuffers.h"
#include "iStatefulGptDecoder.h"
#include "iTensor.h"
#include "lookaheadBuffers.h"
#include "request.h"
#include "utils/sessionUtils.h"

#include <memory>
#include <utility>
#include <vector>

namespace suggestify::batch_manager
{
class LlmRequest;
}

namespace suggestify::runtime
{

namespace decoder_batch
{

class Input
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    explicit Input(std::vector<TensorPtr> const& logits, std::vector<bool> const& active)
        : logits{logits}
        , active{active}
    {
        CHECK_WITH_INFO(
            this->active.size() == logits.size(), "'active' vector size does not match logits vector size");
    }

    explicit Input(std::vector<TensorPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    std::vector<TensorPtr>
        logits;

    std::vector<bool> active;

    TensorPtr cacheIndirection;
    std::vector<std::vector<TensorPtr>>
        predictedDraftLogits;
    TensorPtr seqSlots;

    std::optional<ExplicitDraftTokensBuffers::EngineOutputs> explicitDraftTokensInputs;
    std::optional<ExplicitDraftTokensBuffers::EngineInputs> explicitDraftTokensLastInputs;

    std::optional<EagleBuffers::EngineOutputs> eagleInputs;
    std::optional<EagleBuffers::Inputs> eagleLastInputs;
};

using Output = decoder::Output;

class DecoderFinishedEvent
{
public:
    explicit DecoderFinishedEvent(CudaEvent&& event, std::vector<bool> const& active)
        : event(std::move(event))
        , active(active)
    {
    }

    CudaEvent event;
    std::vector<bool> active;
};
}

class IGptDecoderBatched : public virtual IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using LlmRequestPtr = std::shared_ptr<suggestify::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using TensorPtr = std::shared_ptr<ITensor>;
    using DecoderFinishedEventPtr = std::unique_ptr<decoder_batch::DecoderFinishedEvent const>;

    virtual void setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers) = 0;

    virtual void setupEagle(EagleBuffers::Inputs eagleBuffers) = 0;

    virtual void setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers) = 0;

    virtual void disableLookahead(SizeType32 maxBatchSize, RequestVector const& genRequests) = 0;

    virtual DecoderFinishedEventPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    virtual void forwardSync(decoder_batch::DecoderFinishedEvent const& token, decoder_batch::Output& output,
        decoder_batch::Input const& input)
        = 0;

    virtual void forwardSync(decoder_batch::DecoderFinishedEvent const& token) = 0;

    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input)
    {
        forwardSync(*forwardAsync(output, input));
    }

    [[nodiscard]] virtual TensorPtr getIds(SizeType32 batchIdx) const = 0;

    [[nodiscard]] virtual TensorPtr getGatheredIds(SizeType32 batchIdx) const = 0;

    [[nodiscard]] virtual CudaEvent finalize(
        SizeType32 batchIdx, SamplingConfig const& samplingConfig, bool streaming) const
        = 0;

    [[nodiscard]] virtual std::vector<bool> getFinished() const = 0;

    [[nodiscard]] virtual TensorPtr getFinishReasons() const = 0;

    [[nodiscard]] virtual TensorPtr getCumLogProbs() const = 0;

    [[nodiscard]] virtual TensorPtr getCumLogProbs(SizeType32 batchIdx) const = 0;

    [[nodiscard]] virtual TensorPtr getLogProbs() const = 0;

    [[nodiscard]] virtual TensorPtr getLogProbs(SizeType32 batchIdx) const = 0;

    [[nodiscard]] virtual TensorPtr getParentIds() const = 0;

    [[nodiscard]] virtual std::vector<SizeType32> getNbSteps() const = 0;

    [[nodiscard]] virtual executor::DecodingMode getDecodingMode() const = 0;

    virtual void newRequests(std::vector<SizeType32> const& seqSlots,
        std::vector<decoder_batch::Request> const& requests, std::vector<SamplingConfig> const& samplingConfigs,
        ModelConfig const& modelConfig)
        = 0;

    virtual TensorPtr getNextDraftTokens() const = 0;

    virtual TensorPtr getPrevDraftTokensLengths() const = 0;

    virtual TensorPtr getNextDraftTokensLengths() const = 0;

    virtual TensorPtr getAcceptedLengthsCumSum() const = 0;

    virtual TensorPtr getAcceptedPackedPaths() const = 0;

protected:
    IGptDecoderBatched() = default;
};

}
