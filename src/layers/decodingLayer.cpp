
#include "decodingLayer.h"
#include "beamSearchLayer.h"
#include "decodingParams.h"
#include "eagleDecodingLayer.h"
#include "explicitDraftTokensLayer.h"
#include "externalDraftTokensLayer.h"
#include "layerUtils.h"
#include "lookaheadDecodingLayer.h"
#include "medusaDecodingLayer.h"
#include "samplingLayer.h"

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
DecodingLayer<T>::DecodingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isTopKorTopP())
    {
        mDecodingLayer = std::make_unique<SamplingLayer<T>>(mDecodingMode, decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isBeamSearch())
    {
        mDecodingLayer = std::make_unique<BeamSearchLayer<T>>(decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isMedusa())
    {
        mDecodingLayer = std::make_unique<MedusaDecodingLayer<T>>(decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isLookahead())
    {
        mDecodingLayer = std::make_unique<LookaheadDecodingLayer<T>>(mDecoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        mDecodingLayer = std::make_unique<ExplicitDraftTokensLayer<T>>(decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isExternalDraftTokens())
    {
        mDecodingLayer = std::make_unique<ExternalDraftTokensLayer<T>>(mDecodingMode, decoderDomain, mBufferManager);
    }
    else if (mDecodingMode.isEagle())
    {
        mDecodingLayer = std::make_unique<EagleDecodingLayer<T>>(decoderDomain, mBufferManager);
    }
    else
    {
        CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens, ExternalDraftTokens, Eagle}");
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    CHECK_WITH_INFO(setupParams->decodingParams, "decodingParams for setup is not set");

    if (mDecodingMode.isTopKorTopP())
    {
        CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);
    }
    else if (mDecodingMode.isBeamSearch())
    {
        CHECK_WITH_INFO(beamWidth > 1, "Decoding mode is beam search, but beamWidth <= 1 (%d <= 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);
    }
    else if (mDecodingMode.isMedusa())
    {
        CHECK_WITH_INFO(beamWidth == 1, "Decoding mode is Medusa, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);
    }
    else if (mDecodingMode.isLookahead())
    {
        CHECK_WITH_INFO(beamWidth == 1, "Decoding mode is Lookahead, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is ExplicitDraftTokens, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);
    }
    else if (mDecodingMode.isExternalDraftTokens())
    {
        CHECK_WITH_INFO(
            beamWidth == 1, "Decoding mode is external draft tokens, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);
    }
    else if (mDecodingMode.isEagle())
    {
        CHECK_WITH_INFO(beamWidth == 1, "Decoding mode is Eagle, but beamWidth != 1 (%d != 1)", beamWidth);
        mDecodingLayer->setup(batchSize, beamWidth, batchSlots, setupParams->decodingParams, workspace);
    }
    else
    {
        CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens, ExternalDraftTokens, Eagle}");
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardAsync(outputParams, inputParams, workspace);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DecodingLayer<T>::forwardSync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto [outputParams, inputParams] = prepareParams(baseOutputs, baseInputs);
    mDecodingLayer->forwardSync(outputParams, inputParams, workspace);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t DecodingLayer<T>::getWorkspaceSize() const noexcept
{
    return mDecodingLayer->getWorkspaceSize();
}

template <typename T>
std::tuple<std::shared_ptr<BaseDecodingOutputs>, std::shared_ptr<BaseDecodingInputs>> DecodingLayer<T>::prepareParams(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto params = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(params, mDecoderDomain);
    auto const maxSeqLen = baseOutputs->outputIds->getDimension<-1>();
    auto const& endIds = params->endIds;

    std::shared_ptr<BaseDecodingOutputs> preparedOutputs;
    std::shared_ptr<BaseDecodingInputs> preparedInputs;

    if (mDecodingMode.isBeamSearch())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isTopKorTopP())
    {
        auto const ite = params->ite;
        auto const step = params->step;
        auto const localBatchSize = static_cast<int64_t>(params->localBatchSize);

        CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        TensorConstPtr logitsSlice = ITensor::slice(*params->logits, 0, localBatchSize);
        TensorConstPtr endIdSlice = ITensor::slice(endIds, 0, localBatchSize);
        auto decodeInputs = std::make_shared<SamplingInputs>(endIdSlice, params->batchSlots, step, ite, localBatchSize);

        decodeInputs->finished = params->finished;

        decodeInputs->logits = logitsSlice;

        if (params->inputLengths)
        {
            auto& inputLengths = params->inputLengths.value();
            decodeInputs->inputLengths = ITensor::slice(inputLengths, 0, localBatchSize);
        }
        preparedInputs = decodeInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isMedusa())
    {
        CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is Medusa, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isLookahead())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isExplicitDraftTokens())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isExternalDraftTokens())
    {
        auto externalDraftTokenParams = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);
        auto const ite = externalDraftTokenParams->ite;
        auto const step = externalDraftTokenParams->step;
        auto const localBatchSize = static_cast<int64_t>(externalDraftTokenParams->localBatchSize);

        CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() == 1,
            "Decoding mode is TopK and/or TopP, but beamWidth != 1 (%d != 1)", localDecoderDomain.getBeamWidth());

        TensorConstPtr logitsSlice = ITensor::slice(*externalDraftTokenParams->logits, 0, localBatchSize);
        TensorConstPtr endIdSlice = ITensor::slice(endIds, 0, localBatchSize);
        auto decodeInputs = std::make_shared<ExternalDraftTokensInputs>(
            endIdSlice, externalDraftTokenParams->batchSlots, step, ite, localBatchSize);

        decodeInputs->finished = externalDraftTokenParams->finished;

        decodeInputs->logits = logitsSlice;

        if (externalDraftTokenParams->inputLengths)
        {
            auto& inputLengths = externalDraftTokenParams->inputLengths.value();
            decodeInputs->inputLengths = ITensor::slice(inputLengths, 0, localBatchSize);
        }
        decodeInputs->draftLogits = externalDraftTokenParams->draftLogits;
        decodeInputs->draftProbs = externalDraftTokenParams->draftProbs;
        decodeInputs->targetProbs = externalDraftTokenParams->targetProbs;
        decodeInputs->numDraftTokens = externalDraftTokenParams->numDraftTokens;
        decodeInputs->draftTokenIds = externalDraftTokenParams->draftTokenIds;
        decodeInputs->constantThreshold = externalDraftTokenParams->constantThreshold;
        decodeInputs->useRandomAcceptanceThreshold = externalDraftTokenParams->useRandomAcceptanceThreshold;
        decodeInputs->step = externalDraftTokenParams->step;
        decodeInputs->useDraftLogits = externalDraftTokenParams->useDraftLogits;
        decodeInputs->useDraftLogitsHost = externalDraftTokenParams->useDraftLogitsHost;

        preparedInputs = decodeInputs;
        preparedOutputs = baseOutputs;
    }
    else if (mDecodingMode.isEagle())
    {
        preparedInputs = baseInputs;
        preparedOutputs = baseOutputs;
    }
    else
    {
        CHECK_WITH_INFO(false,
            "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch, Medusa, Lookahead, "
            "ExplicitDraftTokens, ExternalDraftTokens, Eagle}");
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {preparedOutputs, preparedInputs};
}

template class DecodingLayer<float>;
template class DecodingLayer<half>;

}
