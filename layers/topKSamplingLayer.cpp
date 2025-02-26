
#include "topKSamplingLayer.h"
#include "../common/logger.h"
#include "../common/nvtxUtils.h"
#include "../src/decodingCommon.h"
#include "../src/samplingTopKKernels.h"
#include "defaultDecodingParams.h"
#include "layerUtils.h"

#include <algorithm>
#include <cfloat>

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer(mDecoderDomain.getBatchSize());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer(SizeType32 const batchSize)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mWorkspaceSize = getTopKWorkspaceSize<T>(batchSize, 1, TOP_K_MAX, mDecoderDomain.getVocabSizePadded());
    auto const batchSizeShape = ITensor::makeShape({batchSize});
    mRuntimeTopKDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mRuntimeTopPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mSkipDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mRuntimeTopKHost = mBufferManager->cpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mSkipDecodeHost = mBufferManager->cpu(batchSizeShape, TRTDataType<bool>::value);
    mSetupWorkspaceSize = batchSize * sizeof(SizeType32);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopKSamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);
    mNormalizeLogProbs = setupParams->normalizeLogProbs.value_or(false);

    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector{DefaultDecodingParams::getTopK()});
    auto runtimeTopP = setupParams->runtimeTopP.value_or(std::vector{DefaultDecodingParams::getTopP()});

    auto const paramsSize = expandMatchElements(batchSize, runtimeTopK, runtimeTopP);

    CHECK_WITH_INFO(paramsSize != 0,
        fmtstr("TopKSamplingLayer got parameter with unexpected size, want 1 or batchSize(%d), got"
               "runtimeTopK.size() = %zu, runtimeTopP.size() = %zu",
            batchSize, runtimeTopK.size(), runtimeTopP.size()));

    for (size_t i = 0; i < paramsSize; ++i)
    {
        auto& topK = runtimeTopK[i];
        auto& topP = runtimeTopP[i];
        clampTopK(topK);
        clampTopP(topP);
        regularizeTopKTopP(topK, topP);
    }

    SizeType32* topKsPtr = nullptr;
    float* topPsPtr = nullptr;

    if (paramsSize > 1)
    {
        auto initWorkspaceSizes = getTopKInitWorkspaceSizes(batchSize);
        auto workspacePtr = workspace->getRawWorkspaceDevicePtr();
        calcAlignedPointers(workspacePtr, initWorkspaceSizes)(topKsPtr, topPsPtr);
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopK, IBuffer::wrap(topKsPtr, initWorkspaceSizes[0] / sizeof(*topKsPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopP, IBuffer::wrap(topPsPtr, initWorkspaceSizes[1] / sizeof(*topPsPtr)));
    }
    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();
    auto* skipDecodeDevicePtr = bufferCastOrNull<bool>(mSkipDecodeDevice);
    invokeSetupTopKRuntimeArgs(batchSize,
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKDevice)},
        {topPsPtr, runtimeTopP.front(), bufferCast<float>(*mRuntimeTopPDevice)},
        skipDecodeDevicePtr, batchSlotsDevicePtr, true, getStream());

    auto const* batchSlotsHostPtr = bufferCast<SizeType32>(*batchSlots);
    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    topKsPtr = paramsSize > 1 ? runtimeTopK.data() : nullptr;
    invokeSetupTopKRuntimeArgs(batchSize,
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKHost)}, {},
        skipDecodeHostPtr, batchSlotsHostPtr, false);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopKSamplingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(TopKSamplingLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    auto const skip = allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    auto logits = bufferCastOrNull<T>(inputs->logits);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);
    auto const probsComputed = inputs->probsComputed;

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished))
        : nullptr;
    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    auto* runtimeTopKHostPtr = bufferCast<SizeType32>(*mRuntimeTopKHost);

    TopKSamplingKernelParams<T> params;
    params.logProbs = logits;
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.maxTopP = 1.0f;
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.maxTopK = maxOfBatchSlots(batchSlotsHost, runtimeTopKHostPtr, batchSize);
    params.topKs = bufferCastOrNull<SizeType32>(mRuntimeTopKDevice);
    params.sequenceLengths = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    params.endIds = endIds;
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipDecodeDevice);
    params.cumLogProbs = bufferCastOrNull<float>(outputs->cumLogProbs);
    params.outputLogProbs = bufferCastOrNull<float>(outputs->outputLogProbsTiled);
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.normalizeLogProbs = mNormalizeLogProbs;
    params.logitsHasProbs = probsComputed;

    invokeBatchTopKSampling(params, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t TopKSamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mWorkspaceSize, mSetupWorkspaceSize);
}

template class TopKSamplingLayer<float>;
template class TopKSamplingLayer<half>;

}
