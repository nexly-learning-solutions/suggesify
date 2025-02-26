
#include "dynamicDecodeLayer.h"
#include "suggestify/common/nvtxUtils.h"
#include "../src/decodingKernels.h"
#include "layerUtils.h"
#include "layersFactory.h"
#include "bufferManager.h"
#include "iBuffer.h"
#include "iTensor.h"

#include <optional>

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
size_t DynamicDecodeLayer<T>::getWorkspaceSize() const noexcept
{
    size_t maxWorkspaceSize = 0;
    for (auto const& layer : mLayers)
    {
        maxWorkspaceSize = std::max(maxWorkspaceSize, layer->getWorkspaceSize());
    }
    return maxWorkspaceSize;
}

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    initialize();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::initialize()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mOutputIdsPtrHost = mBufferManager->pinnedPool(ITensor::makeShape({}), TRTDataType<TokenIdType*>::value);
    mParentIdsPtrHost = mBufferManager->pinnedPool(ITensor::makeShape({}), TRTDataType<TokenIdType*>::value);
    mOutputIdsPtrDevice = mBufferManager->gpu(
        ITensor::makeShape({static_cast<SizeType32>(mDecoderDomain.getBatchSize())}), TRTDataType<TokenIdType*>::value);
    mParentIdsPtrDevice = mBufferManager->gpu(
        ITensor::makeShape({static_cast<SizeType32>(mDecoderDomain.getBatchSize())}), TRTDataType<TokenIdType*>::value);

    allocateBuffer();

    mCyclicStep = 0;
    mRuntimeMaxSeqLen = 0;
    mConfiguredBeamWidth = -1;

    if (!mDecodingMode.isAuto())
    {
        mConfiguredBeamWidth = mDecoderDomain.getBeamWidth();
        initializeLayers();
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::allocateBuffer()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mZeroParentIdsDevice
        = mBufferManager->gpu(ITensor::makeShape({2 * mDecoderDomain.getBatchSize()}), TRTDataType<TokenIdType>::value);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::initializeLayers()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mLayers = createLayers<T>(mDecodingMode, mDecoderDomain, mBufferManager);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::disableLookahead(DecoderDomain const& decoderDomain, SizeType32 batchSize,
    TensorConstPtr batchSlots, std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mDecodingMode = executor::DecodingMode::TopKTopP();
    mDecoderDomain = std::move(decoderDomain);
    initializeLayers();
    if (batchSize > 0)
    {
        setup(batchSize, 1, batchSlots, baseSetupParams, workspace);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);
    workspace->setDeviceBatchSlots(
        batchSlots);

    CHECK_WITH_INFO(setupParams->decodingParams, "decodingParams for setup is not set");
    if (setupParams->decodingParams->outputLogProbs)
    {
        mOutputLogProbs = std::any_of(setupParams->decodingParams->outputLogProbs->begin(),
            setupParams->decodingParams->outputLogProbs->end(),
            [this](bool outputLogProbs) { return this->mOutputLogProbs | outputLogProbs; });
    }

    if (mConfiguredBeamWidth == -1)
    {
        CHECK(mDecodingMode.isAuto());
        mConfiguredBeamWidth = beamWidth;
        mDecodingMode
            = mConfiguredBeamWidth == 1 ? executor::DecodingMode::TopKTopP() : executor::DecodingMode::BeamSearch();
        initializeLayers();
        auto const workspaceSize = getWorkspaceSize();
        workspace->resize(workspaceSize);
    }

    CHECK_WITH_INFO((mConfiguredBeamWidth == 1 && beamWidth == 1)
            || (mConfiguredBeamWidth > 1 && beamWidth > 1 && beamWidth <= mConfiguredBeamWidth),
        "Decoder is configured with beam width %d, but %d was given", mConfiguredBeamWidth, beamWidth);
    CHECK_WITH_INFO(mConfiguredBeamWidth <= mDecoderDomain.getBeamWidth(),
        "Decoder is created with max beam width %d, but %d was given", mDecoderDomain.getBeamWidth(),
        mConfiguredBeamWidth);

    for (auto& layer : mLayers)
    {
        layer->setup(batchSize, beamWidth, batchSlots, baseSetupParams, workspace);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(DynamicDecodeLayer_forwardAsync);

    auto params = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);

    CHECK_WITH_INFO(
        mDecodingMode.isExplicitDraftTokens() || mDecodingMode.isEagle() || params->logits || params->logitsVec,
        "If not Explicit Draft Tokens or Eagle mode, either logits or logitsVec have to be specified.");
    CHECK_WITH_INFO(
        baseOutputs->sequenceLength.has_value(), "sequenceLength tensor is required in DynamicDecoderLayer.");

    auto const localDecoderDomain = getLocalDecoderDomain(params, mDecoderDomain);
    auto const maxSeqLen = baseOutputs->outputIds->getDimension<-1>();

    CHECK_WITH_INFO((mConfiguredBeamWidth == 1 && localDecoderDomain.getBeamWidth() == 1)
            || (mConfiguredBeamWidth > 1 && localDecoderDomain.getBeamWidth() > 1
                && localDecoderDomain.getBeamWidth() <= mConfiguredBeamWidth),
        "Decoder is configured with beam width %d, but %d was given", mConfiguredBeamWidth,
        localDecoderDomain.getBeamWidth());

    if (mOutputIdsPtrHost->getSize() == 0)
    {
        mOutputIdsPtrHost->reshape(
            ITensor::makeShape({static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(mDecoderDomain.getBatchSize())}));
        mParentIdsPtrHost->reshape(
            ITensor::makeShape({static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(mDecoderDomain.getBatchSize())}));
        mRuntimeMaxSeqLen = maxSeqLen;
    }

    mCyclicStep = mCyclicStep % mRuntimeMaxSeqLen;
    workspace->setDeviceBatchSlots(
        params->batchSlots);
    prepareIdsPtrs(baseOutputs, params->batchSlots, localDecoderDomain.getBatchSize(),
        localDecoderDomain.getBeamWidth(), maxSeqLen);

    for (auto& layer : mLayers)
    {
        layer->forwardAsync(baseOutputs, baseInputs, workspace);
    }

    prepareOutputData(baseOutputs, workspace->getDeviceBatchSlots(), localDecoderDomain.getBatchSize(),
        mDecoderDomain.getBatchSize(), localDecoderDomain.getBeamWidth(), maxSeqLen,
        mDecoderDomain.getMaxDecodingTokens(), mOutputLogProbs, getStream());

    mCyclicStep += 1;

    sync_check_cuda_error();
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::forwardSync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(DynamicDecodeLayer_forwardSync);

    for (auto& layer : mLayers)
    {
        layer->forwardSync(baseOutputs, baseInputs, workspace);
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::prepareIdsPtrs(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    BufferConstPtr batchSlots, SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSeqLen)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TensorPtr outputIdsPtrHostSlice = ITensor::slice(mOutputIdsPtrHost, mCyclicStep, 1);
    TensorPtr parentIdsPtrHostSlice = ITensor::slice(mParentIdsPtrHost, mCyclicStep, 1);
    auto outputIdsPtrHost = runtime::bufferCast<TokenIdType*>(*outputIdsPtrHostSlice);
    auto parentIdsPtrHost = runtime::bufferCast<TokenIdType*>(*parentIdsPtrHostSlice);
    auto const* batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        auto const batchSlot = batchSlotsPtr[bi];
        outputIdsPtrHost[batchSlot] = bufferCast<TokenIdType>(*outputs->outputIds) + batchSlot * beamWidth * maxSeqLen;
    }
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        auto const batchSlot = batchSlotsPtr[bi];
        if (beamWidth > 1)
        {
            parentIdsPtrHost[batchSlot]
                = bufferCast<TokenIdType>(*outputs->parentIds.value()) + batchSlot * beamWidth * maxSeqLen;
        }
        else
        {
            auto mZeroParentIdsDevicePtr = bufferCast<TokenIdType>(*mZeroParentIdsDevice);
            parentIdsPtrHost[batchSlot] = mZeroParentIdsDevicePtr + bi * beamWidth * maxSeqLen;
        }
    }

    mBufferManager->copy(*outputIdsPtrHostSlice, *mOutputIdsPtrDevice);
    mBufferManager->copy(*parentIdsPtrHostSlice, *mParentIdsPtrDevice);
    outputs->outputIdsPtr = ITensor::slice(mOutputIdsPtrDevice, 0, batchSize);
    outputs->parentIdsPtr = ITensor::slice(mParentIdsPtrDevice, 0, batchSize);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::prepareOutputData(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    BufferConstPtr batchSlots, SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth,
    SizeType32 maxSeqLen, SizeType32 maxTokensPerStep, bool outputLogProbs, cudaStream_t stream)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto outputIdsPtrDevice = bufferCast<TokenIdType*>(*mOutputIdsPtrDevice);
    auto const numNewTokens = bufferCastOrNull<SizeType32>(outputs->numNewTokens);
    auto newTokensPtr = bufferCast<TokenIdType>(*outputs->newTokens);
    auto sequenceLengthsPtr = bufferCast<SizeType32>(*outputs->sequenceLength.value());
    auto const* batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);

    invokeCopyNextStepIds(newTokensPtr, outputIdsPtrDevice, sequenceLengthsPtr, numNewTokens, batchSlotsPtr, batchSize,
        maxBatchSize, beamWidth, maxSeqLen, maxTokensPerStep, stream);

    if (outputLogProbs && outputs->outputLogProbsTiled)
    {
        auto logProbsMaxSeqLen = outputs->outputLogProbsTiled.value()->getDimension<0>();

        auto outputLogProbsPtr = bufferCast<float>(*outputs->outputLogProbs.value());
        auto outputLogProbsTiledPtr = bufferCast<float>(*outputs->outputLogProbsTiled.value());
        invokeTransposeLogProbs(outputLogProbsPtr, outputLogProbsTiledPtr, sequenceLengthsPtr, batchSlotsPtr, batchSize,
            maxBatchSize, beamWidth, logProbsMaxSeqLen, stream);
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class DynamicDecodeLayer<float>;
template class DynamicDecodeLayer<half>;

}
