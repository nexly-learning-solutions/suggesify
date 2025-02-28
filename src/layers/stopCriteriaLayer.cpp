
#include "stopCriteriaLayer.h"
#include "../common/nvtxUtils.h"
#include "../src/stopCriteriaKernels.h"
#include "layerUtils.h"

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
size_t StopCriteriaLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template <typename T>
StopCriteriaLayer<T>::StopCriteriaLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const stopWordsWorkspaceSize = DecodingLayerWorkspace::calculateRequiredWorkspaceSize(
        std::make_pair(ITensor::makeShape({decoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value),
        std::make_pair(ITensor::makeShape({decoderDomain.getBatchSize()}), TRTDataType<TokenIdType*>::value),
        std::make_pair(ITensor::makeShape({decoderDomain.getBatchSize(), decoderDomain.getBeamWidth()}),
            TRTDataType<FinishedState::UnderlyingType>::value));
    auto const lengthCriterionWorkspaceSize = DecodingLayerWorkspace::calculateRequiredWorkspaceSize(
        std::make_pair(ITensor::makeShape({1}), TRTDataType<SizeType32>::value),
        std::make_pair(ITensor::makeShape({decoderDomain.getBatchSize(), decoderDomain.getBeamWidth()}),
            TRTDataType<FinishedState::UnderlyingType>::value));
    mWorkspaceSize = std::max(stopWordsWorkspaceSize, lengthCriterionWorkspaceSize);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& setupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(StopCriteriaLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds->getDimension<-1>();
    auto const* batchSlotsPtr = workspace->getDeviceBatchSlotsPtr();

    CHECK_WITH_INFO(inputs->stopCriteriaInputs, "stopCriteriaInputs for forward is not set");

    if (mDecodingMode.isUseStopWords() && inputs->stopCriteriaInputs->maxStopWordsLen != 0)
    {
        checkStopWordsStopCriteria(outputs, inputs, localDecoderDomain, maxSeqLen, *mBufferManager, workspace);
    }
    if (mDecodingMode.isUseExplicitEosStop())
    {
        checkEosToken(outputs, inputs, localDecoderDomain, *mBufferManager, workspace);
    }
    if (mDecodingMode.isUseMaxLengthStop() && inputs->stopCriteriaInputs->sequenceLimitLength)
    {
        checkMaxLengthStopCriteria(outputs, inputs, localDecoderDomain, *mBufferManager, workspace);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkStopWordsStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain, SizeType32 maxSeqLen,
    BufferManager const& bufferManager, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxStopWordsLength = inputs->stopCriteriaInputs->maxStopWordsLen;
    auto* numNewTokens = bufferCastOrNull<SizeType32>(outputs->numNewTokens);
    auto* outputIdsPtr = bufferCast<SizeType32 const*>(*outputs->outputIdsPtr);
    auto* parentIdsPtr = bufferCast<SizeType32 const*>(*outputs->parentIdsPtr);
    auto* sequenceLengthPtr = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto [stopWordsLengthsDevice, stopWordsPtrDevice, finishedDevice]
        = workspace->mirrorInWorkspace(inputs->stopCriteriaInputs->stopWordsLengths.value_or(nullptr),
            inputs->stopCriteriaInputs->stopWordsPtr.value_or(nullptr), outputs->finished.value_or(nullptr));
    auto const* stopWordsLengthsPtr
        = stopWordsLengthsDevice == nullptr ? nullptr : bufferCast<SizeType32>(*stopWordsLengthsDevice);
    auto const* stopWordsPtrPtr
        = stopWordsPtrDevice == nullptr ? nullptr : bufferCast<TokenIdType const*>(*stopWordsPtrDevice);
    auto* finishedPtr = finishedDevice == nullptr
        ? nullptr
        : reinterpret_cast<FinishedState*>(bufferCast<FinishedState::UnderlyingType>(*finishedDevice));
    invokeStopWordsCriterion(outputIdsPtr, parentIdsPtr, stopWordsPtrPtr, finishedPtr, sequenceLengthPtr,
        workspace->getDeviceBatchSlotsPtr(), stopWordsLengthsPtr, numNewTokens, maxStopWordsLength,
        decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), maxSeqLen, bufferManager.getStream().get());
    if (finishedPtr != nullptr)
    {
        bufferManager.copy(*finishedDevice, *outputs->finished.value());
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkMaxLengthStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
    BufferManager const& bufferManager, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto* numNewTokens = bufferCastOrNull<SizeType32>(outputs->numNewTokens);
    auto [finishedSumDevice, finishedDevice]
        = workspace->mirrorInWorkspace(outputs->finishedSum.value_or(nullptr), outputs->finished.value_or(nullptr));
    auto* finishedSumDevicePtr = finishedSumDevice == nullptr ? nullptr : bufferCast<SizeType32>(*finishedSumDevice);
    auto* finishedPtr = finishedDevice == nullptr
        ? nullptr
        : reinterpret_cast<FinishedState*>(bufferCast<FinishedState::UnderlyingType>(*finishedDevice));
    invokeLengthCriterion(finishedPtr, finishedSumDevicePtr,
        bufferCastOrNull<SizeType32>(inputs->stopCriteriaInputs->sequenceLimitLength),
        bufferCastOrNull<SizeType32>(outputs->sequenceLength), numNewTokens, workspace->getDeviceBatchSlotsPtr(),
        decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), bufferManager.getStream().get());
    if (finishedSumDevice != nullptr)
    {
        bufferManager.copy(*finishedSumDevice, *outputs->finishedSum.value());
    }
    if (finishedPtr != nullptr)
    {
        bufferManager.copy(*finishedDevice, *outputs->finished.value());
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void StopCriteriaLayer<T>::checkEosToken(std::shared_ptr<BaseDecodingOutputs>& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
    BufferManager const& bufferManager, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto* numNewTokens = bufferCastOrNull<SizeType32>(outputs->numNewTokens);
    auto* sequenceLengthsPtr = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto const* endIdsPtr = bufferCastOrNull<TokenIdType>(inputs->endIds);
    auto* finishedStatePtr
        = reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished));
    invokeExplicitEOSCriterion(bufferCastOrNull<TokenIdType const*>(outputs->outputIdsPtr), endIdsPtr, finishedStatePtr,
        sequenceLengthsPtr, numNewTokens, workspace->getDeviceBatchSlotsPtr(), decoderDomain.getBatchSize(),
        decoderDomain.getBeamWidth(), decoderDomain.getMaxDecodingTokens(), bufferManager.getStream().get());
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class StopCriteriaLayer<float>;
template class StopCriteriaLayer<half>;

}
