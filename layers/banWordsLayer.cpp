
#include "banWordsLayer.h"
#include "suggestify/kernels/banBadWords.h"
#include "suggestify/kernels/banRepeatNgram.h"
#include "defaultDecodingParams.h"
#include "layerUtils.h"

using namespace suggestify::kernels;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
BanWordsLayer<T>::BanWordsLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    allocateBuffer();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseNoRepeatNgramSize())
    {
        mNoRepeatNgramSizeDevice
            = mBufferManager->gpu(ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value);
    }

    mNoRepeatNgramSize = mBufferManager->pinnedPool(
        ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);
    auto const& banWordsParams = setupParams->banWordsParams;
    TLLM_CHECK_WITH_INFO(banWordsParams, "banWordsParams for setup is not set");
    bool const useNoRepeatNgramSize
        = mDecodingMode.isUseNoRepeatNgramSize() && banWordsParams->noRepeatNgramSize.has_value();
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};
    mUseNoRepeatNgramSize |= useNoRepeatNgramSize;
    if (mUseNoRepeatNgramSize)
    {
        fillBuffers(banWordsParams->noRepeatNgramSize, DefaultDecodingParams::getNoRepeatNgramSize(),
            mNoRepeatNgramSize, mNoRepeatNgramSizeDevice, batchSlots,
            std::make_pair(0.f, std::numeric_limits<float>::max()), "no_repeat_ngram_size");
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::banRepeatNGrams(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots, BufferPtr noRepeatNgramSizeDevice,
    DecoderDomain const& decoderDomain, SizeType32 maxSeqLen, bool useNoRepeatNgramSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (useNoRepeatNgramSize)
    {
        auto const maxStep = maxSeqLen;
        auto logitsPtr = bufferCast<T>(*logits);
        auto outputIdsPtr = bufferCast<TokenIdType const*>(*outputs->outputIdsPtr);
        auto finishedPtr
            = reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished));
        auto parentIdsPtr = bufferCast<SizeType32 const*>(*outputs->parentIdsPtr);
        auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);
        auto sequenceLengthPtr = bufferCast<SizeType32>(*outputs->sequenceLength.value());
        auto noRepeatNgramSizeDevicePtr = bufferCastOrNull<SizeType32>(noRepeatNgramSizeDevice);

        invokeBanRepeatNgram(logitsPtr, outputIdsPtr, finishedPtr, parentIdsPtr, batchSlotsPtr, sequenceLengthPtr,
            decoderDomain.getBatchSize(), decoderDomain.getBeamWidth(), maxSeqLen, noRepeatNgramSizeDevicePtr,
            decoderDomain.getVocabSizePadded(), maxStep, getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::banBadWords(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots, DecoderDomain const& decoderDomain,
    SizeType32 maxSeqLen)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxBadWordsLength = inputs->banWordsInputs->maxBadWordsLen;
    if (maxBadWordsLength != 0)
    {
        auto badWordsPtr = bufferCast<TokenIdType const*>(*inputs->banWordsInputs->badWordsPtr.value());
        auto badWordsLens = bufferCast<SizeType32>(*inputs->banWordsInputs->badWordsLengths.value());
        auto logitsPtr = bufferCast<T>(*logits);
        auto outputIdsPtr = bufferCast<TokenIdType const*>(*outputs->outputIdsPtr);
        auto parentIdsPtr
            = decoderDomain.getBeamWidth() > 1 ? bufferCast<SizeType32 const*>(*outputs->parentIdsPtr) : nullptr;
        auto sequenceLengthPtr = bufferCast<SizeType32>(*outputs->sequenceLength.value());
        auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);

        invokeBanBadWords(logitsPtr, outputIdsPtr, parentIdsPtr, batchSlotsPtr, decoderDomain.getBatchSize(),
            decoderDomain.getBeamWidth(), badWordsPtr, badWordsLens, maxBadWordsLength,
            decoderDomain.getVocabSizePadded(), sequenceLengthPtr, maxSeqLen, getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BanWordsLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);

    TLLM_CHECK_WITH_INFO(inputs->banWordsInputs, "banWordsInputs for forward is not set");

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds->getDimension<-1>();

    banRepeatNGrams(workspace->getDeviceRuntimeLogits(), outputs, inputs, workspace->getDeviceBatchSlots(),
        mNoRepeatNgramSizeDevice, localDecoderDomain, maxSeqLen, mUseNoRepeatNgramSize);
    banBadWords(workspace->getDeviceRuntimeLogits(), outputs, inputs, workspace->getDeviceBatchSlots(),
        localDecoderDomain, maxSeqLen);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BanWordsLayer<float>;
template class BanWordsLayer<half>;

}
