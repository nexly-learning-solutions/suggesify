
#include "penaltyLayer.h"
#include "../common/cudaUtils.h"
#include "../common/nvtxUtils.h"
#include "../src/penaltyKernels.h"
#include "../src/penaltyTypes.h"
#include "defaultDecodingParams.h"
#include "layerUtils.h"
#include "bufferManager.h"

#include <algorithm>

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
size_t PenaltyLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template <typename T>
PenaltyLayer<T>::PenaltyLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    initialize();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::initialize()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    mCyclicStep = 0;
    mRuntimeMaxSeqLen = 0;
    mConfiguredBeamWidth = -1;

    if (!mDecodingMode.isAuto())
    {
        mConfiguredBeamWidth = mDecoderDomain.getBeamWidth();

        allocateWorkspace();
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::allocateWorkspace()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseOccurrencePenalty())
    {

        auto const workspaceSize = mDecoderDomain.getBatchSize() * mDecoderDomain.getMaxDecodingTokens()
            * mConfiguredBeamWidth * mDecoderDomain.getVocabSize();
        mPenaltyWorkspaceDevice = mBufferManager->gpu(workspaceSize, nvinfer1::DataType::kINT32);

        if (mDecodingMode.isBeamSearch())
        {
            mPenaltyWorkspacePrevDevice = mBufferManager->gpu(workspaceSize, nvinfer1::DataType::kINT32);
        }
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::allocateBuffer()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mLogitsPtrsHost = mBufferManager->pinnedPool(ITensor::makeShape({}), TRTDataType<T*>::value);
    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mTemperature = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mRepetitionPenalty = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mPresencePenalty = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mFrequencyPenalty = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mMinLength = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<SizeType32>::value);

    if (mDecodingMode.isUseTemperature())
    {
        mTemperatureDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUseRepetitionPenalty())
    {
        mRepetitionPenaltyDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUsePresencePenalty())
    {
        mPresencePenaltyDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUseFrequencyPenalty())
    {
        mFrequencyPenaltyDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUseMinLength())
    {
        mMinLengthDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kINT32);
    }

    auto const logitsPtrDeviceDesc = std::make_pair(batchSizeShape, TRTDataType<T*>::value);
    mWorkspaceSize = DecodingLayerWorkspace::calculateRequiredWorkspaceSize(logitsPtrDeviceDesc);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    if (mConfiguredBeamWidth == -1)
    {
        CHECK(mDecodingMode.isAuto());
        mConfiguredBeamWidth = beamWidth;
        mDecodingMode
            = mConfiguredBeamWidth == 1 ? executor::DecodingMode::TopKTopP() : executor::DecodingMode::BeamSearch();
        allocateWorkspace();
    }

    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};

    auto const& penaltyParams = setupParams->penaltyParams;
    CHECK_WITH_INFO(penaltyParams, "penaltyParams for setup is not set");

    bool const useTemperature = mDecodingMode.isUseTemperature() && penaltyParams->temperature.has_value();
    bool const useRepetitionPenalty
        = mDecodingMode.isUseRepetitionPenalty() && penaltyParams->repetitionPenalty.has_value();
    bool const usePresencePenalty = mDecodingMode.isUsePresencePenalty() && penaltyParams->presencePenalty.has_value();
    bool const useFrequencyPenalty
        = mDecodingMode.isUseFrequencyPenalty() && penaltyParams->frequencyPenalty.has_value();
    bool const useMinLength = mDecodingMode.isUseMinLength() && penaltyParams->minLength.has_value();
    mUseTemperature |= useTemperature;
    mUseRepetitionPenalty |= useRepetitionPenalty;
    mUsePresencePenalty |= usePresencePenalty;
    mUseFrequencyPenalty |= useFrequencyPenalty;
    mUseMinLength |= useMinLength;

    if (mUseTemperature)
    {
        fillBuffers(penaltyParams->temperature, DefaultDecodingParams::getTemperature(), mTemperature,
            mTemperatureDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Temperature), "temperature penalty");
    }
    if (mUseRepetitionPenalty)
    {
        fillBuffers(penaltyParams->repetitionPenalty, DefaultDecodingParams::getRepetitionPenalty(), mRepetitionPenalty,
            mRepetitionPenaltyDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Repetition),
            "repetition penalty");
    }
    if (mUsePresencePenalty)
    {
        fillBuffers(penaltyParams->presencePenalty, DefaultDecodingParams::getPresencePenalty(), mPresencePenalty,
            mPresencePenaltyDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Presence), "presence penalty");
    }
    if (mUseFrequencyPenalty)
    {
        fillBuffers(penaltyParams->frequencyPenalty, DefaultDecodingParams::getFrequencyPenalty(), mFrequencyPenalty,
            mFrequencyPenaltyDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Frequency), "frequency penalty");
    }
    if (mUseMinLength)
    {
        fillBuffers(penaltyParams->minLength, DefaultDecodingParams::getMinLength(), mMinLength, mMinLengthDevice,
            batchSlots, getLimitsPenalty(DecodingPenaltyType::MinLength), "min length");
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(PenaltyLayer_forwardAsync);

    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);
    auto params = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(params, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds->getDimension<-1>();

    if (mLogitsPtrsHost->data() == nullptr)
    {
        mLogitsPtrsHost->reshape(
            ITensor::makeShape({static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(mDecoderDomain.getBatchSize())}));
        mRuntimeMaxSeqLen = maxSeqLen;
    }

    mCyclicStep = mCyclicStep % mRuntimeMaxSeqLen;

    TensorPtr logitsPtrsHost = ITensor::slice(mLogitsPtrsHost, mCyclicStep, 1);
    logitsPtrsHost->squeeze(0);
    auto logitsPtrsHostData = bufferCast<T const*>(*logitsPtrsHost);
    for (SizeType32 bi = 0; bi < localDecoderDomain.getBatchSize(); bi++)
    {
        if (params->logitsVec)
        {
            CHECK_WITH_INFO(params->logitsVec->size() == localDecoderDomain.getBatchSize(),
                "Logits vector size (%lu) is not equal to the batchSize (%d)", params->logitsVec->size(),
                localDecoderDomain.getBatchSize());
            logitsPtrsHostData[bi] = bufferCastOrNull<T>(params->logitsVec.value()[bi]);
        }
        else
        {
            TensorConstPtr logitsForBatchIndex = ITensor::slice(params->logits.value(), ITensor::makeShape({bi}));
            auto const ptrToLogitsForBatchIndex = bufferCastOrNull<T>(logitsForBatchIndex);
            logitsPtrsHostData[bi] = ptrToLogitsForBatchIndex;
        }
    }

    auto const* inputLengths = bufferCastOrNull<SizeType32>(params->inputLengths);
    auto embeddingBias = bufferCastOrNull<T>(params->embeddingBias);
    auto const* batchSlotsHostPtr = bufferCast<SizeType32>(*params->batchSlots);
#define GET_PENALTIES(capital_name, type)                                                                              \
    (mUse##capital_name                                                                                                \
        && !allOfBatchSlots(batchSlotsHostPtr, bufferCast<type>(*m##capital_name), localDecoderDomain.getBatchSize(),  \
            DefaultDecodingParams::get##capital_name()))                                                               \
        ? m##capital_name##Device                                                                                      \
        : nullptr;

    auto temperatures = GET_PENALTIES(Temperature, float);
    auto repetitionPenalties = GET_PENALTIES(RepetitionPenalty, float);
    auto presencePenalties = GET_PENALTIES(PresencePenalty, float);
    auto frequencyPenalties = GET_PENALTIES(FrequencyPenalty, float);
    auto minLengths = GET_PENALTIES(MinLength, SizeType32);

#undef GET_PENALTIES

    auto* const tokensPerStep = bufferCastOrNull<SizeType32>(params->curTokensPerStep);

    InvokeBatchApplyPenaltyParams<T> penaltyParams{};

    TensorPtr logitsPtrsHostSlice = ITensor::slice(logitsPtrsHost, 0, localDecoderDomain.getBatchSize());
    auto [logitsPtrsDeviceSlice] = workspace->mirrorInWorkspace(logitsPtrsHostSlice);
    auto runtimeLogits = workspace->getDeviceRuntimeLogits();
    penaltyParams.inputLogits = reinterpret_cast<T const* const*>(bufferCast<T const*>(*logitsPtrsDeviceSlice));
    penaltyParams.outputLogits = bufferCast<T>(*runtimeLogits);
    penaltyParams.biases = embeddingBias;
    penaltyParams.penaltyWorkspace = bufferCastOrNull<TokenIdType>(mPenaltyWorkspaceDevice);
    penaltyParams.penaltyWorkspacePrev = bufferCastOrNull<TokenIdType>(mPenaltyWorkspacePrevDevice);
    penaltyParams.temperatures = bufferCastOrNull<float>(temperatures);
    penaltyParams.repetitionPenalties = bufferCastOrNull<float>(repetitionPenalties);
    penaltyParams.presencePenalties = bufferCastOrNull<float>(presencePenalties);
    penaltyParams.frequencyPenalties = bufferCastOrNull<float>(frequencyPenalties);
    penaltyParams.batchSize = localDecoderDomain.getBatchSize();
    penaltyParams.beamWidth = localDecoderDomain.getBeamWidth();
    penaltyParams.maxSeqLen = maxSeqLen;
    penaltyParams.vocabSize = mDecoderDomain.getVocabSize();
    penaltyParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    penaltyParams.outputIdsPtr = bufferCast<TokenIdType const*>(*outputs->outputIdsPtr);
    penaltyParams.parentIdsPtr = bufferCast<SizeType32 const*>(*outputs->parentIdsPtr);
    penaltyParams.inputLengths = inputLengths;
    penaltyParams.sequenceLengths = bufferCast<SizeType32>(*outputs->sequenceLength.value());
    penaltyParams.minLengths = bufferCastOrNull<SizeType32>(minLengths);
    penaltyParams.endIds = bufferCast<TokenIdType>(*params->endIds);
    penaltyParams.batchSlots = workspace->getDeviceBatchSlotsPtr();
    penaltyParams.maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    penaltyParams.tokensPerStep = tokensPerStep;
    penaltyParams.finished = (params->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCast<FinishedState::UnderlyingType>(*params->finished.value()))
        : nullptr;
    penaltyParams.stream = getStream();

    if (penaltyParams.beamWidth > 1)
    {
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logitsPtrs = const_cast<T**>(penaltyParams.inputLogits);
        biasSoftmaxParams.bias = penaltyParams.biases;
        biasSoftmaxParams.endIds = penaltyParams.endIds;
        biasSoftmaxParams.batchSlots = penaltyParams.batchSlots;
        biasSoftmaxParams.batchSize = penaltyParams.batchSize;
        biasSoftmaxParams.maxBatchSize = mDecoderDomain.getBatchSize();
        biasSoftmaxParams.maxBeamWidth = penaltyParams.beamWidth;
        biasSoftmaxParams.vocabSize = penaltyParams.vocabSize;
        biasSoftmaxParams.vocabSizePadded = penaltyParams.vocabSizePadded;
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = penaltyParams.batchSlots != nullptr;
        biasSoftmaxParams.checkParams();
        invokeAddBiasSoftMax(biasSoftmaxParams, penaltyParams.stream);
    }

    invokeBatchApplyPenalty(penaltyParams);
    sync_check_cuda_error();

    mCyclicStep += 1;

    auto const logitsShape = ITensor::makeShape({localDecoderDomain.getBatchSize(),
        mDecoderDomain.getMaxDecodingTokens(), localDecoderDomain.getBeamWidth(), mDecoderDomain.getVocabSizePadded()});
    params->logits = ITensor::view(runtimeLogits, logitsShape);

    if (mDecodingMode.isBeamSearch())
    {
        std::swap(mPenaltyWorkspaceDevice, mPenaltyWorkspacePrevDevice);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class PenaltyLayer<float>;
template class PenaltyLayer<half>;

}
