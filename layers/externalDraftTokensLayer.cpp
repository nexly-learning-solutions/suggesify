
#include "externalDraftTokensLayer.h"
#include "../common/cudaUtils.h"
#include "../common/nvtxUtils.h"
#include "../src/decodingCommon.h"
#include "../src/samplingTopKKernels.h"
#include "../src/samplingTopPKernels.h"
#include "../src/speculativeDecoding/externalDraftTokensKernels.h"
#include "defaultDecodingParams.h"
#include "layerUtils.h"
#include "runtimeKernels.h"

#include <algorithm>

namespace tksd = suggestify::kernels::speculative_decoding;

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
ExternalDraftTokensLayer<T>::ExternalDraftTokensLayer(executor::DecodingMode const& mode,
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager, bool isDeterministic,
    bool isAirTopP)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
    , mIsDeterministic(isDeterministic)
    , mIsAirTopP(isAirTopP)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK_WITH_INFO(!mDecodingMode.isBeamSearch(), "ExternalDraftTokensLayer does not support Beam search mode");

    auto const deviceId = getDevice();
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProp, deviceId));

    allocateBuffer(decoderDomain.getBatchSize());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto workspaceSize = getTopKWorkspaceSize<T>(batchSize, 1, TOP_K_MAX, mDecoderDomain.getVocabSizePadded());
    mWorkspaceSize = std::max(workspaceSize, mWorkspaceSize);
    workspaceSize = getTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());
    mWorkspaceSize = std::max(workspaceSize, mWorkspaceSize);

    workspaceSize = getAirTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded(), mIsDeterministic);
    mWorkspaceSize = std::max(workspaceSize, mWorkspaceSize);

    auto const batchSizeShape = ITensor::makeShape({batchSize});

    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize, sizeof(curandState_t)}), TRTDataType<int8_t>::value);
    mBatchIsAccepted = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mRuntimeMultinomialDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);

    mSkipTopKDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mSkipTopKDecodeHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    mSkipTopPDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mSkipTopPDecodeHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    auto skipTopPDecodeHostRange = BufferRange<bool>(*mSkipTopPDecodeHost);
    std::fill(skipTopPDecodeHostRange.begin(), skipTopPDecodeHostRange.end(), true);

    mOutputIdsAfterSampling = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getVocabSizePadded()}), TRTDataType<TokenIdType>::value);
    mTargetOutputIds = mBufferManager->gpu(ITensor::makeShape({batchSize}), TRTDataType<TokenIdType>::value);

    mRuntimeTopKDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mRuntimeTopKHost = mBufferManager->cpu(batchSizeShape, TRTDataType<SizeType32>::value);

    mRuntimeTopPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);

    mMaskBuffer = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getVocabSizePadded()}), TRTDataType<bool>::value);

    mSetupWorkspaceSize = std::max({mBatchIsAccepted->getSizeInBytes(), mRuntimeMultinomialDevice->getSizeInBytes(),
        mOutputIdsAfterSampling->getSizeInBytes(), mTargetOutputIds->getSizeInBytes(), mMaskBuffer->getSizeInBytes()});

    mTargetLogits = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getVocabSizePadded()}), TRTDataType<T>::value);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<ExternalDraftTokensSetupParams>(baseSetupParams);

    workspace->initializeDeviceCurandStates(
        setupParams->randomSeed, batchSize, workspace->getDeviceBatchSlots(), mCurandStatesDevice);

    auto& runtimeMultinomialDeviceTensor = const_cast<ITensor&>(*mRuntimeMultinomialDevice);
    suggestify::runtime::kernels::invokeFill(runtimeMultinomialDeviceTensor, 1.0f, mBufferManager->getStream());

    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector{DefaultDecodingParams::getTopK()});
    auto runtimeTopP = setupParams->runtimeTopP.value_or(std::vector{DefaultDecodingParams::getTopP()});

    auto const paramsSize = expandMatchElements(batchSize, runtimeTopK, runtimeTopP);

    CHECK_WITH_INFO(paramsSize != 0,
        fmtstr("ExternalDraftTokensLayer got parameter with unexpected size, want 1 or batchSize(%d), got"
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
        calcAlignedPointers(workspace->getRawWorkspaceDevicePtr(), initWorkspaceSizes)(topKsPtr, topPsPtr);
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopK, IBuffer::wrap(topKsPtr, initWorkspaceSizes[0] / sizeof(*topKsPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopP, IBuffer::wrap(topPsPtr, initWorkspaceSizes[1] / sizeof(*topPsPtr)));
    }
    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();
    auto* skipTopKDecodeDevicePtr = bufferCastOrNull<bool>(mSkipTopKDecodeDevice);
    auto* skipTopPDecodeDevicePtr = bufferCastOrNull<bool>(mSkipTopPDecodeDevice);
    invokeSetupTopKTopPRuntimeArgs(batchSize,
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKDevice)},
        {topPsPtr, runtimeTopP.front(), bufferCast<float>(*mRuntimeTopPDevice)},
        skipTopKDecodeDevicePtr, skipTopPDecodeDevicePtr, batchSlotsDevicePtr, true, getStream());

    auto const* batchSlotsHostPtr = bufferCast<SizeType32>(*batchSlots);
    auto* skipDecodeTopKHostPtr = bufferCastOrNull<bool>(mSkipTopKDecodeHost);
    auto* skipDecodeTopPHostPtr = bufferCastOrNull<bool>(mSkipTopPDecodeHost);
    topKsPtr = paramsSize > 1 ? runtimeTopK.data() : nullptr;
    invokeSetupTopKTopPRuntimeArgs(batchSize,
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKHost)}, {},
        skipDecodeTopKHostPtr, skipDecodeTopPHostPtr, batchSlotsHostPtr, false);

    if (mIsAirTopP)
    {
        auto smCnt = mDeviceProp.multiProcessorCount;
        if (smCnt <= 0)
        {
            auto const deviceId = getDevice();
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
            smCnt = prop.multiProcessorCount;
        }
        mAirTopPBlockNum
            = calcAirTopPBlockNum<T>(batchSize, mDecoderDomain.getVocabSizePadded(), smCnt, mIsDeterministic);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto const* endIds = bufferCast<TokenIdType>(*inputs->endIds);

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCast<FinishedState::UnderlyingType>(*inputs->finished.value()))
        : nullptr;

    inputs->curandStates = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    inputs->probsComputed = true;

    auto runtimeLogitsPtr = bufferCast<T>(*workspace->getDeviceRuntimeLogits());
    auto logitsPtrsPtr = static_cast<T**>(nullptr);
    auto biasPtr = static_cast<T*>(nullptr);
    auto const* batchSlotsPtr = workspace->getDeviceBatchSlotsPtr();
    mBufferManager->copy(runtimeLogitsPtr, *mTargetLogits);

    {
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = runtimeLogitsPtr;
        biasSoftmaxParams.logitsPtrs = logitsPtrsPtr;
        biasSoftmaxParams.probs = runtimeLogitsPtr;
        biasSoftmaxParams.bias = biasPtr;
        biasSoftmaxParams.endIds = endIds;
        biasSoftmaxParams.finished = finishedInput;
        biasSoftmaxParams.batchSlots = batchSlotsPtr;
        biasSoftmaxParams.batchSize = batchSize;
        biasSoftmaxParams.maxBatchSize = mDecoderDomain.getBatchSize();
        biasSoftmaxParams.maxBeamWidth = 1;
        biasSoftmaxParams.vocabSize = mDecoderDomain.getVocabSize();
        biasSoftmaxParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = false;
        biasSoftmaxParams.checkParams();

        invokeAddBiasSoftMax(biasSoftmaxParams, getStream());
    }

    auto& outputIdsAfterSamplingTensor = const_cast<ITensor&>(*mOutputIdsAfterSampling);
    mBufferManager->setZero(outputIdsAfterSamplingTensor);

    getAllTopKs(baseInputs, workspace);

    getAllTopPs(baseInputs, workspace);

    acceptDraftTokens(outputs, baseInputs, workspace);

    multinomialSampling(outputs, baseInputs, workspace);

    forwardAcceptedTokens(outputs, baseInputs, workspace);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t ExternalDraftTokensLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mWorkspaceSize, mSetupWorkspaceSize);
}

template <typename T>
void ExternalDraftTokensLayer<T>::acceptDraftTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_acceptDraftTokens);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto const draftLogitsShape = (*inputs->draftLogits).getShape();
    auto const maxBatchSize = mDecoderDomain.getBatchSize();
    auto const maxTokensPerStep = draftLogitsShape.d[1];
    auto const batchSize = static_cast<SizeType32>(inputs->logits.value()->getDimension<0>());
    auto constexpr beamWidth = 1;

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished))
        : nullptr;

    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    tksd::invokeMaskTargetLogits(batchSize, bufferCast<T>(*mTargetLogits), workspace->getDeviceBatchSlotsPtr(),
        beamWidth, mDecoderDomain.getVocabSizePadded(), finishedInput, maxBatchSize,
        bufferCast<SizeType32>(*mOutputIdsAfterSampling), bufferCastOrNull<SizeType32>(mRuntimeTopKDevice),
        bufferCast<bool>(*mMaskBuffer), getStream());

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto const* useDraftLogitsHostPtr = bufferCastOrNull<bool>(inputs->useDraftLogitsHost);
    auto const skipDraftLogits = allOfBatchSlots(batchSlotsHost, useDraftLogitsHostPtr, batchSize, false);

    if (!skipDraftLogits && inputs->step == 0)
    {
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = bufferCast<T>(*inputs->draftLogits);
        biasSoftmaxParams.probs = bufferCast<T>(*inputs->draftProbs);
        biasSoftmaxParams.finished = finishedInput;
        biasSoftmaxParams.batchSlots = workspace->getDeviceBatchSlotsPtr();
        biasSoftmaxParams.batchSize = batchSize;
        biasSoftmaxParams.maxBatchSize = maxBatchSize;
        biasSoftmaxParams.maxBeamWidth = beamWidth * maxTokensPerStep;
        biasSoftmaxParams.vocabSize = mDecoderDomain.getVocabSize();
        biasSoftmaxParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = true;
        biasSoftmaxParams.checkParams();
        invokeAddBiasSoftMax(biasSoftmaxParams, getStream());
    }

    {
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = bufferCast<T>(*mTargetLogits);
        biasSoftmaxParams.probs = bufferCast<T>(*inputs->targetProbs);
        biasSoftmaxParams.finished = finishedInput;
        biasSoftmaxParams.batchSlots = workspace->getDeviceBatchSlotsPtr();
        biasSoftmaxParams.batchSize = batchSize;
        biasSoftmaxParams.maxBatchSize = maxBatchSize;
        biasSoftmaxParams.maxBeamWidth = beamWidth;
        biasSoftmaxParams.vocabSize = mDecoderDomain.getVocabSize();
        biasSoftmaxParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = false;
        biasSoftmaxParams.checkParams();
        invokeAddBiasSoftMax(biasSoftmaxParams, getStream());
    }

    sync_check_cuda_error();

    tksd::invokeAcceptDraftTokens(batchSize, bufferCast<T>(*inputs->draftProbs), bufferCast<T>(*inputs->targetProbs),
        bufferCast<SizeType32>(*inputs->numDraftTokens), bufferCast<bool>(*inputs->useDraftLogits),
        bufferCast<TokenIdType>(*inputs->draftTokenIds), finishedInput, finishedOutput, inputs->curandStates,
        workspace->getDeviceBatchSlotsPtr(), maxTokensPerStep, beamWidth, mDecoderDomain.getVocabSizePadded(),
        inputs->useRandomAcceptanceThreshold, inputs->constantThreshold, inputs->step,
        bufferCast<bool>(*mBatchIsAccepted), bufferCast<SizeType32>(*mTargetOutputIds), getStream());

    sync_check_cuda_error();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::multinomialSampling(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_multinomialSampling);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();
    auto probs = bufferCastOrNull<T>(inputs->targetProbs);
    auto* sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;
    TopPSamplingKernelParams<T> params{};

    params.probs = probs;
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.topPs = bufferCastOrNull<float>(mRuntimeMultinomialDevice);
    params.sequenceLength = sequenceLength;
    params.endIds = endIds;
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = nullptr;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mBatchIsAccepted);
    params.cumLogProbs = nullptr;
    params.outputLogProbs = nullptr;
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    if (!mIsAirTopP)
    {
        invokeBatchTopPSampling<T>(params, getStream());
    }
    else
    {
        params.blockNum = mAirTopPBlockNum;
        params.isDeterministic = mIsDeterministic;
        invokeBatchAirTopPSampling<T>(params, getStream());
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::getAllTopKs(std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_getAllTopKs);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto logits = bufferCastOrNull<T>(inputs->logits);

    auto const batchSize = static_cast<SizeType32>(inputs->logits.value()->getDimension<0>());

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto const* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipTopKDecodeHost);
    auto const skip = allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished))
        : nullptr;

    auto const* runtimeTopKHostPtr = bufferCast<SizeType32>(*mRuntimeTopKHost);

    TopKSamplingKernelParams<T> params{};
    params.logProbs = logits;
    params.outputIds = bufferCastOrNull<TokenIdType>(mOutputIdsAfterSampling);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.maxTopP = 1.0F;
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.maxTopK = maxOfBatchSlots(batchSlotsHost, runtimeTopKHostPtr, batchSize);
    params.topKs = bufferCastOrNull<SizeType32>(mRuntimeTopKDevice);
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipTopKDecodeDevice);
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.returnAllSelectedTokens = true;
    params.maxSeqLen = mDecoderDomain.getVocabSizePadded();
    params.logitsHasProbs = inputs->probsComputed;
    params.outputIdCurrentStep = bufferCastOrNull<TokenIdType>(mTargetOutputIds);
    params.skipOutputIdCurrentStep = bufferCast<bool>(*inputs->useDraftLogits);

    invokeBatchTopKSampling(params, getStream());
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::getAllTopPs(std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_getAllTopPs);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto logits = bufferCastOrNull<T>(inputs->logits);

    auto const batchSize = static_cast<SizeType32>(inputs->logits.value()->getDimension<0>());

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto const* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipTopPDecodeHost);
    auto const skip = allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished))
        : nullptr;

    TopPSamplingKernelParams<T> params{};
    params.probs = logits;
    params.outputIds = bufferCastOrNull<TokenIdType>(mOutputIdsAfterSampling);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipTopPDecodeDevice);
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.returnAllSelectedTokens = true;
    params.maxSeqLen = mDecoderDomain.getVocabSizePadded();
    params.outputIdCurrentStep = bufferCastOrNull<TokenIdType>(mTargetOutputIds);
    params.skipOutputIdCurrentStep = bufferCast<bool>(*inputs->useDraftLogits);

    invokeBatchTopPSampling<T>(params, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::forwardAcceptedTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_forwardAcceptedTokens);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);
    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto const draftLogitsShape = (*inputs->draftLogits).getShape();
    auto const maxTokensPerStep = draftLogitsShape.d[1];

    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    tksd::invokeForwardAcceptedTokens(batchSize, workspace->getDeviceBatchSlotsPtr(),
        bufferCast<bool>(*mBatchIsAccepted), bufferCastOrNull<SizeType32>(outputs->sequenceLength),
        bufferCast<TokenIdType>(*inputs->draftTokenIds), bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr),
        inputs->step, maxTokensPerStep, bufferCastOrNull<TokenIdType>(inputs->endIds), finishedOutput, getStream());

    sync_check_cuda_error();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class ExternalDraftTokensLayer<float>;
template class ExternalDraftTokensLayer<half>;

}
