
#include "topPSamplingLayer.h"
#include "suggestify/common/logger.h"
#include "suggestify/common/memoryUtils.h"
#include "suggestify/common/nvtxUtils.h"
#include "../src/decodingCommon.h"
#include "../src/samplingTopKKernels.h"
#include "../src/samplingTopPKernels.h"
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
TopPSamplingLayer<T>::TopPSamplingLayer(DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager, bool isDeterministic, bool isAirTopP)
    : BaseLayer(decoderDomain, bufferManager)
    , mIsDeterministic(isDeterministic)
    , mIsAirTopP(isAirTopP)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const deviceId = getDevice();
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProp, deviceId));

    allocateBuffer(mDecoderDomain.getBatchSize());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (!mIsAirTopP)
    {
        mWorkspaceSize = getTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());
    }
    else
    {
        mWorkspaceSize = getAirTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded(), mIsDeterministic);
    }

    auto const batchSizeShape = ITensor::makeShape({batchSize});
    mRuntimeTopKDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mRuntimeTopPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mInitialTopPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mTopPDecayDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mTopPMinDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mTopPResetIdsDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<TokenIdType>::value);
    mSkipDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mSkipDecodeHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    auto skipDecodeHostRange = BufferRange<bool>(*mSkipDecodeHost);
    std::fill(skipDecodeHostRange.begin(), skipDecodeHostRange.end(), true);

    mSetupWorkspaceSize = std::max({mRuntimeTopKDevice->getSizeInBytes(), mRuntimeTopPDevice->getSizeInBytes(),
        mInitialTopPDevice->getSizeInBytes(), mTopPDecayDevice->getSizeInBytes(), mTopPMinDevice->getSizeInBytes(),
        mTopPResetIdsDevice->getSizeInBytes(), mSkipDecodeDevice->getSizeInBytes()});

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopPSamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);

    auto constexpr defaultTopPDecay = DefaultDecodingParams::getTopPDecay();
    auto constexpr defaultTopPMin = DefaultDecodingParams::getTopPMin();

    auto const* batchSlotsHostPtr = bufferCastOrNull<SizeType32>(batchSlots);
    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    if (!setupParams->runtimeTopP.has_value() || setupParams->runtimeTopP.value().empty())
    {
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const bid = batchSlotsHostPtr[bi];
            skipDecodeHostPtr[bid] = true;
        }
        auto const maxBatchSize = mDecoderDomain.getBatchSize();
        auto skipDecodeHostSlice = IBuffer::slice(mSkipDecodeHost, 0, maxBatchSize);
        mBufferManager->copy(*skipDecodeHostSlice, *mSkipDecodeDevice);
        return;
    }

    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector{DefaultDecodingParams::getTopK()});
    auto runtimeTopP = setupParams->runtimeTopP.value();
    auto decayVec = setupParams->topPDecay.value_or(std::vector{defaultTopPDecay});
    auto topPMinVec = setupParams->topPMin.value_or(std::vector{defaultTopPMin});
    auto topPResetIdsVec = setupParams->topPResetIds.value_or(std::vector{DefaultDecodingParams::getTopPResetId()});

    auto const paramsSize
        = expandMatchElements(batchSize, runtimeTopK, runtimeTopP, decayVec, topPMinVec, topPResetIdsVec);
    CHECK_WITH_INFO(paramsSize != 0,
        fmtstr("TopPSamplingLayer got parameter with unexpected size, want 1 or batchSize(%d), got"
               "runtimeTopK.size() = %zu, "
               "runtimeTopP.size() = %zu, "
               "topPDecay.size() = %zu, "
               "topPMin.size() = %zu, "
               "topPResetIds.size() = %zu",
            batchSize, runtimeTopK.size(), runtimeTopP.size(), decayVec.size(), topPMinVec.size(),
            topPResetIdsVec.size()));

    for (size_t i = 0; i < paramsSize; ++i)
    {
        auto& topK = runtimeTopK[i];
        auto& topP = runtimeTopP[i];
        clampTopK(topK);
        clampTopP(topP);
        regularizeTopKTopP(topK, topP);

        auto& decay = decayVec[i];
        if (decay <= 0.f || decay > 1.0f)
        {
            LOG_WARNING(
                "Decay (%f) is out of range ((0.0, 1.0f]). Change to default (%f).", decay, defaultTopPDecay);
            decay = defaultTopPDecay;
        }

        auto& topPMin = topPMinVec[i];
        if (topPMin <= 0.f || topPMin > 1.0f)
        {
            LOG_WARNING(
                "TopP min (%f) is out of range ([0.0, 1.0f]). Change to default (%f).", topPMin, defaultTopPMin);
            topPMin = defaultTopPMin;
        }
    }

    SizeType32* topKsPtr = nullptr;
    float* topPsPtr = nullptr;
    float* topPDecayPtr = nullptr;
    float* topPMinPtr = nullptr;
    SizeType32* topPResetIdsPtr = nullptr;

    if (paramsSize > 1)
    {
        auto initWorkspaceSizes = getTopPInitWorkspaceSizes(batchSize);
        std::vector<void*> alignedPointers;
        calcAlignedPointers(workspace->getRawWorkspaceDevicePtr(), initWorkspaceSizes)(
            topKsPtr, topPsPtr, topPDecayPtr, topPMinPtr, topPResetIdsPtr);
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopK, IBuffer::wrap(topKsPtr, initWorkspaceSizes[0] / sizeof(*topKsPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopP, IBuffer::wrap(topPsPtr, initWorkspaceSizes[1] / sizeof(*topPsPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, decayVec, IBuffer::wrap(topPDecayPtr, initWorkspaceSizes[2] / sizeof(*topPDecayPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, topPMinVec, IBuffer::wrap(topPMinPtr, initWorkspaceSizes[3] / sizeof(*topPMinPtr)));
        DecodingLayerWorkspace::copyToWorkspace(*mBufferManager, topPResetIdsVec,
            IBuffer::wrap(topPResetIdsPtr, initWorkspaceSizes[4] / sizeof(*topPResetIdsPtr)));
    }

    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();
    auto* skipDecodeDevicePtr = bufferCastOrNull<bool>(mSkipDecodeDevice);
    auto* initialTopPDevicePtr = bufferCast<float>(*mInitialTopPDevice);
    invokeSetTopPRuntimeArgs(batchSize,
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKDevice)},
        {topPsPtr, runtimeTopP.front(), bufferCast<float>(*mRuntimeTopPDevice)},
        skipDecodeDevicePtr, initialTopPDevicePtr, batchSlotsDevicePtr, true, getStream());

    invokeScatterDecodingParams(topPDecayPtr, decayVec.front(), bufferCast<float>(*mTopPDecayDevice),
        batchSlotsDevicePtr, batchSize, getStream());
    invokeScatterDecodingParams(topPMinPtr, topPMinVec.front(), bufferCast<float>(*mTopPMinDevice), batchSlotsDevicePtr,
        batchSize, getStream());
    invokeScatterDecodingParams(topPResetIdsPtr, topPResetIdsVec.front(), bufferCast<TokenIdType>(*mTopPResetIdsDevice),
        batchSlotsDevicePtr, batchSize, getStream());

    topKsPtr = paramsSize > 1 ? runtimeTopK.data() : nullptr;
    invokeSetTopPRuntimeArgs(batchSize,
        {topKsPtr, runtimeTopK.front(), nullptr}, {},
        skipDecodeHostPtr, nullptr, batchSlotsHostPtr, false);

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
void TopPSamplingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(TopPSamplingLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipDecodeHost);
    auto const skip = allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    auto probs = bufferCastOrNull<T>(inputs->logits);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    auto const* finishedInput = (inputs->finished) ? reinterpret_cast<FinishedState const*>(
                                    bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished.value()))
                                                   : nullptr;
    auto* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished.value()))
        : nullptr;

    auto* cumLogProbs = bufferCastOrNull<float>(outputs->cumLogProbs);
    auto* outputLogProbs = bufferCastOrNull<float>(outputs->outputLogProbsTiled);
    auto* sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);

    TopPSamplingKernelParams<T> params{};
    params.probs = probs;
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.sequenceLength = sequenceLength;
    params.endIds = endIds;
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipDecodeDevice);
    params.cumLogProbs = cumLogProbs;
    params.outputLogProbs = outputLogProbs;
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

    sync_check_cuda_error();
    auto* runtimeTopPDevicePtr = bufferCastOrNull<float>(mRuntimeTopPDevice);
    auto* initialTopPDevicePtr = bufferCastOrNull<float>(mInitialTopPDevice);
    auto* topPDecayDevicePtr = bufferCastOrNull<float>(mTopPDecayDevice);
    auto* topPMinDevicePtr = bufferCastOrNull<float>(mTopPMinDevice);
    auto* topPResetIdsDevice = bufferCastOrNull<TokenIdType>(mTopPResetIdsDevice);
    auto* outputIdsPtr = bufferCastOrNull<TokenIdType const*>(outputs->outputIdsPtr);
    invokeComputeToppDecay(runtimeTopPDevicePtr, initialTopPDevicePtr, outputIdsPtr, topPDecayDevicePtr,
        topPMinDevicePtr, topPResetIdsDevice, sequenceLength, workspace->getDeviceBatchSlotsPtr(), batchSize,
        getStream());
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t TopPSamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mSetupWorkspaceSize, mWorkspaceSize);
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

}
