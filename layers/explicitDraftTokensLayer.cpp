
#include "explicitDraftTokensLayer.h"
#include "suggestify/kernels/penaltyTypes.h"
#include "suggestify/kernels/speculativeDecoding/common.h"
#include "suggestify/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "defaultDecodingParams.h"
#include "layerUtils.h"

#include <algorithm>

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::kernels::speculative_decoding;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
ExplicitDraftTokensLayer<T>::ExplicitDraftTokensLayer(
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::allocateBuffer()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mTemperature
        = mBufferManager->pinnedPool(ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<float>::value);

    mScanWorkspaceSizeInBytes = invokeScanGenerationLengths(
        nullptr, mScanWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), getStream());
    mReduceWorkspaceSizeInBytes = invokeReduceMaxGenerationLengths(
        nullptr, mReduceWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), getStream());

    mWorkspaceSize = std::max(mScanWorkspaceSizeInBytes, mReduceWorkspaceSizeInBytes);

    mCurandStatesDevice = mBufferManager->gpu(
        ITensor::makeShape({mDecoderDomain.getBatchSize(), sizeof(curandState_t)}), TRTDataType<int8_t>::value);
    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mGenerationLengthInclusiveSum = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mMaxGenerationLength = mBufferManager->gpu(ITensor::makeShape({1}), TRTDataType<SizeType32>::value);
    mTemperatureDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mBestPathIndicesSlots = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mLastDraftIndicesSlots = mBufferManager->gpu(ITensor::makeShape({mDecoderDomain.getBatchSize()
                                                     * mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths()
                                                     * mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen()}),
        TRTDataType<SizeType32>::value);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<ExplicitDraftTokensSetupParams>(baseSetupParams);
    workspace->initializeDeviceCurandStates(
        setupParams->randomSeed, batchSize, workspace->getDeviceBatchSlots(), mCurandStatesDevice);

    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};

    if (!mDecoderDtype)
    {
        mDecoderDtype = setupParams->dtype;
    }

    fillBuffers(setupParams->temperature, DefaultDecodingParams::getTemperature(), mTemperature, mTemperatureDevice,
        batchSlots, getLimitsPenalty(DecodingPenaltyType::Temperature), "temperature penalty");

    if (mDecoderDtype == nvinfer1::DataType::kFLOAT)
    {
        fillContextBuffers<float>(batchSize, batchSlots, *setupParams, workspace);
    }
    else if (mDecoderDtype == nvinfer1::DataType::kHALF)
    {
        fillContextBuffers<half>(batchSize, batchSlots, *setupParams, workspace);
    }
    else if (mDecoderDtype == nvinfer1::DataType::kBF16)
    {
        fillContextBuffers<__nv_bfloat16>(batchSize, batchSlots, *setupParams, workspace);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<ExplicitDraftTokensInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<ExplicitDraftTokensOutputs>(baseOutputs);


    convertPackedMask(*outputs, *inputs, workspace);

    if (mDecoderDtype == nvinfer1::DataType::kFLOAT)
    {
        splitInputDataToBatchSlots<float>(*outputs, *inputs, workspace);
    }
    else if (mDecoderDtype == nvinfer1::DataType::kHALF)
    {
        splitInputDataToBatchSlots<half>(*outputs, *inputs, workspace);
    }
    else if (mDecoderDtype == nvinfer1::DataType::kBF16)
    {
        splitInputDataToBatchSlots<__nv_bfloat16>(*outputs, *inputs, workspace);
    }

    packAcceptedPaths(*outputs, *inputs, workspace);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t ExplicitDraftTokensLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template <typename T>
template <typename Dtype>
void ExplicitDraftTokensLayer<T>::fillContextBuffers(SizeType32 batchSize, BufferConstPtr batchSlots,
    ExplicitDraftTokensSetupParams const& setupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    FillContextExplicitDraftTokensParams<Dtype> params;
    params.randDataSample = bufferCast<Dtype>(*setupParams.randomDataSample);
    params.outputTemperatures = bufferCast<Dtype>(*setupParams.temperatures);
    params.inputTemperatures = bufferCastOrNull<float>(mTemperatureDevice);
    params.curandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesDevice));
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.batchSize = batchSize;

    params.checkParams();

    invokeFillContextBuffers(params, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
template <typename Dtype>
void ExplicitDraftTokensLayer<T>::splitInputDataToBatchSlots(ExplicitDraftTokensOutputs const& outputs,
    ExplicitDraftTokensInputs const& inputs, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;
    auto const maxSeqLen = outputs.outputIds->getDimension<-1>();

    ExtractExplicitDraftTokensParams<Dtype> params;

    params.outputIds = bufferCast<TokenIdType>(*outputs.outputIds);
    params.outputPositionIdsBase = bufferCast<SizeType32>(*outputs.positionIdsBase);
    params.outputPositionIds = bufferCast<SizeType32>(*outputs.nextDraftPosIds);
    params.outputNextDraftTokens = bufferCast<TokenIdType>(*outputs.nextDraftTokens);
    params.unpackedNextDraftTokens = bufferCast<TokenIdType>(*outputs.unpackedNextDraftTokens);
    params.unpackedNextDraftIndices = bufferCast<SizeType32>(*outputs.unpackedNextDraftIndices);
    params.acceptedLengths = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    params.nextDraftLengths = bufferCast<SizeType32>(*outputs.nextDraftLengths);
    params.prevDraftLengths = bufferCast<SizeType32>(*outputs.prevDraftLengths);
    params.sequenceLengths = bufferCast<SizeType32>(*outputs.sequenceLength.value());
    params.randDataSample = bufferCast<Dtype>(*outputs.randomDataSample);
    params.randDataVerification = bufferCast<Dtype>(*outputs.randomDataValidation);
    params.outputDraftProbs = bufferCast<Dtype>(*outputs.nextDraftProbs);
    params.outputTemperatures = bufferCast<Dtype>(*outputs.temperatures);
    params.outputGenerationLengths = bufferCast<SizeType32>(*outputs.generationLengths);
    params.outputBestPathIndices = bufferCast<SizeType32>(*mBestPathIndicesSlots);
    params.outputLastDraftIndices = bufferCast<SizeType32>(*mLastDraftIndicesSlots);

    params.batchSlots = bufferCast<SizeType32>(*inputs.seqSlots);
    params.nextDraftTokens = bufferCast<TokenIdType>(*inputs.nextDraftTokens);
    params.lastDraftTokens = bufferCast<TokenIdType>(*inputs.lastDraftTokens);
    params.inputUnpackedNextDraftIndices = bufferCast<SizeType32>(*inputs.nextDraftIndices);
    params.bestPathLengths = bufferCast<SizeType32>(*inputs.bestPathLengths);
    params.bestPathIndices = bufferCast<SizeType32>(*inputs.bestPathIndices);
    params.inputPositionIdsBase = bufferCast<SizeType32>(*inputs.positionIdsBase);
    params.packedPositionIds = bufferCast<SizeType32>(*inputs.packedPosIds);
    params.nextFlatTokens = bufferCast<TokenIdType>(*inputs.nextFlatTokens);
    params.nextDraftProbs = bufferCast<Dtype>(*inputs.nextDraftProbs);
    params.lastGenerationLengths = bufferCastOrNull<SizeType32>(inputs.lastGenerationLengths);
    params.generationLengthInclusiveSum = bufferCast<SizeType32>(*mGenerationLengthInclusiveSum);
    params.lastDraftIndices = bufferCast<SizeType32>(*inputs.lastDraftIndices);
    params.inputTemperatures = bufferCast<float>(*mTemperatureDevice);
    params.curandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesDevice));
    params.batchSize = batchSize;
    params.numPaths = mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths();
    params.maxPathLength = mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen();
    params.maxSeqLen = maxSeqLen;
    params.vocabSize = mDecoderDomain.getVocabSizePadded();
    params.numContextRequests = batchSize - inputs.lastDraftTokens->getDimension<0>();
    params.numGenerationRequests = inputs.lastDraftTokens->getDimension<0>();

    params.checkParams();

    mBufferManager->copy(*inputs.maxGenLengthDevice, *outputs.maxGenLengthHost);

    invokeExtractExplicitDraftTokens(params, getStream());

    invokeCopyProbs(params, getStream());

    mBufferManager->copy(*outputs.generationLengths, *outputs.generationLengthsHost);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::convertPackedMask(ExplicitDraftTokensOutputs const& outputs,
    ExplicitDraftTokensInputs const& inputs, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlots = bufferCast<SizeType32>(*inputs.seqSlots);
    auto masksDevice = bufferCast<bool>(*inputs.masks);
    auto generationLengths = bufferCast<SizeType32>(*inputs.generationLengths);
    auto packedMasksDevice = bufferCast<SizeType32>(*outputs.packedMasks);

    auto const batchSize = inputs.localBatchSize;

    auto generationLengthInclusiveSumPtr = bufferCastOrNull<SizeType32>(mGenerationLengthInclusiveSum);
    auto workSpaceDevicePtr = workspace->getRawWorkspaceDevicePtr();
    auto maxGenerationLengthPtr = bufferCastOrNull<SizeType32>(mMaxGenerationLength);
    invokeScanReduceGenerationLengths(batchSize, generationLengths, workSpaceDevicePtr, mScanWorkspaceSizeInBytes,
        generationLengthInclusiveSumPtr, workSpaceDevicePtr, mReduceWorkspaceSizeInBytes, maxGenerationLengthPtr,
        getStream());

    invokeConvertMaskToPackedMask(batchSize, generationLengthInclusiveSumPtr, maxGenerationLengthPtr, masksDevice,
        batchSlots, mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingDraftTokens(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingTokens(), packedMasksDevice, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::packAcceptedPaths(ExplicitDraftTokensOutputs const& outputs,
    ExplicitDraftTokensInputs const& inputs, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;

    auto numNewTokens = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    auto numNewTokensCumSum = bufferCast<SizeType32>(*outputs.numNewTokensCumSum);
    auto pathsOffsets = bufferCast<SizeType32>(*outputs.pathsOffsets);
    auto batchSlots = workspace->getDeviceBatchSlotsPtr();
    auto bestPathIndicesSlotsPtr = bufferCastOrNull<SizeType32>(mBestPathIndicesSlots);
    auto lastDraftIndicesSlotsPtr = bufferCastOrNull<SizeType32>(mLastDraftIndicesSlots);

    CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for ExplicitDraftTokensLayer");
    CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for ExplicitDraftTokensLayer");
    CHECK_WITH_INFO(
        numNewTokensCumSum != nullptr, "numNewTokensCumSum must be provided for ExplicitDraftTokensLayer");
    CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for ExplicitDraftTokensLayer");
    invokePackAcceptedPaths(numNewTokensCumSum, pathsOffsets, numNewTokens, bestPathIndicesSlotsPtr,
        lastDraftIndicesSlotsPtr, batchSlots, batchSize, batchSize,
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen(), false, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class ExplicitDraftTokensLayer<float>;
template class ExplicitDraftTokensLayer<half>;

}
