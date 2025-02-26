
#include "eagleDecodingLayer.h"
#include "suggestify/common/nvtxUtils.h"
#include "suggestify/common/workspace.h"
#include "../src/speculativeDecoding/common.h"
#include "../src/speculativeDecoding/eagleDecodingKernels.h"
#include "defaultDecodingParams.h"
#include "layerUtils.h"

using namespace suggestify::common;
using namespace suggestify::kernels;
using namespace suggestify::kernels::speculative_decoding;
using namespace suggestify::runtime;

namespace suggestify::layers
{

template <typename T>
EagleDecodingLayer<T>::EagleDecodingLayer(
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::allocateBuffer()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mTemperature = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mTemperatureDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mCurandStatesDevice = mBufferManager->gpu(
        ITensor::makeShape({mDecoderDomain.getBatchSize(), sizeof(curandState_t)}), TRTDataType<int8_t>::value);

    mEagleNetCtxRequestTypes = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mEagleNetCtxContextLengths = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mEagleNetCtxPastKeyValueLengths = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mEagleNetGenRequestTypes = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mEagleNetGenContextLengths = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mEagleNetGenPastKeyValueLengths = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);

    SizeType32 constexpr NUM_BUFFERS{2};
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = mDecoderDomain.getBatchSize() * sizeof(SizeType32);
    workspaces[1] = mDecoderDomain.getBatchSize() * sizeof(SizeType32);
    mWorkspaceSize = calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<EagleSetupParams>(baseSetupParams);
    workspace->initializeDeviceCurandStates(
        setupParams->randomSeed, batchSize, workspace->getDeviceBatchSlots(), mCurandStatesDevice);

    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};

    auto constexpr fltMax = std::numeric_limits<float>::max();
    auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();

    fillBuffers(setupParams->temperature, DefaultDecodingParams::getTemperature(), mTemperature, mTemperatureDevice,
        batchSlots, std::make_pair(-fltEpsilon, fltMax), "temperature penalty");

    fillContextBuffers(batchSize, batchSlots, *setupParams, workspace);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::fillContextBuffers(SizeType32 batchSize, BufferConstPtr batchSlots,
    EagleSetupParams const& setupParams, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    FillContextEagleParams params;
    params.outputRandDataSample = bufferCast<float>(*setupParams.randomDataSample);
    params.outputTemperatures = bufferCast<float>(*setupParams.temperatures);

    params.inputTemperatures = bufferCastOrNull<float>(mTemperatureDevice);
    params.inputCurandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesDevice));
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.batchSize = batchSize;

    params.checkParams();

    invokeFillContextEagleData(params, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(EagleDecodingLayer_forwardSyncCPU);

    auto inputs = std::dynamic_pointer_cast<EagleInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<EagleOutputs>(baseOutputs);

    augmentBatchSlots(*outputs, *inputs, workspace);

    unpackData(*outputs, *inputs, workspace);

    convertToPackedMask(*outputs, *inputs, workspace);

    packAcceptedPaths(*outputs, *inputs, workspace);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::augmentBatchSlots(EagleOutputs const& outputs, EagleInputs const& inputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;
    auto const engineBatchSize = inputs.nextDraftLens->getDimension<0>();

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace->getRawWorkspaceDevicePtr());
    size_t offset{0};

    SizeType32* augmentedSeqSlots = reinterpret_cast<SizeType32*>(
        nextWorkspacePtr(workspaceBytePtr, offset, engineBatchSize * sizeof(SizeType32)));
    SizeType32* augmentedBatchSlots = reinterpret_cast<SizeType32*>(
        nextWorkspacePtr(workspaceBytePtr, offset, engineBatchSize * sizeof(SizeType32)));

    auto chunkedContextNextTokens = bufferCast<SizeType32>(*inputs.chunkedContextNextTokens);
    auto lastDraftLens = bufferCast<SizeType32>(*inputs.lastDraftLens);

    invokeAugmentBatchSlots(augmentedSeqSlots, augmentedBatchSlots, chunkedContextNextTokens, lastDraftLens,
        bufferCast<SizeType32>(*inputs.seqSlots), workspace->getDeviceBatchSlotsPtr(), engineBatchSize, batchSize,
        getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::unpackData(EagleOutputs const& outputs, EagleInputs const& inputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const engineBatchSize = inputs.nextDraftLens->getDimension<0>();
    auto const maxSeqLen = outputs.outputIds->getDimension<-1>();

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace->getRawWorkspaceDevicePtr());
    size_t offset{0};

    SizeType32* augmentedSeqSlots = reinterpret_cast<SizeType32*>(
        nextWorkspacePtr(workspaceBytePtr, offset, engineBatchSize * sizeof(SizeType32)));

    UnpackEagleDataParams params;
    params.batchSlots = augmentedSeqSlots;
    params.inputCurandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesDevice));

    params.inputTemperatures = bufferCast<float>(*mTemperatureDevice);
    params.inputNextDraftTokens = bufferCast<TokenIdType>(*inputs.nextDraftTokens);
    params.inputNextDraftLens = bufferCast<SizeType32>(*inputs.nextDraftLens);
    params.inputNextDraftPaths = bufferCast<SizeType32>(*inputs.nextDraftPaths);
    params.inputLastDraftTokens = bufferCast<TokenIdType>(*inputs.lastDraftTokens);
    params.inputLastDraftLens = bufferCast<SizeType32>(*inputs.lastDraftLens);
    params.inputAcceptedTokens = bufferCast<TokenIdType>(*inputs.acceptedTokens);
    params.inputAcceptedLens = bufferCast<SizeType32>(*inputs.acceptedLens);

    params.outputIds = bufferCast<TokenIdType>(*outputs.outputIds);
    params.outputNumNewTokens = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    params.outputSequenceLengths = bufferCast<SizeType32>(*outputs.sequenceLength.value());
    params.outputUnpackedNextDraftTokens = bufferCast<TokenIdType>(*outputs.unpackedNextDraftTokens);
    params.outputNextDraftTokens = bufferCast<TokenIdType>(*outputs.nextDraftTokens);
    params.outputNextDraftLengths = bufferCast<SizeType32>(*outputs.nextDraftLengths);
    params.outputNextGenerationLength = bufferCast<SizeType32>(*outputs.generationLengths);
    params.outputNextDraftPaths = bufferCast<SizeType32>(*outputs.nextDraftPaths);
    params.outputPrevDraftLengths = bufferCast<SizeType32>(*outputs.prevDraftLengths);
    params.outputPositionIds = bufferCast<SizeType32>(*outputs.nextDraftPosIds);

    params.outputRandDataSample = bufferCast<float>(*outputs.randomDataSample);
    params.outputRandDataVerification = bufferCast<float>(*outputs.randomDataValidation);

    params.outputTemperatures = bufferCast<float>(*outputs.temperatures);

    params.outputEagleNetCtxRequestTypes = bufferCast<SizeType32>(*mEagleNetCtxRequestTypes);
    params.outputEagleNetCtxContextLengths = bufferCast<SizeType32>(*mEagleNetCtxContextLengths);
    params.outputEagleNetCtxPastKeyValueLengths = bufferCast<SizeType32>(*mEagleNetCtxPastKeyValueLengths);
    params.outputEagleNetGenRequestTypes = bufferCast<SizeType32>(*mEagleNetGenRequestTypes);
    params.outputEagleNetGenContextLengths = bufferCast<SizeType32>(*mEagleNetGenContextLengths);
    params.outputEagleNetGenPastKeyValueLengths = bufferCast<SizeType32>(*mEagleNetGenPastKeyValueLengths);

    params.batchSize = engineBatchSize;
    params.maxDecodingTokens = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingTokens();
    params.maxPathLength = mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen();
    params.maxSeqLen = maxSeqLen;

    params.checkParams();

    invokeUnpackEagleData(params, getStream());

    mBufferManager->copy(*mEagleNetCtxRequestTypes, *outputs.eagleNetCtxRequestTypesHost);
    mBufferManager->copy(*mEagleNetCtxContextLengths, *outputs.eagleNetCtxContextLengthsHost);
    mBufferManager->copy(*mEagleNetCtxPastKeyValueLengths, *outputs.eagleNetCtxPastKeyValueLengthsHost);
    mBufferManager->copy(*mEagleNetGenRequestTypes, *outputs.eagleNetGenRequestTypesHost);
    mBufferManager->copy(*mEagleNetGenContextLengths, *outputs.eagleNetGenContextLengthsHost);
    mBufferManager->copy(*mEagleNetGenPastKeyValueLengths, *outputs.eagleNetGenPastKeyValueLengthsHost);

    mBufferManager->copy(*outputs.generationLengths, *outputs.generationLengthsHost);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::convertToPackedMask(EagleOutputs const& outputs, EagleInputs const& inputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const engineBatchSize = inputs.nextDraftLens->getDimension<0>();
    auto const maxDecodingTokens = mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingTokens();
    auto const maxPathLen = mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen();

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace->getRawWorkspaceDevicePtr());
    size_t offset{0};
    SizeType32* augmentedSeqSlots = reinterpret_cast<SizeType32*>(
        nextWorkspacePtr(workspaceBytePtr, offset, engineBatchSize * sizeof(SizeType32)));

    auto batchSlots = augmentedSeqSlots;
    auto packedMasksDevice = bufferCast<SizeType32>(*outputs.packedMasks);
    auto nextDraftPaths = bufferCast<SizeType32>(*outputs.nextDraftPaths);

    invokeGetPackedMaskFromPath(
        packedMasksDevice, batchSlots, nextDraftPaths, engineBatchSize, maxDecodingTokens, maxPathLen, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodingLayer<T>::packAcceptedPaths(EagleOutputs const& outputs, EagleInputs const& inputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;
    auto const engineBatchSize = inputs.nextDraftLens->getDimension<0>();

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace->getRawWorkspaceDevicePtr());
    size_t offset{0};
    nextWorkspacePtr(workspaceBytePtr, offset, engineBatchSize * sizeof(SizeType32));

    SizeType32* augmentedBatchSlots = reinterpret_cast<SizeType32*>(
        nextWorkspacePtr(workspaceBytePtr, offset, engineBatchSize * sizeof(SizeType32)));

    auto numNewTokens = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    auto numNewTokensCumSum = bufferCast<SizeType32>(*outputs.numNewTokensCumSum);
    auto pathsOffsets = bufferCast<SizeType32>(*outputs.pathsOffsets);
    auto batchSlots = augmentedBatchSlots;
    auto bestPathIndicesSlotsPtr = bufferCast<SizeType32>(*inputs.acceptedPathIds);
    auto lastDraftPathsSlotsPtr = bufferCast<SizeType32>(*inputs.lastDraftPaths);

    CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for EagleDecodingLayer");
    CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for EagleDecodingLayer");
    CHECK_WITH_INFO(numNewTokensCumSum != nullptr, "numNewTokensCumSum must be provided for EagleDecodingLayer");
    CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for EagleDecodingLayer");
    invokePackAcceptedPaths(numNewTokensCumSum, pathsOffsets, numNewTokens, bestPathIndicesSlotsPtr,
        lastDraftPathsSlotsPtr, batchSlots, batchSize, engineBatchSize,
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen(), true, getStream());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t EagleDecodingLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template class EagleDecodingLayer<float>;
template class EagleDecodingLayer<half>;

}
