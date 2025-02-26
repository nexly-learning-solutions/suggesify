
#include "rnnStateBuffers.h"
#include "iBuffer.h"
#include "runtimeBuffers.h"
#include "utils/sessionUtils.h"

using namespace suggestify::runtime;

RnnStateBuffers::RnnStateBuffers()
{
    rnnStates = nullptr;
    convStates = nullptr;
    convStatesAlt = nullptr;
    slotMappingHost = nullptr;
    slotMappingDevice = nullptr;
    rnnStatePtrs = nullptr;
    convStatePtrs = nullptr;
}

RnnStateBuffers::RnnStateBuffers(
    TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CHECK(modelConfig.isRnnBased());
    CHECK_WITH_INFO(modelConfig.hasRnnConfig(), "RNN only support Mamba1/Mamba2/RecurrentGemma now.");
    auto maxBatchSize = modelConfig.getMaxBatchSize();
    auto maxBeamWidth = modelConfig.getMaxBeamWidth();
    auto maxBatchBeam = maxBatchSize * maxBeamWidth;
    auto rnnConfig = modelConfig.getRnnConfig();
    CHECK_WITH_INFO(rnnConfig.has_value(), "RnnStateBuffers should be used with rnnConfig.");
    mConvKernel = rnnConfig->convKernel;
    mStateSize = rnnConfig->stateSize;
    mRnnHiddenSize = rnnConfig->rnnHiddenSize;
    mRnnHeadSize = rnnConfig->rnnHeadSize;
    mRnnConvDimSize = rnnConfig->rnnConvDimSize;
    auto dType = modelConfig.getDataType();
    auto const localNbLayers = modelConfig.getNbRnnLayers(worldConfig.getPipelineParallelism());
    mLocalNbLayers = localNbLayers;
    mMaxBeamWidth = maxBeamWidth;
    mUseMambaConv1dPlugin = modelConfig.useMambaConv1dPlugin();
    auto const rnnStatesShape = [&]()
    {
        if (mRnnHeadSize > 0)
        {
            return suggestify::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mRnnHiddenSize / mRnnHeadSize, mStateSize, mRnnHeadSize});
        }
        else
        {
            return suggestify::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mStateSize, mRnnHiddenSize});
        }
    }();
    auto const convStatesShape = [&]()
    {
        if (mUseMambaConv1dPlugin)
        {
            return suggestify::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mConvKernel - 1, mRnnConvDimSize});
        }
        else
        {
            return suggestify::runtime::ITensor::makeShape(
                {localNbLayers * maxBatchBeam, mRnnConvDimSize, mConvKernel - 1});
        }
    }();
    auto& bufferManager = runtime.getBufferManager();
    auto const isRecurrentGemma = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kRecurrentGemma;
    auto stateDType = isRecurrentGemma ? nvinfer1::DataType::kFLOAT : dType;
    rnnStates = bufferManager.gpu(rnnStatesShape, stateDType);
    convStates = bufferManager.gpu(convStatesShape, dType);
    convStatesAlt = bufferManager.gpu(convStatesShape, dType);

    if (modelConfig.usePagedState())
    {
        auto slotMappingShape = ITensor::makeShape({maxBatchSize});
        auto statePtrsShape = ITensor::makeShape({localNbLayers});
        slotMappingDevice = bufferManager.gpu(slotMappingShape, nvinfer1::DataType::kINT32);
        slotMappingHost = BufferManager::cpu(slotMappingShape, nvinfer1::DataType::kINT32);
        rnnStatePtrs = BufferManager::cpu(statePtrsShape, TRTDataType<void*>::value);
        convStatePtrs = BufferManager::cpu(statePtrsShape, TRTDataType<void*>::value);
    }
    else
    {
        slotMappingHost = nullptr;
        slotMappingDevice = nullptr;
        rnnStatePtrs = nullptr;
        convStatePtrs = nullptr;
    }

    reshape(maxBatchSize);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::reshape(SizeType32 batchSize)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const rnnStatesShape = [&]()
    {
        if (mRnnHeadSize > 0)
        {
            return suggestify::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mRnnHiddenSize / mRnnHeadSize, mStateSize, mRnnHeadSize});
        }
        else
        {
            return suggestify::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mStateSize, mRnnHiddenSize});
        }
    }();
    auto const convStatesShape = [&]()
    {
        if (mUseMambaConv1dPlugin)
        {
            return suggestify::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mConvKernel - 1, mRnnConvDimSize});
        }
        else
        {
            return suggestify::runtime::ITensor::makeShape(
                {mLocalNbLayers * batchSize * mMaxBeamWidth, mRnnConvDimSize, mConvKernel - 1});
        }
    }();
    rnnStates->reshape(rnnStatesShape);
    convStates->reshape(convStatesShape);
    convStatesAlt->reshape(convStatesShape);

    rnnState.resize(mLocalNbLayers);
    convState.resize(mLocalNbLayers);
    convStateAlt.resize(mLocalNbLayers);
    for (int i = 0; i < mLocalNbLayers; i++)
    {
        size_t offset = batchSize * mMaxBeamWidth * i;
        rnnState[i] = suggestify::runtime::ITensor::slice(rnnStates, offset, batchSize * mMaxBeamWidth);
        convState[i] = suggestify::runtime::ITensor::slice(convStates, offset, batchSize * mMaxBeamWidth);
        convStateAlt[i] = suggestify::runtime::ITensor::slice(convStatesAlt, offset, batchSize * mMaxBeamWidth);
    }
    if (slotMappingDevice != nullptr)
    {
        CHECK(slotMappingHost != nullptr);
        CHECK(rnnStates != nullptr && convStates != nullptr);
        CHECK(rnnStatePtrs != nullptr && convStatePtrs != nullptr);

        auto slotMappingShape = ITensor::makeShape({batchSize});
        slotMappingDevice->reshape(slotMappingShape);
        slotMappingHost->reshape(slotMappingShape);

        int* slotMapping = static_cast<int*>(slotMappingHost->data());
        for (int b = 0; b < batchSize; b++)
        {
            slotMapping[b] = b;
        }
        fillStatePtrs();
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::fillStatePtrs()
{
    auto statePtrsShape = ITensor::makeShape({mLocalNbLayers});
    rnnStatePtrs->reshape(statePtrsShape);
    convStatePtrs->reshape(statePtrsShape);

    rnnStatePtr.resize(mLocalNbLayers);
    convStatePtr.resize(mLocalNbLayers);

    auto* rnnStatePtrArray = bufferCast<void*>(*rnnStatePtrs);
    auto* convStatePtrArray = bufferCast<void*>(*convStatePtrs);

    for (int i = 0; i < mLocalNbLayers; i++)
    {
        rnnStatePtrArray[i] = rnnState[i]->data();
        convStatePtrArray[i] = convState[i]->data();
        rnnStatePtr[i] = suggestify::runtime::ITensor::slice(rnnStatePtrs, i, 1);
        convStatePtr[i] = suggestify::runtime::ITensor::slice(convStatePtrs, i, 1);
    }
}

void RnnStateBuffers::reshape(
    GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    auto const batchSize = generationConfig.batchSize;

    reshape(batchSize);
}

void RnnStateBuffers::reset(BufferManager& manager)
{
    manager.setZero(*rnnStates);
    manager.setZero(*convStates);
    manager.setZero(*convStatesAlt);
}

RnnStateBuffers RnnStateBuffers::sliceTo(SizeType32 offset, SizeType32 size)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    RnnStateBuffers buffers;
    buffers.rnnState = utils::sliceBufferVector(rnnState, offset, size);
    buffers.convState = utils::sliceBufferVector(convState, offset, size);
    buffers.convStateAlt = utils::sliceBufferVector(convStateAlt, offset, size);

    if (slotMappingDevice != nullptr)
    {
        CHECK(slotMappingHost != nullptr);
        CHECK(rnnStates != nullptr && convStates != nullptr);
        CHECK(rnnStatePtrs != nullptr && convStatePtrs != nullptr);
        buffers.slotMappingHost = ITensor::slice(slotMappingHost, offset, size);
        buffers.slotMappingDevice = ITensor::slice(slotMappingHost, offset, size);
        int* slotMapping = static_cast<int*>(buffers.slotMappingHost->data());
        for (int b = 0; b < size; b++)
        {
            slotMapping[b] = b;
        }
        buffers.fillStatePtrs();
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return buffers;
}

void RnnStateBuffers::prepareContextStep(RuntimeBuffers* runtimeBuffers, BufferManager& manager)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType32 const batchSize = runtimeBuffers->generationConfig.batchSize;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto RequestTypesPtr = bufferCast<int32_t>(*requestTypes);
    CHECK(requestTypes->getSize() == static_cast<std::size_t>(batchSize));
    std::fill_n(RequestTypesPtr, batchSize, 0);

    manager.setZero(*convStates);
    if (slotMappingDevice != nullptr)
    {
        manager.copy(*slotMappingHost, *slotMappingDevice);
    }
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    CHECK_WITH_INFO(false, "Beam search for mamba is not supported now.");
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& logits = runtimeBuffers->logits;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto const beamWidth = generationConfig.beamWidth;
    CHECK_WITH_INFO(beamWidth > 1, "Tiling is only necessary for beam search.");

    if (worldConfig.isLastPipelineParallelRank() && !modelConfig.computeContextLogits())
    {
        auto logitsShape = logits->getShape();
        logitsShape.d[1] *= beamWidth;
        utils::tileBufferReplace(logits, beamWidth, manager);
        logits->reshape(logitsShape);
    }

    utils::tileBufferReplace(contextLengthsDevice, beamWidth, manager);
    utils::tileCpuBufferReplace(contextLengthsHost, beamWidth);

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
    BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& outputLengths = runtimeBuffers->outputLengths;
    auto& lastTokenIds = runtimeBuffers->lastTokenIds;
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    requestTypes->reshape(ITensor::makeShape({batchSize * beamWidth}));
    auto hostRequestTypes = bufferCast<int32_t>(*requestTypes);
    std::fill_n(hostRequestTypes, requestTypes->getSize(), 1);

    if (modelConfig.computeContextLogits())
    {
        runtimeBuffers->gatherLastTokenLogits(manager, modelConfig, worldConfig);
    }

    if (beamWidth > 1)
    {
        tile(runtimeBuffers, manager, modelConfig, worldConfig);
    }

    manager.copy(*contextLengthsDevice, *outputLengths);
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth}));
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers,
    TensorMap& outputBuffers, SizeType32 const step, TensorPtr const& inputIds, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig) const
{
    LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& logits = runtimeBuffers->logits;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
    auto& lastTokenIds = runtimeBuffers->lastTokenIds;
    auto& requestTypes = runtimeBuffers->requestTypes;

    if (worldConfig.isLastPipelineParallelRank())
    {
        outputBuffers.insert_or_assign("logits", ITensor::view(logits));
    }
    else
    {
        outputBuffers.insert_or_assign("hidden_states_output", hiddenStates);
    }

    if (worldConfig.isFirstPipelineParallelRank())
    {
        inputBuffers.insert_or_assign("input_ids", inputIds);
    }
    else
    {
        inputBuffers.insert_or_assign("hidden_states_input", hiddenStates);
    }

    inputBuffers.insert_or_assign("last_token_ids", lastTokenIds);

    auto const localNbLayers = modelConfig.getNbRnnLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;
    auto const& layerTypes = modelConfig.getLayerTypes();

    if (modelConfig.usePagedState())
    {
        inputBuffers.insert_or_assign("slot_mapping", slotMappingDevice);
        utils::insertTensorVector(inputBuffers, "conv_state_ptr_", convStatePtr, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(
            inputBuffers, "rnn_state_ptr_", rnnStatePtr, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
    }
    else
    {
        utils::insertTensorVector(inputBuffers, "past_conv_state_", (step % 2) ? convState : convStateAlt, firstLayerId,
            layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(outputBuffers, "present_conv_state_", (step % 2) ? convStateAlt : convState,
            firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(
            inputBuffers, "past_rnn_state_", rnnState, firstLayerId, layerTypes, ModelConfig::LayerType::kRECURRENT);
        utils::insertTensorVector(outputBuffers, "present_rnn_state_", rnnState, firstLayerId, layerTypes,
            ModelConfig::LayerType::kRECURRENT);
    }

    inputBuffers.insert_or_assign("host_request_types", requestTypes);
    inputBuffers.insert_or_assign("host_context_lengths", runtimeBuffers->contextLengthsHost);
    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}
