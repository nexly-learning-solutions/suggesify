
#include "explicitDraftTokensBuffers.h"

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../src/speculativeDecoding/explicitDraftTokensKernels.h"
#include "common.h"
#include "iBuffer.h"

namespace tksd = suggestify::kernels::speculative_decoding;

namespace suggestify::runtime
{

void ExplicitDraftTokensBuffers::Inputs::create(SizeType32 maxNumSequences, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    auto const& manager = runtime.getBufferManager();

    auto const& speculativeDecodingModule = modelConfig.getSpeculativeDecodingModule();
    auto const maxNumPaths = speculativeDecodingModule.getMaxNumPaths();
    auto const maxDraftPathLen = speculativeDecodingModule.getMaxDraftPathLen();
    auto const maxPathLen = speculativeDecodingModule.getMaxPathLen();
    auto const maxDecodingTokens = speculativeDecodingModule.getMaxDecodingTokens();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;
    auto const dtype = modelConfig.getDataType();

    maxGenLengthHost = manager.pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    temperatures = manager.gpu(ITensor::makeShape({maxNumSequences}), dtype);
    positionIdsBase = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    generationLengths = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    generationLengthsHost = manager.pinned(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    randomDataSample = manager.gpu(ITensor::makeShape({maxNumSequences}), dtype);
    randomDataValidation = manager.gpu(ITensor::makeShape({maxNumSequences, maxNumPaths, maxDraftPathLen}), dtype);
    draftTokens = manager.gpu(ITensor::makeShape({maxNumSequences, maxNumPaths, maxPathLen}), TRTTokenIdType);
    draftIndices
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxNumPaths, maxPathLen}), nvinfer1::DataType::kINT32);
    draftProbs
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxNumPaths, maxDraftPathLen, vocabSizePadded}), dtype);
    packedMasks
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingTokens, common::ceilDiv(maxDecodingTokens, 32)}),
            nvinfer1::DataType::kINT32);
    positionIds = manager.gpu(ITensor::makeShape({maxNumSequences * maxDecodingTokens}), nvinfer1::DataType::kINT32);
    useSpecDecoding = manager.cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
}

ExplicitDraftTokensBuffers::ExplicitDraftTokensBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, executor::DecodingConfig const& decodingConfig,
    runtime::TllmRuntime const& runtime)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    CHECK_WITH_INFO(maxBeamWidth == 1, "Explicit draft tokens does not support beam search");

    auto const maxNumSequences = maxBatchSize;
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const explicitDraftTokensModule
        = std::dynamic_pointer_cast<suggestify::runtime::ExplicitDraftTokensModule const>(
            modelConfig.getSpeculativeDecodingModulePtr());

    auto const numBeams = explicitDraftTokensModule->getMaxNumPaths();
    auto const beamDraftLength = explicitDraftTokensModule->getMaxDraftPathLen();
    auto const beamLength = explicitDraftTokensModule->getMaxPathLen();

    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;
    auto const dtype = modelConfig.getDataType();

    engineInputs.requestTypesDevice = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.temperatures = manager.emptyTensor(runtime::MemoryType::kGPU, dtype);

    engineInputs.draftTokens = manager.gpu(ITensor::makeShape({maxNumSequences, numBeams, beamLength}), TRTTokenIdType);
    engineInputs.draftIndices
        = manager.gpu(ITensor::makeShape({maxNumSequences, numBeams, beamLength}), nvinfer1::DataType::kINT32);
    engineInputs.draftProbs
        = manager.gpu(ITensor::makeShape({maxNumSequences, numBeams, beamDraftLength, vocabSizePadded}), dtype);

    engineInputs.generationLengths = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.positionIds = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.positionOffsets = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.packedMasks = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);

    engineInputs.randomDataSample = manager.emptyTensor(runtime::MemoryType::kGPU, dtype);
    engineInputs.randomDataValidation = manager.emptyTensor(runtime::MemoryType::kGPU, dtype);
    engineInputs.positionIdsBase = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.useSpecDecoding = manager.cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    bufferCast<SizeType32>(*engineInputs.useSpecDecoding)[0] = 1;

    engineOutputs.nextDraftTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences, numBeams, beamLength}), TRTTokenIdType);
    engineOutputs.nextDraftIndices
        = manager.gpu(ITensor::makeShape({maxNumSequences, numBeams, beamLength}), nvinfer1::DataType::kINT32);
    engineOutputs.nextDraftProbs
        = manager.gpu(ITensor::makeShape({maxNumSequences, numBeams, beamDraftLength, vocabSizePadded}), dtype);

    engineOutputs.maxGenToken = manager.gpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    engineOutputs.totalGenToken = manager.gpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    engineOutputs.nextGenerationLengths = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineOutputs.nextPositionOffsets = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineOutputs.masks = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kBOOL);

    engineOutputs.nextFlatTokens = manager.emptyTensor(runtime::MemoryType::kGPU, TRTTokenIdType);
    engineOutputs.bestPathLengths = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineOutputs.bestPathIndices = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineOutputs.packedPositionIds = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);

    auto const& stream = manager.getStream();
    scanTempStorageBytes
        = tksd::invokeScanGenerationLengths(nullptr, 0, nullptr, nullptr, maxNumSequences, stream.get());
    scanTempStorage = manager.gpu(scanTempStorageBytes);
    cumSumGenerationLengths = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);

    reshape(0, maxNumSequences, modelConfig);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void ExplicitDraftTokensBuffers::reshape(
    SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const numSequences = numCtxSequences + numGenSequences;

    auto const explicitDraftTokensModule
        = std::dynamic_pointer_cast<suggestify::runtime::ExplicitDraftTokensModule const>(
            modelConfig.getSpeculativeDecodingModulePtr());

    auto const numBeams = explicitDraftTokensModule->getMaxNumPaths();
    auto const beamDraftLength = explicitDraftTokensModule->getMaxDraftPathLen();
    auto const maxDecodingTokens = explicitDraftTokensModule->getMaxDecodingTokens();

    engineInputs.requestTypesDevice->reshape(ITensor::makeShape({numSequences}));
    engineInputs.temperatures->reshape(ITensor::makeShape({numSequences}));

    auto draftTokensShape = engineInputs.draftTokens->getShape();
    draftTokensShape.d[0] = numGenSequences;
    engineInputs.draftTokens->reshape(draftTokensShape);
    auto draftIndicesShape = engineInputs.draftIndices->getShape();
    draftIndicesShape.d[0] = numGenSequences;
    engineInputs.draftIndices->reshape(draftIndicesShape);
    auto draftProbsShape = engineInputs.draftProbs->getShape();
    draftProbsShape.d[0] = numGenSequences;
    engineInputs.draftProbs->reshape(draftProbsShape);

    engineInputs.generationLengths->reshape(ITensor::makeShape({numGenSequences}));
    engineInputs.positionIds->reshape(ITensor::makeShape({numSequences * maxDecodingTokens}));
    engineInputs.positionOffsets->reshape(ITensor::makeShape({numGenSequences, maxDecodingTokens}));
    engineInputs.packedMasks->reshape(
        ITensor::makeShape({numGenSequences * maxDecodingTokens, common::ceilDiv(maxDecodingTokens, 32)}));

    engineInputs.randomDataSample->reshape(ITensor::makeShape({numSequences}));
    engineInputs.randomDataValidation->reshape(ITensor::makeShape({numGenSequences, numBeams, beamDraftLength}));
    engineInputs.positionIdsBase->reshape(ITensor::makeShape({numSequences}));

    engineOutputs.nextGenerationLengths->reshape(ITensor::makeShape({numSequences}));
    engineOutputs.nextPositionOffsets->reshape(ITensor::makeShape({numSequences, maxDecodingTokens}));
    engineOutputs.masks->reshape(ITensor::makeShape({numSequences, maxDecodingTokens, maxDecodingTokens}));

    engineOutputs.nextFlatTokens->reshape(ITensor::makeShape({numSequences * maxDecodingTokens}));
    engineOutputs.bestPathLengths->reshape(ITensor::makeShape({numSequences}));
    engineOutputs.bestPathIndices->reshape(ITensor::makeShape({numSequences}));
    engineOutputs.packedPositionIds->reshape(ITensor::makeShape({numSequences * maxDecodingTokens}));

    cumSumGenerationLengths->reshape(ITensor::makeShape({numSequences}));

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensBuffers::setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences,
    SizeType32 vocabSizePadded, ITensor const& seqSlots, ExplicitDraftTokensBuffers::Inputs const& draftBuffers,
    ITensor const& contextPositionIds, runtime::ExplicitDraftTokensModule const& explicitDraftTokensModule,
    runtime::CudaStream const& stream) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    using runtime::bufferCast;

    tksd::PackExplicitDraftTokensParams<T> params;
    params.batchSize = numCtxSequences + numGenSequences;
    params.numPaths = explicitDraftTokensModule.getMaxNumPaths();
    params.maxPathLength = explicitDraftTokensModule.getMaxPathLen();
    params.vocabSize = vocabSizePadded;
    params.numContextRequests = numCtxSequences;
    params.numGenerationRequests = numGenSequences;
    params.numContextTokens = contextPositionIds.getShape().d[0];

    params.batchSlots = bufferCast<SizeType32>(seqSlots);

    params.maxGenerationLength = bufferCast<SizeType32>(*engineOutputs.maxGenToken);

    params.inputTemperatures = bufferCast<T>(*draftBuffers.temperatures);
    params.inputPositionIdsBase = bufferCast<SizeType32>(*draftBuffers.positionIdsBase);
    params.inputGenerationLengths = bufferCast<SizeType32>(*draftBuffers.generationLengths);
    params.inputRandomDataSample = bufferCast<T>(*draftBuffers.randomDataSample);
    params.inputRandomDataValidation = bufferCast<T>(*draftBuffers.randomDataValidation);
    params.inputNextDraftTokens = bufferCast<runtime::TokenIdType>(*draftBuffers.draftTokens);
    params.inputNextDraftIndices = bufferCast<SizeType32>(*draftBuffers.draftIndices);
    params.inputDraftProbs = bufferCast<T>(*draftBuffers.draftProbs);
    params.inputPackedMask = bufferCast<int32_t>(*draftBuffers.packedMasks);
    params.inputPositionIds = bufferCast<SizeType32>(*draftBuffers.positionIds);

    params.outputTemperatures = bufferCast<T>(*engineInputs.temperatures);
    params.outputPositionIdsBase = bufferCast<SizeType32>(*engineInputs.positionIdsBase);
    params.outputGenerationLengths = bufferCast<SizeType32>(*engineInputs.generationLengths);
    params.outputRandomDataSample = bufferCast<T>(*engineInputs.randomDataSample);
    params.outputRandomDataValidation = bufferCast<T>(*engineInputs.randomDataValidation);
    params.outputNextDraftTokens = bufferCast<runtime::TokenIdType>(*engineInputs.draftTokens);
    params.outputNextDraftIndices = bufferCast<SizeType32>(*engineInputs.draftIndices);
    params.outputDraftProbs = bufferCast<T>(*engineInputs.draftProbs);
    params.outputPackedMask = bufferCast<int32_t>(*engineInputs.packedMasks);
    params.outputPositionOffsets = bufferCast<SizeType32>(*engineInputs.positionOffsets);
    params.outputPositionIds = bufferCast<SizeType32>(*engineInputs.positionIds);

    params.cumSumGenerationLengths = bufferCast<SizeType32>(*cumSumGenerationLengths);

    params.checkParams();

    tksd::invokePackGenerationLengths(params, stream.get());

    if (numGenSequences)
    {
        tksd::invokeScanGenerationLengths(bufferCast<uint8_t>(*scanTempStorage), scanTempStorageBytes,
            bufferCast<SizeType32>(*engineInputs.generationLengths), bufferCast<SizeType32>(*cumSumGenerationLengths),
            numGenSequences, stream.get());
    }

    tksd::invokePackExplicitDraftTokens(params, stream.get());

    if (numGenSequences)
    {
        tksd::invokeCopyProbs(params, stream.get());
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void ExplicitDraftTokensBuffers::setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences,
    ITensor const& requestTypes, ITensor const& seqSlots, ExplicitDraftTokensBuffers::Inputs const& draftBuffers,
    ITensor const& contextPositionIds, runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();
    auto const& stream = runtime.getStream();

    auto const explicitDraftTokensModule = std::dynamic_pointer_cast<runtime::ExplicitDraftTokensModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());

    auto const seqSlotsPtr = bufferCast<SizeType32>(seqSlots);
    auto const generationLengthsPtr = bufferCast<SizeType32>(*draftBuffers.generationLengthsHost);
    SizeType32 totalGenLengths = 0;
    for (SizeType32 si = 0; si < numGenSequences; ++si)
    {
        auto const slot = seqSlotsPtr[numCtxSequences + si];
        totalGenLengths += generationLengthsPtr[slot];
    }

    engineInputs.positionIds->reshape(ITensor::makeShape({contextPositionIds.getShape().d[0] + totalGenLengths}));
    TensorPtr posIdsSlice = ITensor::slice(engineInputs.positionIds, 0, contextPositionIds.getShape().d[0]);
    manager.copy(contextPositionIds, *posIdsSlice);

    manager.copy(requestTypes, *engineInputs.requestTypesDevice);

    auto const numSequences = numCtxSequences + numGenSequences;
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const dtype = modelConfig.getDataType();

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        setFromInputs<float>(numCtxSequences, numGenSequences, vocabSizePadded, seqSlots, draftBuffers,
            contextPositionIds, *explicitDraftTokensModule, stream);
        break;
    case nvinfer1::DataType::kHALF:
        setFromInputs<half>(numCtxSequences, numGenSequences, vocabSizePadded, seqSlots, draftBuffers,
            contextPositionIds, *explicitDraftTokensModule, stream);
        break;
    case nvinfer1::DataType::kBF16:
        setFromInputs<__nv_bfloat16>(numCtxSequences, numGenSequences, vocabSizePadded, seqSlots, draftBuffers,
            contextPositionIds, *explicitDraftTokensModule, stream);
        break;
    default:
        THROW("DataType %d not supported in ExplicitDraftTokensBuffers", static_cast<SizeType32>(dtype));
        break;
    }

    auto draftTokensShape = engineOutputs.nextDraftTokens->getShape();
    draftTokensShape.d[0] = numSequences;
    engineOutputs.nextDraftTokens->reshape(draftTokensShape);
    auto draftIndicesShape = engineOutputs.nextDraftIndices->getShape();
    draftIndicesShape.d[0] = numSequences;
    engineOutputs.nextDraftIndices->reshape(draftIndicesShape);
    auto draftProbsShape = engineOutputs.nextDraftProbs->getShape();
    draftProbsShape.d[0] = numSequences;
    engineOutputs.nextDraftProbs->reshape(draftProbsShape);

    auto maxGenLength = bufferCast<SizeType32>(*draftBuffers.maxGenLengthHost)[0];
    if (maxGenLength == 0)
    {
        maxGenLength = explicitDraftTokensModule->getMaxDecodingTokens();
    }
    auto positionOffsetsShape = engineInputs.positionOffsets->getShape();
    positionOffsetsShape.d[1] = maxGenLength;
    engineInputs.positionOffsets->reshape(positionOffsetsShape);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void ExplicitDraftTokensBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const&) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.insert_or_assign("redrafter_inverted_temperature", engineInputs.temperatures);
    inputBuffers.insert_or_assign("device_request_types", engineInputs.requestTypesDevice);

    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", engineInputs.generationLengths);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", engineInputs.positionOffsets);
    inputBuffers.insert_or_assign("spec_decoding_packed_mask", engineInputs.packedMasks);

    inputBuffers.insert_or_assign("draft_tokens", engineInputs.draftTokens);
    inputBuffers.insert_or_assign("draft_indices", engineInputs.draftIndices);
    inputBuffers.insert_or_assign("draft_probs", engineInputs.draftProbs);

    inputBuffers.insert_or_assign("rand_data_sample", engineInputs.randomDataSample);
    inputBuffers.insert_or_assign("rand_data_validation", engineInputs.randomDataValidation);
    inputBuffers.insert_or_assign("position_ids_base", engineInputs.positionIdsBase);
    inputBuffers.insert_or_assign("position_ids", engineInputs.positionIds);
    inputBuffers.insert_or_assign("spec_decoding_use", engineInputs.useSpecDecoding);

    outputBuffers.insert_or_assign("next_spec_decoding_generation_lengths", engineOutputs.nextGenerationLengths);
    outputBuffers.insert_or_assign("next_spec_decoding_position_offsets", engineOutputs.nextPositionOffsets);
    outputBuffers.insert_or_assign("spec_decoding_mask", engineOutputs.masks);

    outputBuffers.insert_or_assign("next_draft_tokens", engineOutputs.nextDraftTokens);
    outputBuffers.insert_or_assign("next_draft_indices", engineOutputs.nextDraftIndices);
    outputBuffers.insert_or_assign("next_draft_probs", engineOutputs.nextDraftProbs);
    outputBuffers.insert_or_assign("next_flat_tokens", engineOutputs.nextFlatTokens);

    outputBuffers.insert_or_assign("num_accepted_tokens", engineOutputs.bestPathLengths);
    outputBuffers.insert_or_assign("accepted_beam_index", engineOutputs.bestPathIndices);
    outputBuffers.insert_or_assign("max_gen_token", engineOutputs.maxGenToken);
    outputBuffers.insert_or_assign("total_gen_token", engineOutputs.totalGenToken);
    outputBuffers.insert_or_assign("packed_position_ids", engineOutputs.packedPositionIds);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

}
