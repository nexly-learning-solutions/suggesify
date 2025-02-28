
#include "eagleBuffers.h"
#include "../llmRequest.h"

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../src/speculativeDecoding/eagleDecodingKernels.h"
#include "../src/speculativeDecoding/explicitDraftTokensKernels.h"
#include "common.h"
#include "iBuffer.h"
#include "runtimeKernels.h"

namespace tksd = suggestify::kernels::speculative_decoding;

namespace suggestify::runtime
{

void EagleBuffers::Inputs::create(SizeType32 maxNumSequences, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    auto const& manager = runtime.getBufferManager();

    auto const& speculativeDecodingModule = modelConfig.getSpeculativeDecodingModule();
    auto const maxNumPaths = speculativeDecodingModule.getMaxNumPaths();
    auto const maxPathLen = speculativeDecodingModule.getMaxPathLen();
    auto const maxDecodingTokens = speculativeDecodingModule.getMaxDecodingTokens();
    auto const maxDecodingDraftTokens = speculativeDecodingModule.getMaxDecodingDraftTokens();

    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;

    temperatures = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kFLOAT);
    randomDataSample = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kFLOAT);
    randomDataValidation
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingTokens}), nvinfer1::DataType::kFLOAT);
    draftTokens = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingDraftTokens}), TRTTokenIdType);
    draftLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    draftPaths
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxNumPaths, maxPathLen}), nvinfer1::DataType::kINT32);
    specDecodingGenerationLengths = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    specDecodingGenerationLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    specDecodingPackedMasks
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingTokens, common::ceilDiv(maxDecodingTokens, 32)}),
            nvinfer1::DataType::kINT32);
    specDecodingPositionOffsets
        = manager.gpu(ITensor::makeShape({maxNumSequences * maxDecodingTokens}), nvinfer1::DataType::kINT32);

    eagleNetCtxRequestTypesHost = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetCtxContextLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetCtxPastKeyValueLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetGenRequestTypesHost = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetGenContextLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    eagleNetGenPastKeyValueLengthsHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    inputGenTokensHost
        = manager.pinnedPool(ITensor::makeShape({maxNumSequences * maxDecodingTokens}), nvinfer1::DataType::kINT32);
    chunkedContextNextTokens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    useSpecDecoding = manager.cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
}

EagleBuffers::EagleBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    CHECK_WITH_INFO(maxBeamWidth == 1, "EAGLE does not support beam search");

    auto const maxNumSequences = maxBatchSize;

    auto const eagleModule = std::dynamic_pointer_cast<suggestify::runtime::EagleModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());

    auto const numPaths = eagleModule->getMaxNumPaths();
    auto const pathLen = eagleModule->getMaxPathLen();
    auto const maxDecodingDraftTokens = eagleModule->getMaxDecodingDraftTokens();

    auto constexpr TRTTokenIdType = runtime::TRTDataType<runtime::TokenIdType>::value;

    engineInputs.temperatures = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    engineInputs.posteriorAlpha = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    engineInputs.posteriorThreshold = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    posteriorAlphaHost = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kFLOAT);
    posteriorThresholdHost = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kFLOAT);
    greedySamplingHost = manager.pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    engineInputs.draftTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences, maxDecodingDraftTokens}), TRTTokenIdType);
    engineInputs.draftLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    engineInputs.draftPaths
        = manager.gpu(ITensor::makeShape({maxNumSequences, numPaths, pathLen}), nvinfer1::DataType::kINT32);

    engineInputs.specDecodingGenerationLengths
        = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.specDecodingPositionOffsets
        = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.specDecodingPackedMasks = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);

    engineInputs.randomDataSample = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    engineInputs.randomDataValidation = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);

    engineInputs.eagleNetCtxRequestTypesHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetCtxContextLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetCtxPastKeyValueLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetGenRequestTypesHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetGenContextLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.eagleNetGenPastKeyValueLengthsHost
        = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.inputGenTokensHost = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    engineInputs.chunkedContextNextTokens = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    engineInputs.useSpecDecoding = manager.cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    bufferCast<SizeType32>(*engineInputs.useSpecDecoding)[0] = 1;
    chunkedContextNextTokensHost = manager.emptyTensor(runtime::MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);

    engineInputs.useDynamicTreeHost = manager.cpu(ITensor::makeShape({1}), nvinfer1::DataType::kBOOL);
    auto useDynamicTreeHostPtr = bufferCast<bool>(*(engineInputs.useDynamicTreeHost));
    useDynamicTreeHostPtr[0] = 0;

    engineOutputs.nextDraftTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences, numPaths, pathLen}), TRTTokenIdType);
    engineOutputs.nextDraftLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    engineOutputs.nextDraftPaths
        = manager.gpu(ITensor::makeShape({maxNumSequences, numPaths, pathLen}), nvinfer1::DataType::kINT32);

    engineOutputs.acceptedTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences, pathLen}), nvinfer1::DataType::kINT32);
    engineOutputs.acceptedLens = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    engineOutputs.acceptedPaths = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);
    engineOutputs.chunkedContextNextTokens
        = manager.gpu(ITensor::makeShape({maxNumSequences}), nvinfer1::DataType::kINT32);

    auto const& stream = manager.getStream();
    scanTempStorageBytes
        = tksd::invokeScanGenerationLengths(nullptr, 0, nullptr, nullptr, maxNumSequences, stream.get());
    reduceTempStorageBytes
        = tksd::invokeReduceMaxGenerationLengths(nullptr, 0, nullptr, nullptr, maxNumSequences, stream.get());
    scanReduceTempStorage = manager.gpu(std::max(reduceTempStorageBytes, scanTempStorageBytes));
    cumSumGenerationLengths = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
    maxGenerationLength = manager.gpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    reshape(0, maxNumSequences, modelConfig);

    auto const defaultConfig = decodingConfig.getEagleConfig().value_or(suggestify::executor::EagleConfig());
    mDoGreedySampling = defaultConfig.isGreedySampling();
    mDefaultPosteriorThreshold = defaultConfig.getPosteriorThreshold().value_or(mDefaultPosteriorThreshold);
    bufferCast<SizeType32>(*greedySamplingHost)[0] = mDoGreedySampling;

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EagleBuffers::reshape(
    SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const numSequences = numCtxSequences + numGenSequences;

    auto const eagleModule = std::dynamic_pointer_cast<suggestify::runtime::EagleModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());

    auto const maxDecodingTokens = eagleModule->getMaxDecodingTokens();

    engineInputs.temperatures->reshape(ITensor::makeShape({numSequences}));
    engineInputs.posteriorAlpha->reshape(ITensor::makeShape({numSequences}));
    engineInputs.posteriorThreshold->reshape(ITensor::makeShape({numSequences}));
    posteriorAlphaHost->reshape(ITensor::makeShape({numSequences}));
    posteriorThresholdHost->reshape(ITensor::makeShape({numSequences}));

    auto draftTokensShape = engineInputs.draftTokens->getShape();
    draftTokensShape.d[0] = numSequences;
    engineInputs.draftTokens->reshape(draftTokensShape);
    auto draftLensShape = engineInputs.draftLens->getShape();
    draftLensShape.d[0] = numSequences;
    engineInputs.draftLens->reshape(draftLensShape);
    auto draftPathsShape = engineInputs.draftPaths->getShape();
    draftPathsShape.d[0] = numSequences;
    engineInputs.draftPaths->reshape(draftPathsShape);

    engineInputs.specDecodingGenerationLengths->reshape(ITensor::makeShape({numGenSequences}));
    engineInputs.specDecodingPositionOffsets->reshape(ITensor::makeShape({numGenSequences, maxDecodingTokens}));
    engineInputs.specDecodingPackedMasks->reshape(
        ITensor::makeShape({numGenSequences * maxDecodingTokens, common::ceilDiv(maxDecodingTokens, 32)}));

    engineInputs.randomDataSample->reshape(ITensor::makeShape({numSequences}));
    engineInputs.randomDataValidation->reshape(ITensor::makeShape({numSequences, maxDecodingTokens}));

    engineInputs.eagleNetCtxRequestTypesHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetCtxContextLengthsHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetCtxPastKeyValueLengthsHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetGenRequestTypesHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetGenContextLengthsHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.eagleNetGenPastKeyValueLengthsHost->reshape(ITensor::makeShape({numSequences}));
    engineInputs.inputGenTokensHost->reshape(ITensor::makeShape({numSequences * maxDecodingTokens}));
    engineInputs.chunkedContextNextTokens->reshape(ITensor::makeShape({numSequences}));
    chunkedContextNextTokensHost->reshape(ITensor::makeShape({numSequences}));
    engineOutputs.chunkedContextNextTokens->reshape(ITensor::makeShape({numSequences}));

    cumSumGenerationLengths->reshape(ITensor::makeShape({numSequences + 1}));

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleBuffers::setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
    SizeType32 vocabSizePadded, ITensor const& seqSlots, EagleBuffers::Inputs const& draftBuffers,
    runtime::EagleModule const& eagleModule, runtime::BufferManager const& manager) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    using runtime::bufferCast;

    auto const numCtxSequences = static_cast<SizeType32>(contextRequests.size());
    auto const numGenSequences = static_cast<SizeType32>(genRequests.size());

    tksd::PackEagleParams params;
    params.batchSize = numCtxSequences + numGenSequences;
    params.maxNumPaths = eagleModule.getMaxNumPaths();
    params.maxDecodingTokens = eagleModule.getMaxDecodingTokens();
    params.maxPathLength = eagleModule.getMaxPathLen();
    params.numContextRequests = numCtxSequences;
    params.numGenerationRequests = numGenSequences;

    params.batchSlots = bufferCast<SizeType32>(seqSlots);

    params.inputTemperatures = bufferCast<float>(*draftBuffers.temperatures);
    params.inputRandomDataSample = bufferCast<float>(*draftBuffers.randomDataSample);
    params.inputRandomDataValidation = bufferCast<float>(*draftBuffers.randomDataValidation);

    params.inputNextDraftTokens = bufferCast<runtime::TokenIdType>(*draftBuffers.draftTokens);
    params.inputNextDraftPaths = bufferCast<SizeType32>(*draftBuffers.draftPaths);

    params.inputSpecDecodingGenerationLengths = bufferCast<SizeType32>(*draftBuffers.specDecodingGenerationLengths);
    params.inputSpecDecodingPositionOffsets = bufferCast<SizeType32>(*draftBuffers.specDecodingPositionOffsets);
    params.inputSpecDecodingPackedMasks = bufferCast<int32_t>(*draftBuffers.specDecodingPackedMasks);

    params.outputTemperatures = bufferCast<float>(*engineInputs.temperatures);
    params.outputRandomDataSample = bufferCast<float>(*engineInputs.randomDataSample);
    params.outputRandomDataValidation = bufferCast<float>(*engineInputs.randomDataValidation);

    params.outputNextDraftTokens = bufferCast<runtime::TokenIdType>(*engineInputs.draftTokens);
    params.outputNextDraftLens = bufferCast<SizeType32>(*engineInputs.draftLens);
    params.outputNextDraftPaths = bufferCast<SizeType32>(*engineInputs.draftPaths);

    params.outputSpecDecodingGenerationLengths = bufferCast<SizeType32>(*engineInputs.specDecodingGenerationLengths);
    params.outputSpecDecodingPositionOffsets = bufferCast<SizeType32>(*engineInputs.specDecodingPositionOffsets);
    params.outputSpecDecodingPackedMasks = bufferCast<int32_t>(*engineInputs.specDecodingPackedMasks);

    params.maxGenerationLength = bufferCast<SizeType32>(*maxGenerationLength);
    params.cumSumGenerationLengths = bufferCast<SizeType32>(*cumSumGenerationLengths);

    params.checkParams();

    tksd::invokePackEagleGenerationLengths(params, manager.getStream().get());

    if (numGenSequences)
    {
        tksd::invokeScanReduceGenerationLengths(numGenSequences,
            bufferCast<SizeType32>(*engineInputs.specDecodingGenerationLengths),
            bufferCast<uint8_t>(*scanReduceTempStorage), scanTempStorageBytes,
            bufferCast<SizeType32>(*cumSumGenerationLengths), bufferCast<uint8_t>(*scanReduceTempStorage),
            reduceTempStorageBytes, bufferCast<SizeType32>(*maxGenerationLength), manager.getStream().get());
    }

    tksd::invokePackEagle(params, manager.getStream().get());

    SizeType32 maxGenerationLengthHostValue{-1};
    SizeType32 numGenerationTokens{0};
    SizeType32 batchIdx{0};

    auto chunkedContextNextTokensHostPtr = bufferCast<TokenIdType>(*chunkedContextNextTokensHost);
    std::fill(chunkedContextNextTokensHostPtr, chunkedContextNextTokensHostPtr + params.batchSize, -1);

    auto setupEagleNetHostBuffers = [this, &draftBuffers](SizeType32 batchIdx, SizeType32 batchSlot)
    {
        bufferCast<SizeType32>(*this->engineInputs.eagleNetCtxRequestTypesHost)[batchIdx]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetCtxRequestTypesHost)[batchSlot];

        bufferCast<SizeType32>(*this->engineInputs.eagleNetCtxContextLengthsHost)[batchIdx]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetCtxContextLengthsHost)[batchSlot];

        bufferCast<SizeType32>(*this->engineInputs.eagleNetCtxPastKeyValueLengthsHost)[batchIdx]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetCtxPastKeyValueLengthsHost)[batchSlot];

        bufferCast<SizeType32>(*this->engineInputs.eagleNetGenRequestTypesHost)[batchIdx]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetGenRequestTypesHost)[batchSlot];

        bufferCast<SizeType32>(*this->engineInputs.eagleNetGenContextLengthsHost)[batchIdx]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetGenContextLengthsHost)[batchSlot];

        bufferCast<SizeType32>(*this->engineInputs.eagleNetGenPastKeyValueLengthsHost)[batchIdx]
            = bufferCast<SizeType32>(*draftBuffers.eagleNetGenPastKeyValueLengthsHost)[batchSlot];
    };

    auto posteriorAlphaHostPtr = bufferCast<float>(*posteriorAlphaHost);
    auto posteriorThresholdHostPtr = bufferCast<float>(*posteriorThresholdHost);
    auto setPosteriorThresholds
        = [this, posteriorAlphaHostPtr, posteriorThresholdHostPtr](LlmRequestPtr const& llmReq, SizeType32 batchIdx)
    {
        auto const eagleConfig = llmReq->getEagleConfig();

        float posteriorThreshold{this->mDefaultPosteriorThreshold};
        if (eagleConfig.has_value())
        {
            posteriorThreshold = eagleConfig->getPosteriorThreshold().value_or(posteriorThreshold);
        }
        posteriorAlphaHostPtr[batchIdx] = std::sqrt(posteriorThreshold);
        posteriorThresholdHostPtr[batchIdx] = posteriorThreshold;
    };

    for (auto const& llmReq : contextRequests)
    {
        if (llmReq->isLastContextChunk())
        {
            auto const batchSlot = params.batchSlots[batchIdx];
            setupEagleNetHostBuffers(batchIdx, batchSlot);
        }
        else
        {
            auto const contextChunkSize = llmReq->getContextChunkSize();
            auto const beginCompute = llmReq->getContextCurrentPosition();
            auto const endCompute = beginCompute + contextChunkSize;

            bufferCast<SizeType32>(*engineInputs.eagleNetCtxRequestTypesHost)[batchIdx] = 0;
            bufferCast<SizeType32>(*engineInputs.eagleNetCtxContextLengthsHost)[batchIdx] = contextChunkSize;
            bufferCast<SizeType32>(*engineInputs.eagleNetCtxPastKeyValueLengthsHost)[batchIdx]
                = beginCompute + contextChunkSize;

            bufferCast<SizeType32>(*engineInputs.eagleNetGenRequestTypesHost)[batchIdx] = 1;
            bufferCast<SizeType32>(*engineInputs.eagleNetGenContextLengthsHost)[batchIdx]
                = beginCompute + contextChunkSize;
            bufferCast<SizeType32>(*engineInputs.eagleNetGenPastKeyValueLengthsHost)[batchIdx]
                = beginCompute + contextChunkSize;

            TensorPtr draftPathsHost = BufferManager::pinnedPool(
                ITensor::makeShape({1, eagleModule.getMaxPathLen()}), nvinfer1::DataType::kINT32);
            for (SizeType32 ti = 0; ti < eagleModule.getMaxPathLen(); ++ti)
            {
                bufferCast<SizeType32>(*draftPathsHost)[ti] = ti;
            }

            TensorPtr draftPathsBatchSlice = ITensor::slice(engineInputs.draftPaths, batchIdx, 1);
            draftPathsBatchSlice->squeeze(0);
            kernels::invokeFill(*draftPathsBatchSlice, -1, manager.getStream());
            TensorPtr draftPathsBatchPathSlice = ITensor::slice(draftPathsBatchSlice, 0, 1);
            manager.copy(*draftPathsHost, *draftPathsBatchPathSlice);

            auto const& reqTokens = llmReq->getTokens(0);
            chunkedContextNextTokensHostPtr[batchIdx] = reqTokens[endCompute];
        }

        setPosteriorThresholds(llmReq, batchIdx);

        ++batchIdx;
    }

    for (auto const& llmReq : genRequests)
    {
        auto const batchSlot = params.batchSlots[batchIdx];
        setupEagleNetHostBuffers(batchIdx, batchSlot);
        setPosteriorThresholds(llmReq, batchIdx);

        auto const generationLength
            = bufferCast<SizeType32>(*draftBuffers.specDecodingGenerationLengthsHost)[batchSlot];
        maxGenerationLengthHostValue = std::max(maxGenerationLengthHostValue, generationLength);
        numGenerationTokens += generationLength;

        ++batchIdx;
    }

    if (maxGenerationLengthHostValue <= 0)
    {
        maxGenerationLengthHostValue = params.maxDecodingTokens;
    }

    auto specDecodingPositionOffsetsShape = engineInputs.specDecodingPositionOffsets->getShape();
    specDecodingPositionOffsetsShape.d[1] = maxGenerationLengthHostValue;
    engineInputs.specDecodingPositionOffsets->reshape(specDecodingPositionOffsetsShape);

    auto inputGenTokensHostShape = engineInputs.inputGenTokensHost->getShape();
    inputGenTokensHostShape.d[0] = numGenerationTokens;
    engineInputs.inputGenTokensHost->reshape(inputGenTokensHostShape);

    manager.copy(*chunkedContextNextTokensHost, *engineInputs.chunkedContextNextTokens);
    manager.copy(*chunkedContextNextTokensHost, *engineOutputs.chunkedContextNextTokens);
    manager.copy(*posteriorAlphaHost, *engineInputs.posteriorAlpha);
    manager.copy(*posteriorThresholdHost, *engineInputs.posteriorThreshold);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EagleBuffers::setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
    ITensor const& requestTypes, ITensor const& seqSlots, EagleBuffers::Inputs const& draftBuffers,
    runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& manager = runtime.getBufferManager();

    auto const eagleModule
        = std::dynamic_pointer_cast<runtime::EagleModule const>(modelConfig.getSpeculativeDecodingModulePtr());

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const dtype = modelConfig.getDataType();

    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        setFromInputs<float>(
            contextRequests, genRequests, vocabSizePadded, seqSlots, draftBuffers, *eagleModule, manager);
        break;
    case nvinfer1::DataType::kHALF:
        setFromInputs<half>(
            contextRequests, genRequests, vocabSizePadded, seqSlots, draftBuffers, *eagleModule, manager);
        break;
    default: THROW("DataType %d not supported in EagleBuffers", static_cast<SizeType32>(dtype)); break;
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void EagleBuffers::insertInputTensors(
    TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const&) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.insert_or_assign("greedy_sampling", greedySamplingHost);
    inputBuffers.insert_or_assign("eagle_temperature", engineInputs.temperatures);
    inputBuffers.insert_or_assign("posterior_alpha", engineInputs.posteriorAlpha);
    inputBuffers.insert_or_assign("posterior_threshold", engineInputs.posteriorThreshold);

    inputBuffers.insert_or_assign("spec_decoding_generation_lengths", engineInputs.specDecodingGenerationLengths);
    inputBuffers.insert_or_assign("spec_decoding_position_offsets", engineInputs.specDecodingPositionOffsets);
    inputBuffers.insert_or_assign("spec_decoding_packed_mask", engineInputs.specDecodingPackedMasks);

    inputBuffers.insert_or_assign("rand_data_sample", engineInputs.randomDataSample);
    inputBuffers.insert_or_assign("rand_data_validation", engineInputs.randomDataValidation);

    inputBuffers.insert_or_assign("draft_tokens", engineInputs.draftTokens);
    inputBuffers.insert_or_assign("draft_lens", engineInputs.draftLens);
    inputBuffers.insert_or_assign("draft_paths", engineInputs.draftPaths);

    inputBuffers.insert_or_assign("host_ctx_eagle_net_request_types", engineInputs.eagleNetCtxRequestTypesHost);
    inputBuffers.insert_or_assign("host_ctx_eagle_net_context_lengths", engineInputs.eagleNetCtxContextLengthsHost);
    inputBuffers.insert_or_assign(
        "host_ctx_eagle_net_past_key_value_lengths", engineInputs.eagleNetCtxPastKeyValueLengthsHost);
    inputBuffers.insert_or_assign("host_gen_eagle_net_request_types", engineInputs.eagleNetGenRequestTypesHost);
    inputBuffers.insert_or_assign("host_gen_eagle_net_context_lengths", engineInputs.eagleNetGenContextLengthsHost);
    inputBuffers.insert_or_assign(
        "host_gen_eagle_net_past_key_value_lengths", engineInputs.eagleNetGenPastKeyValueLengthsHost);
    inputBuffers.insert_or_assign("input_gen_tokens", engineInputs.inputGenTokensHost);
    inputBuffers.insert_or_assign("chunked_context_next_tokens", engineInputs.chunkedContextNextTokens);
    inputBuffers.insert_or_assign("use_dynamic_tree", engineInputs.useDynamicTreeHost);
    inputBuffers.insert_or_assign("spec_decoding_use", engineInputs.useSpecDecoding);

    outputBuffers.insert_or_assign("next_draft_tokens", engineOutputs.nextDraftTokens);
    outputBuffers.insert_or_assign("next_draft_lens", engineOutputs.nextDraftLens);
    outputBuffers.insert_or_assign("next_draft_paths", engineOutputs.nextDraftPaths);

    outputBuffers.insert_or_assign("accepted_tokens", engineOutputs.acceptedTokens);
    outputBuffers.insert_or_assign("num_accepted_tokens", engineOutputs.acceptedLens);
    outputBuffers.insert_or_assign("accepted_paths", engineOutputs.acceptedPaths);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

}
