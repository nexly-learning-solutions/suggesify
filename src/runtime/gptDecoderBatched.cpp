#include "gptDecoderBatched.h"
#include "../llmRequest.h"

#include "../common/assert.h"
#include "../types.h"
#include "../src/decodingCommon.h"
#include "../src/decodingKernels.h"
#include "bufferManager.h"
#include "cudaEvent.h"
#include "memoryCounters.h"
#include "runtimeBuffers.h"
#include "runtimeKernels.h"
#include "utils/speculativeChoicesUtils.h"

#include <algorithm>
#include <cassert>
#include <memory>

using namespace suggestify::runtime;

namespace tc = suggestify::common;
namespace tk = suggestify::kernels;

namespace
{
SamplingConfig extractSamplingConfig(SamplingConfig const& batchSamplingConfig, SizeType32 batchIdx)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    SamplingConfig samplingConfig{batchSamplingConfig.beamWidth};

    auto extractOptional = [&batchIdx](auto& single, auto const& batch)
    {
        using T = typename std::remove_reference_t<decltype(batch)>::value_type;
        if (batch)
        {
            if (batch->size() > 1)
                single.emplace(T{batch->at(batchIdx)});
            else
                single.emplace(T{batch->at(0)});
        }
    };

    extractOptional(samplingConfig.temperature, batchSamplingConfig.temperature);
    extractOptional(samplingConfig.originalTemperature, batchSamplingConfig.originalTemperature);
    extractOptional(samplingConfig.minLength, batchSamplingConfig.minLength);
    extractOptional(samplingConfig.repetitionPenalty, batchSamplingConfig.repetitionPenalty);
    extractOptional(samplingConfig.presencePenalty, batchSamplingConfig.presencePenalty);
    extractOptional(samplingConfig.frequencyPenalty, batchSamplingConfig.frequencyPenalty);
    extractOptional(samplingConfig.noRepeatNgramSize, batchSamplingConfig.noRepeatNgramSize);
    extractOptional(samplingConfig.topK, batchSamplingConfig.topK);
    extractOptional(samplingConfig.topP, batchSamplingConfig.topP);
    extractOptional(samplingConfig.randomSeed, batchSamplingConfig.randomSeed);
    extractOptional(samplingConfig.topPDecay, batchSamplingConfig.topPDecay);
    extractOptional(samplingConfig.topPMin, batchSamplingConfig.topPMin);
    extractOptional(samplingConfig.topPResetIds, batchSamplingConfig.topPResetIds);

    extractOptional(samplingConfig.beamSearchDiversityRate, batchSamplingConfig.beamSearchDiversityRate);
    extractOptional(samplingConfig.lengthPenalty, batchSamplingConfig.lengthPenalty);
    extractOptional(samplingConfig.earlyStopping, batchSamplingConfig.earlyStopping);
    samplingConfig.normalizeLogProbs = batchSamplingConfig.normalizeLogProbs;

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return samplingConfig;
}

}

GptDecoderBatched::GptDecoderBatched(std::size_t vocabSize, std::size_t vocabSizePadded,
    GptDecoderBatched::CudaStreamPtr stream, SpeculativeDecodingMode const& speculativeDecodingMode,
    nvinfer1::DataType dtype)
    : mVocabSize{vocabSize}
    , mVocabSizePadded{vocabSizePadded}
    , mRuntimeStream{std::move(stream)}
    , mBufferManager{mRuntimeStream}
    , mSpeculativeDecodingMode{speculativeDecodingMode}
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mJointDecodingInput;
    {
        auto dummyLogits = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
        auto endIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        auto batchSlots = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
        dInput = std::make_unique<DecodingInput>(
            0, 0, 0, 0, std::move(dummyLogits), std::move(endIds), std::move(batchSlots));
    }
    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mJointDecodingOutput;
    {
        auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        auto gatheredOutputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        dOutput = std::make_unique<DecodingOutput>(std::move(outputIds), std::move(gatheredOutputIds));
    }
    dOutput->newTokensSteps = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    mFinishedSteps
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mBatchSlotsSetup = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    mBatchSlotsDecoder = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dOutput->finishedSum = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    mFinishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvSizeType);
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->logProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);
    dOutput->finishReasons
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);

    dOutput->logProbsTiled = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);

    dInput->stopWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->stopWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->badWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->badWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->embeddingBias = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    mNumSMs = deviceProp.multiProcessorCount;

    if (!mSpeculativeDecodingMode.isNone())
    {
        allocateSpeculativeDecodingBuffers(dtype);
    }

    LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::allocateSpeculativeDecodingBuffers(nvinfer1::DataType dtype)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;

    auto& dInput = mJointDecodingInput;
    auto& dOutput = mJointDecodingOutput;

    if (mSpeculativeDecodingMode.isMedusa())
    {
        DecodingInput::MedusaInputs medusaInputs;
        medusaInputs.medusaPaths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTreeIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaCurTokensPerStep = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTargetTokensPerStep = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        dInput->medusaInputs = medusaInputs;
    }

    DecodingOutput::SpeculativeDecodingOutputs speculativeDecodingOutputs;
    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        speculativeDecodingOutputs.nextDraftTokens
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        if (mSpeculativeDecodingMode.variableDraftLength())
        {
            speculativeDecodingOutputs.nextDraftTokensLen
                = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            speculativeDecodingOutputs.prevDraftTokensLen
                = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        }
    }
    if (mSpeculativeDecodingMode.isLookaheadDecoding())
    {
        dInput->lookaheadInputs = DecodingInput::LookaheadInputs();
    }
    if (mSpeculativeDecodingMode.needsKVCacheRewind())
    {
        speculativeDecodingOutputs.acceptedTokensLen
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.acceptedLengthsCumSum
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.pathsOffsets
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    }
    dOutput->speculativeDecodingOutputs = speculativeDecodingOutputs;

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        DecodingInput::ExternalDraftTokensInputs externalDraftTokensInputs;

        externalDraftTokensInputs.draftLogits = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.draftProbs = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.targetProbs = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.numDraftTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        externalDraftTokensInputs.useDraftLogits
            = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<bool>::value);
        externalDraftTokensInputs.useDraftLogitsHost
            = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<bool>::value);
        externalDraftTokensInputs.draftTokenIds
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

        dInput->externalDraftTokensInputs = externalDraftTokensInputs;
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK(mSpeculativeDecodingMode.isExplicitDraftTokens());
    mJointDecodingOutput->explicitDraftTokensBuffers = std::move(explicitDraftTokensBuffers);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK(mSpeculativeDecodingMode.isLookaheadDecoding());
    mJointDecodingOutput->lookaheadOutputs = std::move(lookaheadDecodingBuffers);
    mJointDecodingInput->lookaheadInputs->tokensPerStep = mJointDecodingOutput->lookaheadOutputs->generationLengths;

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupEagle(EagleBuffers::Inputs eagleBuffers)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK(mSpeculativeDecodingMode.isEagle());
    mJointDecodingOutput->eagleBuffers = std::move(eagleBuffers);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::disableLookahead(SizeType32 maxBatchSize, RequestVector const& genRequests)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSpeculativeDecodingMode = SpeculativeDecodingMode::None();
    mMaxDecodingEngineTokens = 1;
    mMaxDecodingDecoderTokens = 1;
    mDecodingMode = executor::DecodingMode::TopKTopP();
    mJointDecodingInput->lookaheadInputs.reset();
    mJointDecodingOutput->newTokensSteps->reshape(ITensor::makeShape({1, maxBatchSize, 1}));
    mFinishedSteps->reshape(ITensor::makeShape({1, maxBatchSize, 1}));
    mBatchSlotsDecoder->reshape(ITensor::makeShape({1, maxBatchSize}));
    mNumDecodingEngineTokens.clear();
    mNumDecodingEngineTokens.resize(maxBatchSize, 0);

    std::vector<SamplingConfig> samplingConfigs;
    auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlotsSetup);
    SizeType32 bi = 0;
    for (auto const& llmReq : genRequests)
    {
        mNumDecodingEngineTokens[llmReq->mSeqSlot.value()] = 1;
        mMaxNewTokens[llmReq->mSeqSlot.value()] = mMaxSequenceLength - llmReq->getPromptLen();
        samplingConfigs.push_back(llmReq->mSamplingConfig);
        batchSlotsPtr[bi] = llmReq->mSeqSlot.value();
        bi += 1;
    }
    std::optional<SamplingConfig> samplingConfig;
    if (bi > 0)
    {
        samplingConfig = SamplingConfig(samplingConfigs);
    }
    TensorPtr batchSlotsView = ITensor::slice(mBatchSlotsSetup, 0, bi);
    mDecoder->disableLookahead(samplingConfig, bi, batchSlotsView);

    auto const& stream = mDecoderStream;
    CudaEvent event{};
    stream->record(event);
    mRuntimeStream->wait(event);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
    SizeType32 maxTokensPerEngineStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    CHECK(maxBatchSize > 0);
    CHECK(maxBeamWidth > 0);
    CHECK(maxTokensPerEngineStep > 0);
    CHECK(maxSequenceLength > 0);
    mActualBatchSize = maxBatchSize;
    mMaxSequenceLength = maxSequenceLength;
    mMaxAttentionWindow = maxAttentionWindow;
    mSinkTokenLength = sinkTokenLength;
    mMaxDecodingEngineTokens = maxTokensPerEngineStep;
    mDecodingMode = mode;

    CHECK_WITH_INFO((mMaxDecodingEngineTokens == 1 && mSpeculativeDecodingMode.isNone())
            || (mMaxDecodingEngineTokens > 1 && !mSpeculativeDecodingMode.isNone()),
        "Max tokens per engine step must be equal to 1 when no speculative decoding is configured, "
        "or > 1 for any speculative decoding mode");

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    auto const maxBatchSizeXmaxBeamWidth = ITensor::makeShape({maxBatchSize, maxBeamWidth});
    auto const maxTokensPerStepXmaxBatchSizeXmaxBeamWidth
        = ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize, maxBeamWidth});
    auto const maxBatchSizeXmaxTokensPerStep = ITensor::makeShape({maxBatchSize, maxTokensPerEngineStep});
    auto const jointOutputIdsShape = ITensor::makeShape({maxBatchSize, maxBeamWidth, maxSequenceLength});

    auto& dInput = *mJointDecodingInput;
    dInput.maxLength = mMaxSequenceLength;
    dInput.maxAttentionWindow = mMaxAttentionWindow;
    dInput.sinkTokenLength = mSinkTokenLength;
    dInput.stopWordsLists.resize(maxBatchSize);
    dInput.badWordsLists.resize(maxBatchSize);

    const_cast<ITensor&>(*dInput.endIds).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.batchSlots).reshape(maxBatchSizeShape);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(maxBatchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, *mRuntimeStream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(inputLengths);

    auto& dOutput = *mJointDecodingOutput;
    dOutput.ids->reshape(jointOutputIdsShape);

    if (maxBeamWidth > 1)
    {
        dOutput.gatheredIds->reshape(jointOutputIdsShape);

        mOutputBeamHypotheses = std::make_shared<DecodingOutput::BeamHypotheses>();
        mOutputBeamHypotheses->empty(mBufferManager);
        mOutputBeamHypotheses->reshape(1, maxBeamWidth, mMaxSequenceLength);
        mCumLogProbsTmp = mBufferManager.gpu(ITensor::makeShape({1, maxBeamWidth}), nvinfer1::DataType::kFLOAT);
    }
    else
    {
        dOutput.gatheredIds = dOutput.ids;
    }

    mBufferManager.setZero(*dOutput.newTokensSteps);
    mFinishedSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*mFinishedSteps);

    dOutput.finishReasons->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.finishReasons);

    mBatchSlotsSetup->reshape(ITensor::makeShape({maxBatchSize}));
    mBatchSlotsDecoder->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize}));

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        dInput.externalDraftTokensInputs->draftProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerEngineStep, maxBeamWidth, static_cast<SizeType32>(mVocabSizePadded)}));
        dInput.externalDraftTokensInputs->targetProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerEngineStep, maxBeamWidth, static_cast<SizeType32>(mVocabSizePadded)}));
        dInput.externalDraftTokensInputs->draftLogits->reshape(
            ITensor::makeShape({maxBatchSize, maxTokensPerEngineStep, static_cast<SizeType32>(mVocabSizePadded)}));
        dInput.externalDraftTokensInputs->draftTokenIds->reshape(maxBatchSizeXmaxTokensPerStep);
        dInput.externalDraftTokensInputs->numDraftTokens->reshape(ITensor::makeShape({maxBatchSize}));
        dInput.externalDraftTokensInputs->useDraftLogits->reshape(ITensor::makeShape({maxBatchSize}));
        dInput.externalDraftTokensInputs->useDraftLogitsHost->reshape(ITensor::makeShape({maxBatchSize}));
    }

    dOutput.parentIds->reshape(jointOutputIdsShape);
    dOutput.finishedSum->reshape(maxBatchSizeShape);
    mBufferManager.setZero(*dOutput.finishedSum);

    dOutput.newTokensSteps->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize, maxBeamWidth}));

    dOutput.cumLogProbs->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.cumLogProbs);

    dOutput.logProbs->reshape(jointOutputIdsShape);
    mBufferManager.setZero(*dOutput.logProbs);

    if (maxBeamWidth > 1)
    {
        dOutput.beamHypotheses.reshape(maxBatchSize, maxBeamWidth, mMaxSequenceLength);
    }

    dOutput.logProbsTiled->reshape(ITensor::makeShape({maxSequenceLength, maxBatchSize, maxBeamWidth}));
    mBufferManager.setZero(*dOutput.logProbsTiled);

    const_cast<ITensor&>(*dInput.embeddingBias)
        .reshape(ITensor::makeShape({maxBatchSize, static_cast<SizeType32>(mVocabSizePadded)}));
    const_cast<ITensor&>(*dInput.badWordsPtrs).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.badWordsLens).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.stopWordsPtrs).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.stopWordsLens).reshape(ITensor::makeShape({maxBatchSize}));

    std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModulePtr = nullptr;
    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        speculativeDecodingModulePtr = modelConfig.getSpeculativeDecodingModulePtr();
        setupSpeculativeDecoding(modelConfig);
    }
    else
    {
        mMaxDecodingDecoderTokens = 1;
    }

    auto const device = mRuntimeStream->getDevice();
    mDecoderStream = std::make_shared<CudaStream>();
    CHECK(mDecoderStream->getDevice() == device);

    mDecoder = IGptDecoder::create(mode, dtype, maxBatchSize, maxBeamWidth, mVocabSize, mVocabSizePadded,
        mMaxSequenceLength, mDecoderStream, speculativeDecodingModulePtr);

    mNbSteps.clear();
    mNbSteps.resize(maxBatchSize, 0);
    mFinished.clear();
    mFinished.resize(maxBatchSize, true);
    mMaxNewTokens.clear();
    mMaxNewTokens.resize(maxBatchSize, 0);
    mBeamWidths.clear();
    mBeamWidths.resize(maxBatchSize, 0);
    mNumDecodingEngineTokens.clear();
    mNumDecodingEngineTokens.resize(maxBatchSize, 0);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupSpeculativeDecoding(ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& dInput = *mJointDecodingInput;
    auto& dOutput = *mJointDecodingOutput;

    auto const speculativeDecodingModule = modelConfig.getSpeculativeDecodingModulePtr();
    if (mSpeculativeDecodingMode.isMedusa())
    {
        auto& medusaPaths = const_cast<ITensor&>(*dInput.medusaInputs->medusaPaths);
        medusaPaths.reshape(ITensor::makeShape({mActualBatchSize, speculativeDecodingModule->getMaxDecodingTokens(),
            speculativeDecodingModule->getMaxPathLen()}));
        mBufferManager.setMem(medusaPaths, -1);

        auto& medusaTreeIds = const_cast<ITensor&>(*dInput.medusaInputs->medusaTreeIds);
        medusaTreeIds.reshape(
            ITensor::makeShape({mActualBatchSize, speculativeDecodingModule->getMaxDecodingDraftTokens()}));
        mBufferManager.setZero(medusaTreeIds);
        auto& curTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaCurTokensPerStep);
        auto& targetTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaTargetTokensPerStep);
        curTokensPerStep.reshape(ITensor::makeShape({mActualBatchSize}));
        targetTokensPerStep.reshape(ITensor::makeShape({mActualBatchSize}));
        mBufferManager.setZero(curTokensPerStep);
        mBufferManager.setZero(targetTokensPerStep);
    }

    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        dOutput.speculativeDecodingOutputs->nextDraftTokens->reshape(
            ITensor::makeShape({mActualBatchSize, mMaxDecodingEngineTokens - 1}));
        if (mSpeculativeDecodingMode.variableDraftLength())
        {
            dOutput.speculativeDecodingOutputs->nextDraftTokensLen->reshape(ITensor::makeShape({mActualBatchSize}));
            dOutput.speculativeDecodingOutputs->prevDraftTokensLen->reshape(ITensor::makeShape({mActualBatchSize}));
        }
    }
    if (mSpeculativeDecodingMode.needsKVCacheRewind())
    {
        dOutput.speculativeDecodingOutputs->acceptedTokensLen->reshape(ITensor::makeShape({mActualBatchSize}));
        dOutput.speculativeDecodingOutputs->acceptedLengthsCumSum->reshape(ITensor::makeShape({mActualBatchSize + 1}));
        dOutput.speculativeDecodingOutputs->pathsOffsets->reshape(
            ITensor::makeShape({mActualBatchSize * speculativeDecodingModule->getMaxDraftPathLen()}));
    }

    mMaxDecodingDecoderTokens = mMaxDecodingEngineTokens;

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequest(SizeType32 batchSlot, decoder_batch::Request const& request,
    SamplingConfig const& samplingConfig, ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    CHECK(batchSlot >= 0);
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const batchSize = jointOutputIdsShape.d[0];
    CHECK(0 <= batchSize && batchSlot < batchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    auto const beamWidth = samplingConfig.beamWidth;
    CHECK_WITH_INFO(beamWidth <= maxBeamWidth,
        tc::fmtstr("Beam width (%d) must be smaller than maxBeamWidth (" FMT_DIM ") passed to decoder setup function.",
            beamWidth, maxBeamWidth));
    auto const& requestIds = request.ids;
    auto const inputLength = request.inputLen;
    auto const numDecodingEngineTokens = request.generatedTokensPerEngineStep;
    auto const numDecodingDraftEngineTokens = numDecodingEngineTokens - 1;
    auto const maxNewTokens
        = request.maxNewTokens.value_or(mMaxSequenceLength - inputLength - numDecodingDraftEngineTokens);

    CHECK_WITH_INFO(inputLength + maxNewTokens + numDecodingDraftEngineTokens <= mMaxSequenceLength,
        tc::fmtstr(
            "Input length (%d) + max new tokens (%d) + draft tokens (%d) must be less than max sequence length (%d).",
            inputLength, maxNewTokens, numDecodingDraftEngineTokens, mMaxSequenceLength));
    CHECK(requestIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = request.endId.value_or(-1);

    auto const& stream = mDecoderStream;
    BufferManager manager{stream};

    auto& dJointInput = *mJointDecodingInput;

    TensorPtr endIdTensorPtr{ITensor::slice(constPointerCast(dJointInput.endIds), batchSlot, 1)};
    kernels::invokeFill(*endIdTensorPtr, endId, *stream);

    TensorPtr embeddingBiasSlice = ITensor::slice(constPointerCast(dJointInput.embeddingBias), batchSlot, 1);
    if (request.embeddingBias)
    {
        CHECK(request.embeddingBias->getShape().nbDims == 2);
        CHECK(request.embeddingBias->getShape().d[0] == 1);
        CHECK_WITH_INFO(request.embeddingBias->getShape().d[1] == static_cast<SizeType32>(mVocabSize),
            "The embedding bias shape is not as expected. Expected last dimension to be same as vocab size: %lu.",
            mVocabSize);
        manager.copy(*request.embeddingBias, *embeddingBiasSlice);
    }
    else
    {
        manager.setZero(*embeddingBiasSlice);
    }

    auto setupWords = [](std::vector<runtime::ITensor::SharedPtr>& jointWordsLists, TensorPtr const& requestWordsList,
                          SharedConstPtr& jointWordsPtrs, SharedConstPtr& jointWordsLens, SizeType32& jointMaxWordsLen,
                          SizeType32 batchSlot)
    {
        if (requestWordsList)
        {
            auto const wordsLen = requestWordsList->getShape().d[1];
            BufferRange<int32_t*>(*constPointerCast(jointWordsPtrs))[batchSlot]
                = bufferCast<TokenIdType>(*requestWordsList);
            bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = wordsLen;
            jointMaxWordsLen = std::max(static_cast<SizeType32>(wordsLen), jointMaxWordsLen);

            jointWordsLists[batchSlot] = requestWordsList;
        }
        else
        {
            bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = 0;
        }
    };

    setupWords(dJointInput.stopWordsLists, request.stopWordsList, dJointInput.stopWordsPtrs, dJointInput.stopWordsLens,
        dJointInput.maxStopWordsLen, batchSlot);

    setupWords(dJointInput.badWordsLists, request.badWordsList, dJointInput.badWordsPtrs, dJointInput.badWordsLens,
        dJointInput.maxBadWordsLen, batchSlot);

    TensorPtr sequenceLimitLength{ITensor::slice(constPointerCast(dJointInput.sequenceLimitLength), batchSlot, 1)};
    kernels::invokeFill(*sequenceLimitLength, inputLength + maxNewTokens, *stream);

    TensorPtr inputLengths{ITensor::slice(constPointerCast(dJointInput.lengths), batchSlot, 1)};
    kernels::invokeFill(*inputLengths, inputLength, *stream);

    auto& dJointOutput = *mJointDecodingOutput;
    auto const outputIdsShape = ITensor::makeShape({1, beamWidth, mMaxSequenceLength});

    auto finishedSum = ITensor::slice(dJointOutput.finishedSum, batchSlot, 1);
    manager.setZero(*finishedSum);

    for (SizeType32 ti = 0; ti < mMaxDecodingEngineTokens; ++ti)
    {
        TensorPtr newTokensStepView = ITensor::slice(dJointOutput.newTokensSteps, ti, 1);
        newTokensStepView->squeeze(0);
        auto newTokensVec = ITensor::slice(newTokensStepView, batchSlot, 1);
        manager.setZero(*newTokensVec);
    }

    for (SizeType32 ti = 0; ti < mMaxDecodingEngineTokens; ++ti)
    {
        TensorPtr finishedStepsView = ITensor::slice(mFinishedSteps, ti, 1);
        finishedStepsView->squeeze(0);
        TensorPtr finishedSteps = ITensor::slice(finishedStepsView, batchSlot, 1);
        if (ti < numDecodingEngineTokens)
        {
            manager.setZero(*finishedSteps);
        }
        else
        {
            kernels::invokeFill(*finishedSteps, tk::FinishedState::skipDecoding().toUnderlying(), *stream);
        }
    }

    if ((samplingConfig.cumLogProbs.has_value() && samplingConfig.cumLogProbs->at(0)) || beamWidth > 1)
    {
        auto cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchSlot, 1);
        manager.setZero(*cumLogProbs);
    }

    if (samplingConfig.outputLogProbs.has_value() && samplingConfig.outputLogProbs->at(0))
    {
        auto logProbs = ITensor::slice(dJointOutput.logProbs, batchSlot, 1);
        manager.setZero(*logProbs);
    }

    if (beamWidth > 1)
    {
        TensorPtr cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchSlot, 1);
        kernels::invokeFill(*IBuffer::slice(cumLogProbs, 1, beamWidth - 1), DecodingOutput::kNegativeInfinity, *stream);

        auto parentIds = ITensor::slice(dJointOutput.parentIds, batchSlot, 1);
        parentIds->reshape(outputIdsShape);
        manager.setZero(*parentIds);

        auto beamHypotheses = dJointOutput.beamHypotheses.slice(batchSlot, 1);
        beamHypotheses.init(manager, endId);
    }

    if (numDecodingEngineTokens > 1 || mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        CHECK(beamWidth == 1);
        newRequestSpeculativeDecoding(batchSlot, request, samplingConfig, modelConfig);
    }

    mBeamWidths[batchSlot] = beamWidth;
    mNbSteps[batchSlot] = 0;
    mFinished[batchSlot] = false;
    mMaxNewTokens[batchSlot] = maxNewTokens;
    mNumDecodingEngineTokens[batchSlot] = numDecodingEngineTokens;

    auto const requestIdsShape = requestIds->getShape();
    auto inputIdsView = ITensor::view(requestIds, ITensor::makeShape({1, requestIdsShape.d[0]}));
    TensorPtr outputIds = ITensor::slice(dJointOutput.ids, batchSlot, 1);
    auto outputIdsView = ITensor::view(outputIds, ITensor::makeShape({beamWidth, mMaxSequenceLength}));
    kernels::invokeFill(*outputIdsView, endId, *stream);
    kernels::tileTensor(*outputIdsView, *inputIdsView, beamWidth, *stream);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequestSpeculativeDecoding(SizeType32 batchIdx, decoder_batch::Request const& request,
    SamplingConfig const& samplingConfig, ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        auto const& stream = mDecoderStream;
        BufferManager manager{stream};

        auto& dJointOutput = *mJointDecodingOutput;

        TensorPtr nextDraftTokens
            = ITensor::slice(dJointOutput.speculativeDecodingOutputs->nextDraftTokens, batchIdx, 1);
        manager.setZero(*nextDraftTokens);
        if (mSpeculativeDecodingMode.variableDraftLength())
        {
            TensorPtr nextDraftTokensLen
                = ITensor::slice(dJointOutput.speculativeDecodingOutputs->nextDraftTokensLen, batchIdx, 1);
            manager.setZero(*nextDraftTokensLen);
        }
    }

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        newRequestDraftTokensExternal(batchIdx, request, samplingConfig);
    }
    else if (mSpeculativeDecodingMode.isMedusa())
    {
        newRequestMedusa(batchIdx, request);
    }
    else if (mSpeculativeDecodingMode.isLookaheadDecoding())
    {
        newRequestLookahead(batchIdx, request);
    }
    else if (mSpeculativeDecodingMode.isExplicitDraftTokens())
    {
        newRequestExplicitDraftTokens(batchIdx, request);
    }
    else if (mSpeculativeDecodingMode.isEagle())
    {
        newRequestEagle(batchIdx, request, modelConfig);
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequestDraftTokensExternal(
    SizeType32 batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& stream = mDecoderStream;
    BufferManager manager{stream};

    auto& dJointInput = *mJointDecodingInput;

    auto const numDraftTokens = request.generatedTokensPerEngineStep - 1;

    auto const useDraftLogits = request.draftLogits.has_value();
    if (useDraftLogits)
    {
        TensorPtr draftLogitsView = ITensor::view(request.draftLogits.value());

        TensorPtr draftLogitsReqBatchSlice
            = ITensor::slice(dJointInput.externalDraftTokensInputs->draftLogits, batchIdx, 1);
        draftLogitsReqBatchSlice->squeeze(0);
        TensorPtr draftLogitsReqTokensSlice = ITensor::slice(draftLogitsReqBatchSlice, 0, numDraftTokens);
        manager.copy(*draftLogitsView, *draftLogitsReqTokensSlice);
    }
    auto* useDraftLogitsHostPtr = bufferCast<bool>(*dJointInput.externalDraftTokensInputs->useDraftLogitsHost);
    useDraftLogitsHostPtr[batchIdx] = useDraftLogits;
    auto useDraftLogitsView = ITensor::slice(dJointInput.externalDraftTokensInputs->useDraftLogits, batchIdx, 1);
    kernels::invokeFill(*useDraftLogitsView, useDraftLogits, *stream);

    if (numDraftTokens > 0)
    {
        TensorPtr draftTokensReqBatchSlice
            = ITensor::slice(dJointInput.externalDraftTokensInputs->draftTokenIds, batchIdx, 1);
        draftTokensReqBatchSlice->squeeze(0);
        TensorPtr draftTokensReqTokensSlice = ITensor::slice(draftTokensReqBatchSlice, 0, numDraftTokens);
        TensorPtr draftTokensView = ITensor::view(request.draftTokens, ITensor::makeShape({numDraftTokens}));
        manager.copy(*draftTokensView, *draftTokensReqTokensSlice);
    }

    auto numDraftTokensView = ITensor::slice(dJointInput.externalDraftTokensInputs->numDraftTokens, batchIdx, 1);
    kernels::invokeFill(*numDraftTokensView, numDraftTokens, *stream);

    bool const useRandomAcceptanceThreshold = !samplingConfig.draftAcceptanceThreshold.has_value();
    float const constantThreshold
        = useRandomAcceptanceThreshold ? 0 : samplingConfig.draftAcceptanceThreshold.value()[0];

    dJointInput.externalDraftTokensInputs->useRandomAcceptanceThreshold = useRandomAcceptanceThreshold;
    dJointInput.externalDraftTokensInputs->constantThreshold = constantThreshold;

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequestMedusa(SizeType32 batchIdx, decoder_batch::Request const& request)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& stream = mDecoderStream;
    BufferManager manager{stream};

    auto& dJointInput = *mJointDecodingInput;

    TensorPtr curTokensPerStepSlice
        = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaCurTokensPerStep), batchIdx, 1);
    kernels::invokeFill(*curTokensPerStepSlice, 1, *stream);
    TensorPtr targetTokensPerStepSlice
        = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaTargetTokensPerStep), batchIdx, 1);
    auto const generatedTokensPerEngineStep = request.generatedTokensPerEngineStep;
    CHECK_WITH_INFO(generatedTokensPerEngineStep <= mMaxDecodingEngineTokens,
        "Tokens per step for (%d) is larger than maximum tokens per step (%d)", generatedTokensPerEngineStep,
        mMaxDecodingEngineTokens);
    kernels::invokeFill(*targetTokensPerStepSlice, generatedTokensPerEngineStep, *stream);

    TensorPtr pathsSlice = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaPaths), batchIdx, 1);
    manager.copy(*request.medusaPaths, *pathsSlice);

    TensorPtr treeIdsSlice = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaTreeIds), batchIdx, 1);
    manager.copy(*request.medusaTreeIds, *treeIdsSlice);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequestLookahead(SizeType32 batchIdx, decoder_batch::Request const& request)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK(mJointDecodingOutput->lookaheadOutputs);

    auto& stream = mRuntimeStream;

    TensorPtr curTokensPerStepSlice
        = ITensor::slice(constPointerCast(mJointDecodingInput->lookaheadInputs->tokensPerStep), batchIdx, 1);
    kernels::invokeFill(*curTokensPerStepSlice, 1, *stream);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequestExplicitDraftTokens(SizeType32 batchIdx, decoder_batch::Request const& request)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK(mJointDecodingOutput->explicitDraftTokensBuffers);

    auto& stream = mRuntimeStream;

    TensorPtr positionIdsBaseSlice
        = ITensor::slice(mJointDecodingOutput->explicitDraftTokensBuffers->positionIdsBase, batchIdx, 1);
    kernels::invokeFill(*positionIdsBaseSlice, request.inputLen, *stream);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequestEagle(
    SizeType32 batchIdx, decoder_batch::Request const& request, ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK(mJointDecodingOutput->eagleBuffers);

    auto& stream = mRuntimeStream;
    BufferManager manager{stream};

    TensorPtr eagleNetCtxRequestTypesHostSlice
        = ITensor::slice(mJointDecodingOutput->eagleBuffers->eagleNetCtxRequestTypesHost, batchIdx, 1);
    TensorPtr eagleNetCtxContextLengthsHostSlice
        = ITensor::slice(mJointDecodingOutput->eagleBuffers->eagleNetCtxContextLengthsHost, batchIdx, 1);
    TensorPtr eagleNetCtxPastKeyValueLengthsHostSlice
        = ITensor::slice(mJointDecodingOutput->eagleBuffers->eagleNetCtxPastKeyValueLengthsHost, batchIdx, 1);

    bufferCast<SizeType32>(*eagleNetCtxRequestTypesHostSlice)[0] = 0;
    bufferCast<SizeType32>(*eagleNetCtxContextLengthsHostSlice)[0] = request.inputLen;
    bufferCast<SizeType32>(*eagleNetCtxPastKeyValueLengthsHostSlice)[0] = request.inputLen;

    TensorPtr eagleNetGenRequestTypesHostSlice
        = ITensor::slice(mJointDecodingOutput->eagleBuffers->eagleNetGenRequestTypesHost, batchIdx, 1);
    TensorPtr eagleNetGenContextLengthsHostSlice
        = ITensor::slice(mJointDecodingOutput->eagleBuffers->eagleNetGenContextLengthsHost, batchIdx, 1);
    TensorPtr eagleNetGenPastKeyValueLengthsHostSlice
        = ITensor::slice(mJointDecodingOutput->eagleBuffers->eagleNetGenPastKeyValueLengthsHost, batchIdx, 1);

    bufferCast<SizeType32>(*eagleNetGenRequestTypesHostSlice)[0] = 1;
    bufferCast<SizeType32>(*eagleNetGenContextLengthsHostSlice)[0] = request.inputLen;
    bufferCast<SizeType32>(*eagleNetGenPastKeyValueLengthsHostSlice)[0] = request.inputLen;

    auto const eagleModule = std::dynamic_pointer_cast<suggestify::runtime::EagleModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());
    std::optional<executor::EagleChoices> eagleChoicesOpt;
    if (request.eagleConfig)
    {
        eagleChoicesOpt = request.eagleConfig->getEagleChoices();
    }

    std::vector<SizeType32> topKs;
    TensorPtr draftPathsSlice = ITensor::slice(mJointDecodingOutput->eagleBuffers->draftPaths, batchIdx, 1);
    TensorPtr draftPathsHost = manager.pinnedPool(draftPathsSlice->getShape(), nvinfer1::DataType::kINT32);
    auto const depth = utils::initTensorsFromChoices(modelConfig.getSpeculativeDecodingModule(),
        eagleChoicesOpt.value_or(eagleModule->getDefaultEagleChoices()), topKs, nullptr, nullptr, nullptr,
        draftPathsHost, nullptr, {eagleModule->getMaxNonLeafNodesPerLayer()});
    CHECK_WITH_INFO(depth == modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(),
        "EAGLE-1 requires Eagle-tree depth being equal to the the number of build-time EAGLE layers.");

    manager.copy(*draftPathsHost, *draftPathsSlice);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setExplicitDraftTokensInputs(decoder_batch::Input const& input)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto explicitDraftTokensInputs = DecodingInput::ExplicitDraftTokensInputs();
    CHECK(input.explicitDraftTokensInputs.has_value());
    CHECK(input.explicitDraftTokensLastInputs.has_value());

    explicitDraftTokensInputs.nextDraftTokens = input.explicitDraftTokensInputs->nextDraftTokens;
    explicitDraftTokensInputs.nextFlatTokens = input.explicitDraftTokensInputs->nextFlatTokens;
    explicitDraftTokensInputs.nextDraftIndices = input.explicitDraftTokensInputs->nextDraftIndices;
    explicitDraftTokensInputs.nextDraftProbs = input.explicitDraftTokensInputs->nextDraftProbs;
    explicitDraftTokensInputs.lastDraftTokens = input.explicitDraftTokensLastInputs->draftTokens;
    explicitDraftTokensInputs.lastDraftIndices = input.explicitDraftTokensLastInputs->draftIndices;
    explicitDraftTokensInputs.lastPositionIdsBase = input.explicitDraftTokensLastInputs->positionIdsBase;
    explicitDraftTokensInputs.masks = input.explicitDraftTokensInputs->masks;
    explicitDraftTokensInputs.packedPositionIds = input.explicitDraftTokensInputs->packedPositionIds;
    explicitDraftTokensInputs.bestPathLengths = input.explicitDraftTokensInputs->bestPathLengths;
    explicitDraftTokensInputs.bestPathIndices = input.explicitDraftTokensInputs->bestPathIndices;
    explicitDraftTokensInputs.nextGenerationLengths = input.explicitDraftTokensInputs->nextGenerationLengths;
    explicitDraftTokensInputs.lastGenerationLengths = input.explicitDraftTokensLastInputs->generationLengths;
    explicitDraftTokensInputs.maxGenLengthDevice = input.explicitDraftTokensInputs->maxGenToken;
    explicitDraftTokensInputs.seqSlots = input.seqSlots;
    mJointDecodingInput->explicitDraftTokensInputs = explicitDraftTokensInputs;

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setEagleInputs(decoder_batch::Input const& input)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK(input.eagleInputs.has_value());
    CHECK(input.eagleLastInputs.has_value());

    auto eagleInputs = DecodingInput::EagleInputs(input.eagleInputs->nextDraftTokens, input.eagleInputs->nextDraftLens,
        input.eagleInputs->nextDraftPaths, input.eagleLastInputs->draftTokens, input.eagleLastInputs->draftLens,
        input.eagleLastInputs->draftPaths, input.eagleInputs->acceptedTokens, input.eagleInputs->acceptedLens,
        input.eagleInputs->acceptedPaths, input.eagleInputs->chunkedContextNextTokens, input.seqSlots);

    mJointDecodingInput->eagleInputs = eagleInputs;

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::newRequests(std::vector<SizeType32> const& seqSlots,
    std::vector<decoder_batch::Request> const& requests, std::vector<SamplingConfig> const& samplingConfigs,
    ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlotsSetup);
    SizeType32 const localBatchSize = seqSlots.size();
    for (SizeType32 bi = 0; bi < localBatchSize; ++bi)
    {
        newRequest(seqSlots[bi], requests[bi], samplingConfigs[bi], modelConfig);
        batchSlotsPtr[bi] = seqSlots[bi];
    }

    TensorPtr batchSlotsView = ITensor::slice(mBatchSlotsSetup, 0, localBatchSize);
    auto samplingConfig = SamplingConfig(samplingConfigs);
    mDecoder->setup(samplingConfig, localBatchSize, batchSlotsView, {*mJointDecodingOutput}, {requests});

    auto const& stream = mDecoderStream;
    CudaEvent event{};
    stream->record(event);
    mRuntimeStream->wait(event);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardDispatch(
    decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType)
{
    auto const maxDecodingEngineTokens
        = *std::max_element(std::begin(mNumDecodingEngineTokens), std::end(mNumDecodingEngineTokens));

    for (SizeType32 si = 0; si < maxDecodingEngineTokens; si += mMaxDecodingDecoderTokens)
    {
        forwardDecoder(si, output, input, forwardType);
    }
}

GptDecoderBatched::DecoderFinishedEventPtr GptDecoderBatched::forwardAsync(
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    forwardDispatch(output, input, ForwardType::kASYNC);

    CudaEvent eventStop{};
    mRuntimeStream->record(eventStop);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return std::make_unique<decoder_batch::DecoderFinishedEvent>(std::move(eventStop), input.active);
}

void GptDecoderBatched::forwardDecoder(
    SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto eventStart = CudaEvent{};
    mRuntimeStream->record(eventStart);

    auto& allTargetLogits = input.logits;
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];

    auto constexpr singleRequest = 1;

    CHECK(static_cast<SizeType32>(output.sequenceLengths->getSize()) == mActualBatchSize * maxBeamWidth);
    TensorPtr sequenceLengths
        = ITensor::view(output.sequenceLengths, ITensor::makeShape({mActualBatchSize, maxBeamWidth}));
    CHECK(sequenceLengths);

    auto batchSlotsDecoderPtr = maxBeamWidth > 1 && input.seqSlots ? bufferCast<SizeType32>(*input.seqSlots)
                                                                   : bufferCast<SizeType32>(*mBatchSlotsDecoder);
    auto& dInput = *mJointDecodingInput;
    auto& dOutput = *mJointDecodingOutput;
    auto& decoder = *mDecoder;
    auto const& stream = mDecoderStream;

    if (maxBeamWidth > 1)
    {
        dInput.cacheIndirection = input.cacheIndirection;
        dOutput.cacheIndirection = output.cacheIndirection;
    }

    if (mSpeculativeDecodingMode.isExplicitDraftTokens())
    {
        setExplicitDraftTokensInputs(input);
    }
    else if (mSpeculativeDecodingMode.isEagle())
    {
        setEagleInputs(input);
    }

    bool const async = forwardType == ForwardType::kASYNC;

    if (async)
    {
        stream->wait(eventStart.get());
    }

    SizeType32 localBatchDecoderIdx = 0;
    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }
        batchSlotsDecoderPtr[step * mActualBatchSize + localBatchDecoderIdx] = bi;
        localBatchDecoderIdx++;
    }

    auto const maxDecodingEngineTokens
        = *std::max_element(std::begin(mNumDecodingEngineTokens), std::end(mNumDecodingEngineTokens));

    std::vector<SharedConstPtr> logitsVec;
    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }
        auto const& targetLogits = allTargetLogits[bi];
        TensorPtr logitsSlice = ITensor::slice(targetLogits, step, singleRequest);
        logitsVec.push_back(logitsSlice);
    }

    TensorPtr finishedStepsInput = ITensor::slice(mFinishedSteps, step, 1);
    TensorPtr finishedStepsOutput = ITensor::slice(mFinishedSteps, std::min(maxDecodingEngineTokens - 1, step + 1), 1);
    finishedStepsInput->squeeze(0);
    finishedStepsOutput->squeeze(0);
    TensorPtr newTokensStepView = ITensor::slice(dOutput.newTokensSteps, step, mMaxDecodingDecoderTokens);

    dInput.logitsVec = logitsVec;
    dInput.finishReasons = finishedStepsInput;

    if (maxBeamWidth > 1 && input.seqSlots)
    {
        dInput.batchSlots = input.seqSlots;
    }
    else
    {
        TensorPtr batchSlotsDecoderSlice = ITensor::slice(mBatchSlotsDecoder, step, 1);
        batchSlotsDecoderSlice->squeeze(0);
        dInput.batchSlots = batchSlotsDecoderSlice;
    }

    dInput.batchSize = localBatchDecoderIdx;
    if (mSpeculativeDecodingMode.isMedusa())
    {
        dInput.medusaInputs->medusaLogits = input.predictedDraftLogits;
    }

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        dInput.externalDraftTokensInputs->step = step;
    }

    dOutput.newTokens = newTokensStepView;
    dOutput.finishReasons = finishedStepsOutput;
    dOutput.lengths = sequenceLengths;

    if (localBatchDecoderIdx > 0)
    {
        if (forwardType == ForwardType::kASYNC)
        {
            decoder.forwardAsync(dOutput, dInput);
        }
        else if (forwardType == ForwardType::kSYNC)
        {
            decoder.forwardSync(dOutput, dInput);
        }
        else
        {
            THROW("Unknown ForwardType");
        }
    }

    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }
        mNbSteps[bi] += 1;
        mFinished[bi] = mNbSteps[bi] >= mMaxNewTokens[bi];
    }

    if (async && step == maxDecodingEngineTokens - mMaxDecodingDecoderTokens)
    {
        CudaEvent event{};
        stream->record(event);
        mRuntimeStream->wait(event);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::updateFinished(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    for (std::int32_t i = 0; i < mActualBatchSize; ++i)
    {
        if (decoderFinishEvent.active[i] && !mFinished[i])
        {
            auto finishedSum = ITensor::slice(mJointDecodingOutput->finishedSum, i, 1);
            mFinished[i]
                = mFinished[i] || bufferCast<SizeType32>(*finishedSum)[0] == static_cast<SizeType32>(mBeamWidths[i]);
        }
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardSync(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    decoderFinishEvent.event.synchronize();

    updateFinished(decoderFinishEvent);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardSync(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent,
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    decoderFinishEvent.event.synchronize();

    forwardDispatch(output, input, ForwardType::kSYNC);

    updateFinished(decoderFinishEvent);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatched::postProcessRequest(
    SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& stream = mRuntimeStream;
    auto manager = BufferManager{stream};

    auto& dJointInput = *mJointDecodingInput;
    auto& dJointOutput = *mJointDecodingOutput;

    auto slice = [batchSlot](auto& a, auto const& b)
    {
        if (b && b->getShape().d[0] > 0)
        {
            a = ITensor::slice(b, batchSlot, 1);
        }
    };

    DecodingInput dInput{dJointInput};
    slice(dInput.endIds, dJointInput.endIds);
    slice(dInput.lengths, dJointInput.lengths);

    DecodingOutput dOutput{
        ITensor::slice(dJointOutput.ids, batchSlot, 1), ITensor::slice(dJointOutput.gatheredIds, batchSlot, 1)};
    dOutput.beamHypotheses = dJointOutput.beamHypotheses.slice(batchSlot, 1);
    slice(dOutput.parentIds, dJointOutput.parentIds);
    slice(dOutput.cumLogProbs, dJointOutput.cumLogProbs);
    slice(dOutput.cacheIndirection, dJointOutput.cacheIndirection);
    slice(dOutput.lengths, dJointOutput.lengths);
    slice(dOutput.finishReasons, dJointOutput.finishReasons);
    slice(dOutput.logProbs, dJointOutput.logProbs);

    dOutput.newTokens = ITensor::view(dJointOutput.newTokens);
    CHECK(dOutput.newTokens->getShape().d[0] == 1);
    dOutput.newTokens->squeeze(0);
    dOutput.newTokens = ITensor::slice(dOutput.newTokens, batchSlot, 1);
    dOutput.logProbsTiled = dJointOutput.logProbsTiled;
    if (streaming)
    {
        suggestify::kernels::invokeCopyBeamHypotheses(
            dOutput.beamHypotheses, *mOutputBeamHypotheses, *dOutput.cumLogProbs, *mCumLogProbsTmp, *stream, mNumSMs);
        dOutput.beamHypotheses = *mOutputBeamHypotheses;
        dOutput.cumLogProbs = mCumLogProbsTmp;
    }

    kernels::gatherTree(dOutput, dInput, manager, samplingConfig);

    CudaEvent event{};
    stream->record(event);
    mRuntimeStream->wait(event);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}

void GptDecoderBatched::newBatch(GenerationInput const& inputs, GenerationOutput const& outputs,
    SamplingConfig const& samplingConfig, ModelConfig const& modelConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& inputLengths = inputs.lengths;
    mActualBatchSize = inputLengths->getShape().d[0];
    mNumDecodingEngineTokens.resize(mActualBatchSize);

    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBatchSize = jointOutputIdsShape.d[0];
    CHECK(mActualBatchSize <= maxBatchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    CHECK(samplingConfig.beamWidth <= maxBeamWidth);

    auto const inputIdsShape = inputs.ids->getShape();
    TensorPtr inputIdsFlatView = ITensor::view(inputs.ids);

    TensorPtr batchSlotsView = ITensor::slice(mBatchSlotsSetup, 0, mActualBatchSize);
    auto batchSlots = BufferRange<SizeType32>(*batchSlotsView);
    std::iota(batchSlots.begin(), batchSlots.end(), 0);

    if (inputs.packed && inputIdsShape.nbDims == 2)
    {
        inputIdsFlatView->squeeze(0);
    }
    auto inputLengthsHost = mBufferManager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto inputLengthsPtr = bufferCast<SizeType32>(*inputLengthsHost);
    auto inputOffset = 0;
    std::vector<SamplingConfig> samplingConfigs;
    for (auto batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        mNumDecodingEngineTokens[batchIdx] = 1;
        auto const inputLength = inputLengthsPtr[batchIdx];
        auto const inputShape = ITensor::makeShape({inputLength});
        TensorPtr inputView;
        if (inputs.packed)
        {
            CHECK(inputIdsFlatView->getShape().nbDims == 1);
            inputView = ITensor::slice(inputIdsFlatView, inputOffset, inputLength);
            inputOffset += inputLength;
        }
        else
        {
            inputView = ITensor::slice(inputs.ids, batchIdx, 1);
            inputView->reshape(inputShape);
        }
        auto request = decoder_batch::Request{inputView, inputLength, inputs.maxNewTokens, inputs.endId};

        if (inputs.embeddingBias)
        {
            THROW("newBatch doesn't support embeddingBias yet.");
        }
        if (inputs.badWordsList)
        {
            auto const& shape = inputs.badWordsList->getShape();
            if (shape.nbDims == 2)
            {
                request.badWordsList = inputs.badWordsList;
            }
            else
            {
                assert(shape.nbDims == 3);
                TensorPtr badWordsListView = ITensor::slice(inputs.badWordsList, batchIdx, 1);
                badWordsListView->squeeze(0);
                request.badWordsList = badWordsListView;
            }
        }
        if (inputs.stopWordsList)
        {
            TensorPtr stopWordsListView = ITensor::slice(inputs.stopWordsList, batchIdx, 1);
            stopWordsListView->squeeze(0);
            request.stopWordsList = stopWordsListView;
        }
        auto requestSamplingConfig = extractSamplingConfig(samplingConfig, batchIdx);
        requestSamplingConfig.cumLogProbs = {{outputs.cumLogProbs != nullptr}};
        requestSamplingConfig.outputLogProbs = {{outputs.logProbs != nullptr}};
        newRequest(batchIdx, request, requestSamplingConfig, modelConfig);
        samplingConfigs.push_back(requestSamplingConfig);
    }

    auto fusedSamplingConfig = samplingConfig;
    fusedSamplingConfig.cumLogProbs = std::vector<bool>(mActualBatchSize, outputs.cumLogProbs != nullptr);
    fusedSamplingConfig.outputLogProbs = std::vector<bool>(mActualBatchSize, outputs.logProbs != nullptr);

    mDecoder->setup(fusedSamplingConfig, mActualBatchSize, batchSlotsView, {*mJointDecodingOutput});

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& logitsShape = input.logits->getShape();
    auto const batchSize = logitsShape.d[0];
    auto constexpr singleRequest = 1;
    std::vector<ITensor::SharedPtr> logits;
    logits.reserve(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto logitsSlice = std::shared_ptr(ITensor::slice(input.logits, batchIdx, singleRequest));
        logits.emplace_back(
            ITensor::view(logitsSlice, ITensor::makeShape({singleRequest, mBeamWidths[batchIdx], logitsShape.d[2]})));
    }

    decoder_batch::Input batchInput{logits};
    batchInput.cacheIndirection = input.cacheIndirection;

    decoder_batch::Output batchOutput;
    batchOutput.cacheIndirection = output.cacheIndirection;
    batchOutput.sequenceLengths = output.sequenceLengths;

    mDecoderFinishEvent = forwardAsync(batchOutput, batchInput);
    mBufferManager.setZero(*mFinishedSum);
    kernels::reduce(
        *mFinishedSum, *ITensor::slice(mJointDecodingOutput->finishedSum, 0, mActualBatchSize), *mRuntimeStream);
    mRuntimeStream->record(mForwardEvent);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardSync()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    forwardSync(*mDecoderFinishEvent);
    mForwardEvent.synchronize();
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::finalize(SamplingConfig const& samplingConfig) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto batchSlots = bufferCast<SizeType32>(*mBatchSlotsSetup);
    for (SizeType32 batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        auto slot = batchSlots[batchIdx];
        auto requestSamplingConfig = extractSamplingConfig(samplingConfig, slot);
        auto event = postProcessRequest(slot, requestSamplingConfig, false);
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatched::finalize(SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto event = postProcessRequest(batchSlot, samplingConfig, streaming);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}
