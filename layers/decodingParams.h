
#pragma once

#include "suggestify/common/assert.h"
#include "suggestify/executor/executor.h"
#include "../src/beamSearchKernels.h"
#include "iTensor.h"
#include <common.h>
#include <speculativeDecodingModule.h>

#include <optional>
#include <utility>
#include <vector>

namespace suggestify::layers
{

using TensorPtr = runtime::ITensor::SharedPtr;
using TensorConstPtr = runtime::ITensor::SharedConstPtr;
using BufferPtr = runtime::IBuffer::SharedPtr;
using BufferConstPtr = runtime::IBuffer::SharedConstPtr;


class DecoderDomain
{
public:
    DecoderDomain(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 vocabSize,
        std::optional<runtime::SizeType32> vocabSizePadded = std::nullopt,
        std::shared_ptr<runtime::SpeculativeDecodingModule const> speculativeDecodingModule = nullptr)
        : mBatchSize(batchSize)
        , mBeamWidth(beamWidth)
        , mVocabSize(vocabSize)
        , mVocabSizePadded(vocabSizePadded.value_or(vocabSize))
        , mSpeculativeDecodingModule(std::move(speculativeDecodingModule))
    {
    }

    [[nodiscard]] runtime::SizeType32 getBatchSize() const
    {
        return mBatchSize;
    }

    [[nodiscard]] runtime::SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] runtime::SizeType32 getVocabSize() const
    {
        return mVocabSize;
    }

    [[nodiscard]] runtime::SizeType32 getVocabSizePadded() const
    {
        return mVocabSizePadded;
    }

    [[nodiscard]] runtime::SizeType32 getMaxDecodingTokens() const
    {
        return mSpeculativeDecodingModule ? mSpeculativeDecodingModule->getMaxDecodingTokens() : 1;
    }

    [[nodiscard]] std::shared_ptr<runtime::SpeculativeDecodingModule const> getSpeculativeDecodingModule() const
    {
        CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set to decoder domain");
        return mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<runtime::SpeculativeDecodingModule const> getSpeculativeDecodingModulePtr() const
    {
        return mSpeculativeDecodingModule;
    }

private:
    runtime::SizeType32 mBatchSize;
    runtime::SizeType32 mBeamWidth;
    runtime::SizeType32 mVocabSize;
    runtime::SizeType32 mVocabSizePadded;
    std::shared_ptr<runtime::SpeculativeDecodingModule const> mSpeculativeDecodingModule;
};

class BaseSetupParams
{
public:
    virtual ~BaseSetupParams() = default;
};

class PenaltySetupParams : public BaseSetupParams
{
public:
    std::optional<std::vector<float>> temperature;
    std::optional<std::vector<runtime::SizeType32>> minLength;
    std::optional<std::vector<float>> repetitionPenalty;
    std::optional<std::vector<float>> presencePenalty;
    std::optional<std::vector<float>> frequencyPenalty;
};

class BanWordsSetupParams : public BaseSetupParams
{
public:
    std::optional<std::vector<runtime::SizeType32>> noRepeatNgramSize;
};

class DecodingSetupParams : public BaseSetupParams
{
public:
    virtual ~DecodingSetupParams() = default;

    std::optional<std::vector<uint64_t>> randomSeed;
    std::optional<std::vector<bool>> outputLogProbs;
    std::optional<std::vector<bool>> cumLogProbs;
};

class SamplingSetupParams : public DecodingSetupParams
{
public:
    std::optional<std::vector<runtime::SizeType32>> runtimeTopK;
    std::optional<std::vector<float>> runtimeTopP;

    std::optional<std::vector<float>> topPDecay;
    std::optional<std::vector<float>> topPMin;
    std::optional<std::vector<runtime::TokenIdType>> topPResetIds;
    std::optional<bool> normalizeLogProbs;
};

class BeamSearchSetupParams : public DecodingSetupParams
{
public:
    std::optional<std::vector<float>> beamSearchDiversityRate;
    std::optional<std::vector<float>> lengthPenalty;
    std::optional<std::vector<int>> earlyStopping;
    bool hasDiffRuntimeArgs{false};
};

class MedusaSetupParams : public DecodingSetupParams
{
public:
    std::optional<std::vector<runtime::SizeType32>> runtimeTopK;
    std::optional<std::vector<std::vector<runtime::SizeType32>>> runtimeHeadsTopK;
};

class ExplicitDraftTokensSetupParams : public DecodingSetupParams
{
public:
    std::optional<std::vector<float>> temperature;
    TensorPtr randomDataSample;
    TensorPtr temperatures;
    nvinfer1::DataType dtype;
};

class EagleSetupParams : public DecodingSetupParams
{
public:
    std::optional<std::vector<float>> temperature;
    TensorPtr randomDataSample;
    TensorPtr temperatures;
    nvinfer1::DataType dtype;
};

class DynamicDecodeSetupParams : public BaseSetupParams
{
public:
    std::shared_ptr<PenaltySetupParams> penaltyParams;

    std::shared_ptr<BanWordsSetupParams> banWordsParams;

    std::shared_ptr<DecodingSetupParams> decodingParams;
};

struct LookaheadSetupParams : public DecodingSetupParams
{
    using TensorPtr = runtime::ITensor::SharedPtr;

    std::vector<runtime::ITensor::SharedConstPtr> prompt;
    std::vector<executor::LookaheadDecodingConfig> algoConfigs;

    TensorPtr generationLengths;
    TensorPtr positionOffsets;
    TensorPtr attentionPackedMasks;
};

class ExternalDraftTokensSetupParams : public DecodingSetupParams
{
public:
    std::optional<std::vector<runtime::SizeType32>> runtimeTopK;
    std::optional<std::vector<float>> runtimeTopP;
};

class BaseDecodingInputs
{
public:
    BaseDecodingInputs(runtime::SizeType32 localBatchSize)
        : localBatchSize(localBatchSize)
    {
    }

    virtual ~BaseDecodingInputs() = default;

    runtime::SizeType32 localBatchSize;
};

class BanWordsDecodingInputs : public BaseDecodingInputs
{
public:
    BanWordsDecodingInputs(runtime::SizeType32 localBatchSize)
        : BaseDecodingInputs(localBatchSize)
    {
    }

    runtime::SizeType32 maxBadWordsLen{0};
    std::optional<TensorConstPtr> badWordsPtr;
    std::optional<TensorConstPtr> badWordsLengths;
};

class StopCriteriaDecodingInputs : public BaseDecodingInputs
{
public:
    StopCriteriaDecodingInputs(runtime::SizeType32 localBatchSize)
        : BaseDecodingInputs(localBatchSize)
    {
    }

    runtime::SizeType32 maxStopWordsLen{0};
    std::optional<TensorConstPtr> sequenceLimitLength;
    std::optional<TensorConstPtr> stopWordsPtr;
    std::optional<TensorConstPtr> stopWordsLengths;
};

class DecodingInputs : public BaseDecodingInputs
{
public:
    DecodingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 step = 0,
        runtime::SizeType32 ite = 0, runtime::SizeType32 localBatchSize = 0, runtime::SizeType32 maxAttentionWindow = 0,
        runtime::SizeType32 sinkTokenLength = 0)
        : BaseDecodingInputs(localBatchSize)
        , endIds{std::move(endIds)}
        , step{step}
        , ite{ite}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , batchSlots{std::move(batchSlots)}
    {
    }

    TensorConstPtr endIds;

    runtime::SizeType32 step;
    runtime::SizeType32 ite;

    runtime::SizeType32 maxAttentionWindow;
    runtime::SizeType32 sinkTokenLength;

    std::optional<TensorConstPtr> logits;
    std::optional<std::vector<TensorConstPtr>> logitsVec;
    TensorConstPtr batchSlots;

    std::optional<TensorPtr> srcCacheIndirection;
    std::optional<TensorConstPtr> embeddingBias;
    std::optional<TensorConstPtr> inputLengths;
    std::optional<TensorConstPtr> finished;
    std::optional<TensorPtr> curTokensPerStep;

    std::shared_ptr<BanWordsDecodingInputs> banWordsInputs;

    std::shared_ptr<StopCriteriaDecodingInputs> stopCriteriaInputs;
};

class SamplingInputs : public DecodingInputs
{
public:
    explicit SamplingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 step,
        runtime::SizeType32 ite, runtime::SizeType32 localBatchSize)
        : DecodingInputs{std::move(endIds), std::move(batchSlots), step, ite, localBatchSize}
    {
    }

    curandState_t* curandStates{};

    bool probsComputed{};
};

class ExternalDraftTokensInputs : public DecodingInputs
{
public:
    explicit ExternalDraftTokensInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 step,
        runtime::SizeType32 ite, runtime::SizeType32 localBatchSize)
        : DecodingInputs{std::move(endIds), std::move(batchSlots), step, ite, localBatchSize}
    {
    }

    TensorPtr draftLogits;
    TensorPtr draftProbs;
    TensorPtr targetProbs;
    TensorPtr numDraftTokens;
    TensorPtr draftTokenIds;
    TensorPtr useDraftLogits;
    TensorPtr useDraftLogitsHost;
    runtime::SizeType32 step;
    float constantThreshold;
    bool useRandomAcceptanceThreshold;

    curandState_t* curandStates{};

    bool probsComputed{};
};

class MedusaDecodingInputs : public DecodingInputs
{
public:
    explicit MedusaDecodingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 localBatchSize)
        : DecodingInputs(std::move(endIds), std::move(batchSlots), 0, 0, localBatchSize)
    {
    }

    TensorConstPtr targetTokensPerStep;
    TensorConstPtr paths;
    TensorConstPtr treeIds;
    std::vector<std::vector<TensorPtr>> medusaLogits;
};

class ExplicitDraftTokensInputs : public DecodingInputs
{
public:
    explicit ExplicitDraftTokensInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 batchSize)
        : DecodingInputs(std::move(endIds), std::move(batchSlots), 0, 0, batchSize)
    {
    }

    TensorConstPtr nextDraftTokens;
    TensorConstPtr nextFlatTokens;
    TensorConstPtr nextDraftIndices;
    TensorConstPtr nextDraftProbs;
    TensorConstPtr lastDraftTokens;
    TensorConstPtr lastDraftIndices;
    TensorConstPtr masks;
    TensorConstPtr packedPosIds;
    TensorConstPtr bestPathLengths;
    TensorConstPtr bestPathIndices;
    TensorConstPtr generationLengths;
    TensorConstPtr positionIdsBase;
    TensorConstPtr lastGenerationLengths;
    TensorConstPtr maxGenLengthDevice;
    TensorConstPtr seqSlots;
};

class LookaheadDecodingInputs : public DecodingInputs
{
public:
    explicit LookaheadDecodingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots)
        : DecodingInputs{std::move(endIds), std::move(batchSlots)}
    {
    }
};

class EagleInputs : public DecodingInputs
{
public:
    explicit EagleInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 batchSize,
        TensorConstPtr nextDraftTokens, TensorConstPtr nextDraftLens, TensorConstPtr nextDraftPaths,
        TensorConstPtr lastDraftTokens, TensorConstPtr lastDraftLens, TensorConstPtr lastDraftPaths,
        TensorConstPtr acceptedTokens, TensorConstPtr acceptedLens, TensorConstPtr acceptedPathIds,
        TensorConstPtr chunkedContextNextTokens, TensorConstPtr seqSlots)
        : DecodingInputs(std::move(endIds), std::move(batchSlots), 0, 0, batchSize)
        , nextDraftTokens(nextDraftTokens)
        , nextDraftLens(nextDraftLens)
        , nextDraftPaths(nextDraftPaths)
        , lastDraftTokens(lastDraftTokens)
        , lastDraftLens(lastDraftLens)
        , lastDraftPaths(lastDraftPaths)
        , acceptedTokens(acceptedTokens)
        , acceptedLens(acceptedLens)
        , acceptedPathIds(acceptedPathIds)
        , chunkedContextNextTokens(chunkedContextNextTokens)
        , seqSlots(seqSlots)
    {
    }

    TensorConstPtr nextDraftTokens;
    TensorConstPtr nextDraftLens;
    TensorConstPtr nextDraftPaths;
    TensorConstPtr lastDraftTokens;
    TensorConstPtr lastDraftLens;
    TensorConstPtr lastDraftPaths;

    TensorConstPtr acceptedTokens;
    TensorConstPtr acceptedLens;
    TensorConstPtr acceptedPathIds;
    TensorConstPtr chunkedContextNextTokens;

    TensorConstPtr seqSlots;
};

class BaseDecodingOutputs
{
public:
    explicit BaseDecodingOutputs(TensorPtr outputIds)
        : outputIds{std::move(outputIds)}
    {
    }

    virtual ~BaseDecodingOutputs() = default;

    TensorPtr outputIds;

    std::optional<TensorPtr> finished;
    std::optional<TensorPtr> sequenceLength;
    std::optional<TensorPtr> cumLogProbs;
    std::optional<TensorPtr> outputLogProbs;
    std::optional<TensorPtr> parentIds;

    TensorPtr outputIdsPtr;
    TensorPtr parentIdsPtr;

    TensorPtr newTokens;

    std::optional<TensorPtr> numNewTokens;
    std::optional<TensorPtr> finishedSum;
    std::optional<TensorPtr> outputLogProbsTiled;
};

class BeamSearchOutputs : public BaseDecodingOutputs
{
public:
    explicit BeamSearchOutputs(TensorPtr outputIds)
        : BaseDecodingOutputs{std::move(outputIds)}
    {
    }

    TensorPtr tgtCacheIndirection;
    std::unique_ptr<kernels::BeamHypotheses> beamHypotheses;
};

class SpeculativeDecodingOutputs : public BaseDecodingOutputs
{
public:
    explicit SpeculativeDecodingOutputs(TensorPtr outputIds)
        : BaseDecodingOutputs{std::move(outputIds)}
    {
    }

    TensorPtr nextDraftTokens;
    TensorPtr nextDraftPosIds;
    TensorPtr prevDraftLengths;
    TensorPtr nextDraftLengths;
    TensorPtr numNewTokensCumSum;
    TensorPtr pathsOffsets;
    TensorPtr packedMasks;
};

class LookaheadDecodingOutputs : public SpeculativeDecodingOutputs
{
    using TensorPtr = runtime::ITensor::SharedPtr;

public:
    explicit LookaheadDecodingOutputs(TensorPtr outputIds)
        : SpeculativeDecodingOutputs{std::move(outputIds)}
    {
    }

    TensorPtr generationLengths;
    TensorPtr positionOffsets;
    TensorPtr positionIds;
};

class ExplicitDraftTokensOutputs : public SpeculativeDecodingOutputs
{
public:
    explicit ExplicitDraftTokensOutputs(TensorPtr outputIds)
        : SpeculativeDecodingOutputs{std::move(outputIds)}
    {
    }

    TensorPtr unpackedNextDraftTokens;
    TensorPtr unpackedNextDraftIndices;
    TensorPtr nextDraftProbs;
    TensorPtr positionIdsBase;
    TensorPtr randomDataSample;
    TensorPtr randomDataValidation;
    TensorPtr temperatures;
    TensorPtr generationLengths;
    TensorPtr generationLengthsHost;
    TensorPtr maxGenLengthHost;
};

class EagleOutputs : public SpeculativeDecodingOutputs
{
public:
    explicit EagleOutputs(TensorPtr outputIds)
        : SpeculativeDecodingOutputs{std::move(outputIds)}
    {
    }

    TensorPtr unpackedNextDraftTokens;
    TensorPtr nextDraftPaths;
    TensorPtr randomDataSample;
    TensorPtr randomDataValidation;
    TensorPtr temperatures;
    TensorPtr generationLengths;
    TensorPtr generationLengthsHost;

    TensorPtr eagleNetCtxRequestTypesHost;
    TensorPtr eagleNetCtxContextLengthsHost;
    TensorPtr eagleNetCtxPastKeyValueLengthsHost;
    TensorPtr eagleNetGenRequestTypesHost;
    TensorPtr eagleNetGenContextLengthsHost;
    TensorPtr eagleNetGenPastKeyValueLengthsHost;
};

}
