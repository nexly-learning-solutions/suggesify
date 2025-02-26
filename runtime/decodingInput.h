
#pragma once

#include "common.h"
#include "iTensor.h"

#include <optional>

namespace suggestify::runtime
{

class DecodingInput
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    DecodingInput(SizeType32 maxLength, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 batchSize,
        TensorConstPtr logits, TensorPtr endIds, TensorConstPtr batchSlots)
        : step{maxLength}
        , maxLength{maxLength}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , batchSize{batchSize}
        , maxStopWordsLen{0}
        , maxBadWordsLen{0}
        , logits{std::move(logits)}
        , endIds{std::move(endIds)}
        , batchSlots{std::move(batchSlots)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->endIds), "Invalid endIds tensor");
    }


    SizeType32 step;

    SizeType32 maxLength;

    SizeType32 maxAttentionWindow;

    SizeType32 sinkTokenLength;

    SizeType32 batchSize;

    SizeType32 maxStopWordsLen;

    SizeType32 maxBadWordsLen;

    TensorConstPtr logits;
    std::optional<std::vector<TensorConstPtr>>
        logitsVec;

    TensorConstPtr endIds;

    TensorConstPtr
        batchSlots;

    TensorConstPtr finishReasons;
    TensorConstPtr
        sequenceLimitLength;
    TensorConstPtr embeddingBias;
    TensorConstPtr lengths;
    std::vector<TensorPtr> badWordsLists;
    TensorConstPtr badWordsPtrs;
    TensorConstPtr badWordsLens;
    std::vector<TensorPtr> stopWordsLists;
    TensorConstPtr stopWordsPtrs;
    TensorConstPtr stopWordsLens;
    TensorConstPtr noRepeatNgramSize;

    TensorPtr cacheIndirection;

    class MedusaInputs
    {
    public:
        TensorConstPtr medusaPaths;
        TensorConstPtr medusaTreeIds;
        std::vector<std::vector<TensorPtr>>
            medusaLogits;
        TensorPtr medusaCurTokensPerStep;
        TensorConstPtr medusaTargetTokensPerStep;
    };

    class ExternalDraftTokensInputs
    {
    public:
        TensorPtr draftLogits;
        TensorPtr draftProbs;
        TensorPtr targetProbs;
        TensorPtr numDraftTokens;
        TensorPtr draftTokenIds;
        TensorPtr useDraftLogits;
        TensorPtr useDraftLogitsHost;
        SizeType32 step;
        float constantThreshold;
        bool useRandomAcceptanceThreshold;
    };

    class ExplicitDraftTokensInputs
    {
    public:
        TensorConstPtr nextDraftTokens;
        TensorConstPtr nextFlatTokens;
        TensorConstPtr nextDraftIndices;
        TensorConstPtr nextDraftProbs;
        TensorConstPtr lastDraftTokens;
        TensorConstPtr lastDraftIndices;
        TensorConstPtr masks;
        TensorConstPtr packedPositionIds;
        TensorConstPtr bestPathLengths;
        TensorConstPtr bestPathIndices;
        TensorConstPtr nextGenerationLengths;
        TensorConstPtr lastPositionIdsBase;
        TensorConstPtr lastGenerationLengths;
        TensorConstPtr maxGenLengthDevice;
        TensorConstPtr seqSlots;
    };

    struct LookaheadInputs
    {
        TensorPtr tokensPerStep;
    };

    struct EagleInputs
    {
        EagleInputs(TensorConstPtr nextDraftTokens, TensorConstPtr nextDraftLens, TensorConstPtr nextDraftPaths,
            TensorConstPtr lastDraftTokens, TensorConstPtr lastDraftLens, TensorConstPtr lastDraftPaths,
            TensorConstPtr acceptedTokens, TensorConstPtr acceptedLens, TensorConstPtr acceptedPathIds,
            TensorConstPtr chunkedContextNextTokens, TensorConstPtr seqSlots)
            : nextDraftTokens(nextDraftTokens)
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

    std::optional<MedusaInputs> medusaInputs;

    std::optional<ExplicitDraftTokensInputs> explicitDraftTokensInputs;

    std::optional<LookaheadInputs> lookaheadInputs;

    std::optional<ExternalDraftTokensInputs> externalDraftTokensInputs;

    std::optional<EagleInputs> eagleInputs;
};

}
