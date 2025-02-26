
#pragma once

#include "bufferManager.h"
#include "common.h"
#include "eagleBuffers.h"
#include "explicitDraftTokensBuffers.h"
#include "iTensor.h"
#include "lookaheadBuffers.h"
#include <optional>
#include <utility>

namespace suggestify::batch_manager
{
class LookaheadDecodingBuffers;
}

namespace suggestify::runtime
{
class DecodingOutput
{
public:
    using TensorPtr = ITensor::SharedPtr;


    class BeamHypotheses
    {
    public:
        TensorPtr outputIdsCBA;
        TensorPtr logProbsCBA;
        TensorPtr sequenceLengthsCBA;
        TensorPtr cumLogProbsCBA;
        TensorPtr normedScoresCBA;
        TensorPtr numBeamsCBA;
        TensorPtr minNormedScoresCBA;
        TensorPtr batchDones;

        void empty(BufferManager& manager);

        void reshape(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSequenceLength);

        void release();

        void init(BufferManager& manager, TokenIdType endId);

        BeamHypotheses slice(SizeType32 batchIndex, SizeType32 size) const;
    };

    static float constexpr kNegativeInfinity = -1e20f;

    explicit DecodingOutput(TensorPtr ids, TensorPtr gatheredIds)
        : ids{std::move(ids)}
        , gatheredIds{std::move(gatheredIds)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
    }

    TensorPtr ids;

    TensorPtr gatheredIds;

    TensorPtr newTokensSteps;
    TensorPtr newTokens;
    std::vector<TensorPtr> newTokensVec;

    TensorPtr finishReasons;
    TensorPtr finishedSum;

    TensorPtr logProbs;
    TensorPtr cumLogProbs;
    TensorPtr parentIds;
    TensorPtr lengths;
    TensorPtr cacheIndirection;

    TensorPtr logProbsTiled;

    BeamHypotheses beamHypotheses;

    class SpeculativeDecodingOutputs
    {
    public:
        TensorPtr nextDraftTokens;
        TensorPtr nextDraftTokensLen;
        TensorPtr prevDraftTokensLen;
        TensorPtr acceptedTokensLen;
        TensorPtr acceptedLengthsCumSum;
        TensorPtr pathsOffsets;
    };

    std::optional<SpeculativeDecodingOutputs> speculativeDecodingOutputs;

    std::optional<ExplicitDraftTokensBuffers::Inputs> explicitDraftTokensBuffers;

    std::optional<LookaheadDecodingBuffers> lookaheadOutputs;

    std::optional<EagleBuffers::Inputs> eagleBuffers;
};

}
