
#pragma once

#include "common.h"
#include "iTensor.h"

namespace suggestify::runtime
{

class GenerationConfig
{
public:
    GenerationConfig() = default;

    explicit GenerationConfig(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxInputLength,
        std::vector<SizeType32> maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength,
        SizeType32 maxSeqLength, SizeType32 inputLengthSum = SizeType32(0))
        : batchSize{batchSize}
        , beamWidth{beamWidth}
        , maxInputLength{maxInputLength}
        , maxAttentionWindowVec{maxAttentionWindowVec}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , maxSeqLength{maxSeqLength}
        , inputLengthSum{inputLengthSum}
    {
    }

    SizeType32 batchSize{};
    SizeType32 beamWidth{};
    SizeType32 maxInputLength{};
    std::vector<SizeType32> maxAttentionWindowVec{};
    SizeType32 maxAttentionWindow{};
    SizeType32 sinkTokenLength{};
    SizeType32 maxSeqLength{};
    SizeType32 inputLengthSum{};

    static GenerationConfig fromInput(ITensor const& inputIds, ITensor& inputLengths, bool inputPacked,
        SizeType32 beamWidth, std::vector<SizeType32> maxAttentionWindowVec, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLength, SizeType32 maxSequenceLength);
};

}
