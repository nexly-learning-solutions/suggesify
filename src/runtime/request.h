
#pragma once

#include "../common/assert.h"
#include "../executor/executor.h"
#include "cudaEvent.h"
#include "cudaStream.h"
#include "iStatefulGptDecoder.h"
#include "iTensor.h"
#include "utils/sessionUtils.h"
#include <optional>

namespace suggestify::runtime::decoder_batch
{

class Request
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;
    using BufferPtr = IBuffer::SharedPtr;

    explicit Request(TensorConstPtr ids, SizeType32 inputLen, std::optional<SizeType32> maxNewTokens = std::nullopt,
        std::optional<SizeType32> endId = std::nullopt)
        : ids{std::move(ids)}
        , inputLen(inputLen)
        , maxNewTokens{maxNewTokens}
        , endId{endId}
        , generatedTokensPerEngineStep(1)
    {
    }

    TensorConstPtr ids;
    SizeType32 inputLen;

    std::optional<SizeType32> maxNewTokens;
    std::optional<SizeType32> endId;
    BufferPtr draftTokens;
    std::optional<TensorPtr>
        draftLogits;
    TensorPtr embeddingBias;
    TensorPtr badWordsList;
    TensorPtr stopWordsList;

    SizeType32 generatedTokensPerEngineStep;
    TensorPtr medusaPaths;
    TensorPtr medusaTreeIds;
    std::optional<executor::LookaheadDecodingConfig> lookaheadRuntimeConfig;
    std::optional<executor::EagleConfig> eagleConfig;
    nvinfer1::DataType dtype;
};

}
