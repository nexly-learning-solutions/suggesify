
#pragma once

#include "common.h"
#include "iTensor.h"
#include "promptTuningParams.h"

#include <optional>
#include <utility>

namespace suggestify::runtime
{

template <typename TTensor, typename PromptTuningParams>
class GenericGenerationInput
{
public:
    using TensorPtr = TTensor;

    explicit GenericGenerationInput(
        SizeType32 const endId, SizeType32 const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : endId{endId}
        , padId{padId}
        , ids{std::move(ids)}
        , lengths{std::move(lengths)}
        , packed{packed}
        , maxNewTokens(std::nullopt)
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    SizeType32 endId;
    SizeType32 padId;
    TensorPtr ids;
    TensorPtr lengths;
    bool packed;

    TensorPtr embeddingBias;
    TensorPtr badWordsList;
    TensorPtr stopWordsList;
    std::optional<SizeType32> maxNewTokens;

    PromptTuningParams promptTuningParams;
};

class GenerationInput : public GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>
{
public:
    using Base = GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationInput(
        SizeType32 const endId, SizeType32 const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : GenericGenerationInput(endId, padId, std::move(ids), std::move(lengths), packed)
    {
    }
};

}
