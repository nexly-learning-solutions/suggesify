
#pragma once

#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"

#include <utility>

namespace suggestify::runtime
{

template <typename TTensor>
class GenericPromptTuningParams
{
public:
    using TensorPtr = TTensor;
    using SizeType32 = suggestify::runtime::SizeType32;

    explicit GenericPromptTuningParams(
        TensorPtr embeddingTable = TensorPtr(), TensorPtr tasks = TensorPtr(), TensorPtr vocabSize = TensorPtr())
        : embeddingTable{std::move(embeddingTable)}
        , tasks{std::move(tasks)}
        , vocabSize{std::move(vocabSize)} {};

    TensorPtr embeddingTable;
    TensorPtr tasks;
    TensorPtr vocabSize;

    std::vector<bool>
        promptTuningEnabled;
};

class PromptTuningParams : public GenericPromptTuningParams<ITensor::SharedPtr>
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using SizeType32 = GenericPromptTuningParams::SizeType32;

    explicit PromptTuningParams(
        TensorPtr embeddingTable = nullptr, TensorPtr tasks = nullptr, TensorPtr vocabSize = nullptr)
        : GenericPromptTuningParams(std::move(embeddingTable), std::move(tasks), std::move(vocabSize))
    {
    }

    void fillTasksTensor(TensorPtr tasksHost, const SizeType32 batchSize, const SizeType32 numContextRequests,
        std::vector<SizeType32> const& reqBeamWidths, std::vector<SizeType32> const& reqPromptLengths,
        BufferManager const& manager, bool packedInput);
};

}
