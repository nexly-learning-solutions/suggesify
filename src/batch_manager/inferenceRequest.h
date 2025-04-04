
#pragma once

#include "../llmRequest.h"
#include "../namedTensor.h"
#include "../executor/executor.h"
#include "../runtime/iTensor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sugesstify::batch_manager
{

namespace inference_request
{
auto constexpr kInputIdsTensorName = "input_ids";
auto constexpr kPositionIdsTensorName = "position_ids";
auto constexpr kDraftInputIdsTensorName = "draft_input_ids";
auto constexpr kDraftLogitsTensorName = "draft_logits";
auto constexpr kMaxNewTokensTensorName = "request_output_len";
auto constexpr kBeamWidthTensorName = "beam_width";
auto constexpr kNumReturnSequencesTensorName = "num_return_sequences";
auto constexpr kEndIdTensorName = "end_id";
auto constexpr kPadIdTensorName = "pad_id";
auto constexpr kBadWordsListTensorName = "bad_words_list";
auto constexpr kStopWordsListTensorName = "stop_words_list";
auto constexpr kEmbeddingBiasTensorName = "embedding_bias";
auto constexpr kTemperatureTensorName = "temperature";
auto constexpr kRuntimeTopKTensorName = "runtime_top_k";
auto constexpr kRuntimeTopPTensorName = "runtime_top_p";
auto constexpr kLengthPenaltyTensorName = "len_penalty";
auto constexpr kEarlyStoppingTensorName = "early_stopping";
auto constexpr kRepetitionPenaltyTensorName = "repetition_penalty";
auto constexpr kMinLengthTensorName = "min_length";
auto constexpr kPresencePenaltyTensorName = "presence_penalty";
auto constexpr kFrequencyPenaltyTensorName = "frequency_penalty";
auto constexpr kRandomSeedTensorName = "random_seed";
auto constexpr kReturnLogProbsTensorName = "return_log_probs";
auto constexpr kReturnContextLogitsTensorName = "return_context_logits";
auto constexpr kReturnGenerationLogitsTensorName = "return_generation_logits";
auto constexpr kPromptEmbeddingTableName = "prompt_embedding_table";
auto constexpr kPromptVocabSizeName = "prompt_vocab_size";
auto constexpr kLoraTaskId = "lora_task_id";
auto constexpr kNoRepeatNgramSizeTensorName = "noRepeatNgramSize";
auto constexpr kSkipCrossAttnBlocksTensorName = "skipCrossAttnBlocks";
auto constexpr kLoraWeights = "lora_weights";
auto constexpr kLoraConfig = "lora_config";

auto constexpr kInputLengthsTensorName = "input_lengths";

auto constexpr kOutputIdsTensorName = "output_ids";
auto constexpr kSequenceLengthTensorName = "sequence_length";
auto constexpr kLogProbsTensorName = "output_log_probs";
auto constexpr kCumLogProbsTensorName = "cum_log_probs";
auto constexpr kContextLogitsName = "context_logits";
auto constexpr kGenerationLogitsName = "generation_logits";

}

template <typename TTensor, typename TNamedTensor, typename TStream = runtime::BufferManager::CudaStreamPtr>
class GenericInferenceRequest
{
public:
    using TensorPtr = TTensor;
    using NamedTensorType = TNamedTensor;
    using TensorMap = std::unordered_map<std::string, TTensor>;
    using LogitsPostProcessor = typename GenericLlmRequest<TensorPtr, TStream>::LogitsPostProcessor;

    explicit GenericInferenceRequest(
        uint64_t requestId, std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt)
        : mRequestId{requestId}
        , mIsStreaming{false}
        , mLogitsPostProcessor(logitsPostProcessor)
    {
    }

    GenericInferenceRequest(uint64_t requestId, TensorMap&& tensorMap,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt)
        : mRequestId{requestId}
        , mIsStreaming{false}
        , mInputTensors{std::move(tensorMap)}
        , mLogitsPostProcessor(logitsPostProcessor)
    {
        for (auto const& [name, tensor] : mInputTensors)
        {
            validateTensorName(name);
        }
    }

    GenericInferenceRequest(uint64_t requestId, TensorMap const& tensorMap,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt)
        : GenericInferenceRequest(requestId, TensorMap{tensorMap}, logitsPostProcessor)
    {
    }

    void setIsStreaming(bool isStreaming)
    {
        mIsStreaming = isStreaming;
    }

    [[nodiscard]] bool isStreaming() const
    {
        return mIsStreaming;
    }

    [[nodiscard]] uint64_t getRequestId() const
    {
        return mRequestId;
    }

    TensorMap const& getInputTensors() const
    {
        return mInputTensors;
    }

    void setLogitsPostProcessor(std::optional<LogitsPostProcessor> cb)
    {
        mLogitsPostProcessor = cb;
    }

    [[nodiscard]] std::optional<executor::LookaheadDecodingConfig> getLookaheadConfig() const
    {
        return mLookaheadConfig;
    }

    void setLookaheadConfig(executor::LookaheadDecodingConfig config)
    {
        mLookaheadConfig = config;
    }

    void clearLookaheadConfig()
    {
        mLookaheadConfig = std::nullopt;
    }

    std::optional<LogitsPostProcessor> getLogitsPostProcessor()
    {
        return mLogitsPostProcessor;
    }

    static std::array constexpr kTensorNames = {
        inference_request::kInputIdsTensorName,
        inference_request::kPositionIdsTensorName,
        inference_request::kDraftInputIdsTensorName,
        inference_request::kDraftLogitsTensorName,
        inference_request::kMaxNewTokensTensorName,
        inference_request::kBeamWidthTensorName,
        inference_request::kNumReturnSequencesTensorName,
        inference_request::kEndIdTensorName,
        inference_request::kPadIdTensorName,
        inference_request::kBadWordsListTensorName,
        inference_request::kStopWordsListTensorName,
        inference_request::kEmbeddingBiasTensorName,
        inference_request::kTemperatureTensorName,
        inference_request::kRuntimeTopKTensorName,
        inference_request::kRuntimeTopPTensorName,
        inference_request::kLengthPenaltyTensorName,
        inference_request::kEarlyStoppingTensorName,
        inference_request::kRepetitionPenaltyTensorName,
        inference_request::kMinLengthTensorName,
        inference_request::kPresencePenaltyTensorName,
        inference_request::kFrequencyPenaltyTensorName,
        inference_request::kRandomSeedTensorName,
        inference_request::kReturnLogProbsTensorName,
        inference_request::kReturnContextLogitsTensorName,
        inference_request::kReturnGenerationLogitsTensorName,
        inference_request::kPromptEmbeddingTableName,
        inference_request::kPromptVocabSizeName,
        inference_request::kNoRepeatNgramSizeTensorName,
        inference_request::kInputLengthsTensorName,
        inference_request::kLoraTaskId,
        inference_request::kLoraWeights,
        inference_request::kLoraConfig,
    };

#define TENSOR_GETTER_SETTER(funcName, tensorName)                                                                     \
                                                                                                                       \
    [[nodiscard]] bool has##funcName() const                                                                           \
    {                                                                                                                  \
        return mInputTensors.find(tensorName) != mInputTensors.end();                                                  \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] TensorPtr const& get##funcName() const                                                               \
    {                                                                                                                  \
        auto it = mInputTensors.find(tensorName);                                                                      \
        CHECK_WITH_INFO(it != mInputTensors.end(), "Undefined tensor: %s", tensorName);                           \
        return it->second;                                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] TensorPtr get##funcName##Unchecked() const                                                           \
    {                                                                                                                  \
        auto it = mInputTensors.find(tensorName);                                                                      \
        return it != mInputTensors.end() ? it->second : TensorPtr{};                                                   \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] NamedTensorType get##funcName##Named() const                                                         \
    {                                                                                                                  \
        auto it = mInputTensors.find(tensorName);                                                                      \
        return it != mInputTensors.end() ? NamedTensorType{it->second, tensorName} : NamedTensor{tensorName};          \
    }                                                                                                                  \
                                                                                                                       \
    void set##funcName(TensorPtr const& tensor)                                                                        \
    {                                                                                                                  \
        if constexpr (std::is_same_v<TensorPtr, sugesstify::runtime::ITensor::SharedPtr>)                            \
        {                                                                                                              \
            CHECK_WITH_INFO(tensor, "Cannot set nullptr when calling %s", __FUNCTION__);                          \
        }                                                                                                              \
        mInputTensors[tensorName] = tensor;                                                                            \
    }

    TENSOR_GETTER_SETTER(InputIds, inference_request::kInputIdsTensorName)
    TENSOR_GETTER_SETTER(PositionIds, inference_request::kPositionIdsTensorName)
    TENSOR_GETTER_SETTER(DraftInputIds, inference_request::kDraftInputIdsTensorName)
    TENSOR_GETTER_SETTER(DraftLogits, inference_request::kDraftLogitsTensorName)
    TENSOR_GETTER_SETTER(MaxNewTokens, inference_request::kMaxNewTokensTensorName)
    TENSOR_GETTER_SETTER(BeamWidth, inference_request::kBeamWidthTensorName)
    TENSOR_GETTER_SETTER(NumReturnSequences, inference_request::kNumReturnSequencesTensorName)
    TENSOR_GETTER_SETTER(EndId, inference_request::kEndIdTensorName)
    TENSOR_GETTER_SETTER(PadId, inference_request::kPadIdTensorName)
    TENSOR_GETTER_SETTER(BadWordsList, inference_request::kBadWordsListTensorName)
    TENSOR_GETTER_SETTER(StopWordsList, inference_request::kStopWordsListTensorName)
    TENSOR_GETTER_SETTER(EmbeddingBias, inference_request::kEmbeddingBiasTensorName)
    TENSOR_GETTER_SETTER(Temperature, inference_request::kTemperatureTensorName)
    TENSOR_GETTER_SETTER(RuntimeTopK, inference_request::kRuntimeTopKTensorName)
    TENSOR_GETTER_SETTER(RuntimeTopP, inference_request::kRuntimeTopPTensorName)
    TENSOR_GETTER_SETTER(LengthPenalty, inference_request::kLengthPenaltyTensorName)
    TENSOR_GETTER_SETTER(EarlyStopping, inference_request::kEarlyStoppingTensorName)
    TENSOR_GETTER_SETTER(RepetitionPenalty, inference_request::kRepetitionPenaltyTensorName)
    TENSOR_GETTER_SETTER(MinLength, inference_request::kMinLengthTensorName)
    TENSOR_GETTER_SETTER(PresencePenalty, inference_request::kPresencePenaltyTensorName)
    TENSOR_GETTER_SETTER(FrequencyPenalty, inference_request::kFrequencyPenaltyTensorName)
    TENSOR_GETTER_SETTER(RandomSeed, inference_request::kRandomSeedTensorName)
    TENSOR_GETTER_SETTER(ReturnLogProbs, inference_request::kReturnLogProbsTensorName)
    TENSOR_GETTER_SETTER(ReturnContextLogits, inference_request::kReturnContextLogitsTensorName)
    TENSOR_GETTER_SETTER(ReturnGenerationLogits, inference_request::kReturnGenerationLogitsTensorName)
    TENSOR_GETTER_SETTER(PromptEmbeddingTable, inference_request::kPromptEmbeddingTableName)
    TENSOR_GETTER_SETTER(PromptVocabSize, inference_request::kPromptVocabSizeName)
    TENSOR_GETTER_SETTER(LoraTaskId, inference_request::kLoraTaskId)
    TENSOR_GETTER_SETTER(LoraWeights, inference_request::kLoraWeights)
    TENSOR_GETTER_SETTER(LoraConfig, inference_request::kLoraConfig)
    TENSOR_GETTER_SETTER(NoRepeatNgramSize, inference_request::kNoRepeatNgramSizeTensorName)

#undef TENSOR_GETTER_SETTER

protected:
    static void validateTensorName(std::string const& tensorName)
    {
        CHECK_WITH_INFO(std::find(kTensorNames.begin(), kTensorNames.end(), tensorName) != kTensorNames.end(),
            "Invalid tensor name: %s", tensorName.c_str());
    }

    uint64_t mRequestId;
    bool mIsStreaming;
    TensorMap mInputTensors;
    std::optional<LogitsPostProcessor> mLogitsPostProcessor;
    std::optional<executor::LookaheadDecodingConfig> mLookaheadConfig;
};

class InferenceRequest : public GenericInferenceRequest<sugesstify::runtime::ITensor::SharedPtr, NamedTensor>
{
public:
    using Base = GenericInferenceRequest<sugesstify::runtime::ITensor::SharedPtr, NamedTensor>;
    using TensorPtr = Base::TensorPtr;
    using TensorMap = Base::TensorMap;

    explicit InferenceRequest(uint64_t requestId)
        : Base(requestId)
    {
    }

    InferenceRequest(TensorMap const& inputTensors, uint64_t requestId)
        : Base(requestId, inputTensors)
    {
    }

    InferenceRequest(TensorMap&& inputTensors, uint64_t requestId)
        : Base(requestId, std::move(inputTensors))
    {
    }

    [[deprecated("Use direct tensor access instead")]] [[nodiscard]] TensorPtr const& getInputTensor(
        std::string const& inputTensorName) const
    {
        auto it = Base::mInputTensors.find(inputTensorName);
        CHECK_WITH_INFO(it != Base::mInputTensors.end(), "Invalid input tensor name: %s", inputTensorName.c_str());
        return it->second;
    }

    [[deprecated("Use direct tensor access instead")]] void emplaceInputTensor(
        std::string const& inputTensorName, TensorPtr inputTensor)
    {
        validateTensorName(inputTensorName);
        Base::mInputTensors[inputTensorName] = std::move(inputTensor);
    }

    [[nodiscard]] std::vector<int64_t> serialize() const;

    static std::shared_ptr<InferenceRequest> deserialize(std::vector<int64_t> const& packed);

    static std::shared_ptr<InferenceRequest> deserialize(int64_t const* packed_ptr);
};

}
