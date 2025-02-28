
#pragma once

#include "../batch_manager/BatchManager.h"
#include "../batch_manager/callbacks.h"
#include "../batch_manager/llmRequest.h"
#include "../batch_manager/trtGptModelOptionalParams.h"
#include "../runtime/modelConfig.h"
#include "../runtime/worldConfig.h"

#include <atomic>
#include <filesystem>
#include <optional>

namespace nvinfer1
{
class ILogger;
}

namespace sugesstify::batch_manager
{

class InferenceRequest;
class TrtGptModel;

class [[deprecated("Use the executor API instead.")]] GptManager
{
public:
    using SizeType32 = sugesstify::runtime::SizeType32;
    using TokenIdType = sugesstify::runtime::TokenIdType;
    using RequestList = std::list<std::shared_ptr<LlmRequest>>;
    using TensorPtr = runtime::ITensor::SharedPtr;

    GptManager(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
        GetInferenceRequestsCallback getInferenceRequestsCb, SendResponseCallback sendResponseCb,
        PollStopSignalCallback pollStopSignalCb = nullptr,
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = nullptr,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams(),
        std::optional<uint64_t> terminateReqId = std::nullopt, bool excludeInputInOutput = false);

    BatchManagerErrorCode_t fetchNewRequests();

    BatchManagerErrorCode_t returnCompletedRequests();

    BatchManagerErrorCode_t pollStopSignals();

    BatchManagerErrorCode_t returnBatchManagerStats();

    BatchManagerErrorCode_t waitUntilTerminate();

    BatchManagerErrorCode_t shutdown();

    SizeType32 getNumActiveRequests();

    virtual ~GptManager();

    void setLayerProfiler();

    [[nodiscard]] std::string getLayerProfileInfo() const;

protected:
    virtual BatchManagerErrorCode_t forwardSync();

    virtual BatchManagerErrorCode_t forwardAsync(
        RequestList& activeRequests, std::unordered_set<uint64_t>& activeRequestsIds);

private:
    [[nodiscard]] SizeType32 getMaxInputLen() const;
    [[nodiscard]] SizeType32 getMaxSequenceLen() const;
    [[nodiscard]] SizeType32 getMaxNumSequences() const;
    [[nodiscard]] SizeType32 getMaxDraftLen() const;

    void validateLlmRequest(
        LlmRequest& newReq, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const;
    static std::shared_ptr<LlmRequest> fillLlmRequest(std::shared_ptr<InferenceRequest> newReq);
    static std::shared_ptr<std::vector<TokenIdType>> getReqInputTokens(std::shared_ptr<InferenceRequest> newReq);
    static SizeType32 getMaxNewTokens(std::shared_ptr<InferenceRequest> newReq);

    GetInferenceRequestsCallback mGetInferenceRequestsCb;
    SendResponseCallback mSendResponseCb;
    PollStopSignalCallback mPollStopSignalCb;
    ReturnBatchManagerStatsCallback mReturnBatchManagerStatsCb;

    std::shared_ptr<TrtGptModel> mTrtGptModel;
    std::optional<uint64_t> mTerminateReqId;

    int64_t mIterationCounter;
    RequestList mActiveRequests;
    std::unordered_set<uint64_t> mActiveRequestsIds;
    bool mExcludeInputInOutput;

    std::atomic<bool> shutdown_requested_;
    void decoupled_execution_loop();
    std::shared_ptr<std::thread> worker_thread_;
    std::shared_ptr<nvinfer1::ILogger> mLogger{};

    inline static std::string const kPROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP";
    inline static std::string const kLEGACY_PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_GPTM_PROFILE_START_STOP";
};

}
