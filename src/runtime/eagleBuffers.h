
#pragma once

#include "../executor/executor.h"
#include "eagleModule.h"
#include "iBuffer.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "tllmRuntime.h"
#include "worldConfig.h"

#include <cstddef>

namespace suggestify::batch_manager
{
class LlmRequest;
}

namespace suggestify::runtime
{

class EagleBuffers
{
public:
    using LlmRequestPtr = std::shared_ptr<suggestify::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using SizeType32 = runtime::SizeType32;
    using ITensor = runtime::ITensor;
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    class Inputs
    {
    public:
        TensorPtr temperatures;
        TensorPtr posteriorAlpha;
        TensorPtr posteriorThreshold;
        TensorPtr randomDataSample;
        TensorPtr randomDataValidation;
        TensorPtr draftTokens;
        TensorPtr draftLens;
        TensorPtr draftPaths;
        TensorPtr specDecodingGenerationLengths;
        TensorPtr specDecodingGenerationLengthsHost;
        TensorPtr specDecodingPackedMasks;
        TensorPtr specDecodingPositionOffsets;
        TensorPtr eagleNetCtxRequestTypesHost;
        TensorPtr eagleNetCtxContextLengthsHost;
        TensorPtr eagleNetCtxPastKeyValueLengthsHost;
        TensorPtr eagleNetGenRequestTypesHost;
        TensorPtr eagleNetGenContextLengthsHost;
        TensorPtr eagleNetGenPastKeyValueLengthsHost;
        TensorPtr inputGenTokensHost;
        TensorPtr chunkedContextNextTokens;
        TensorPtr useSpecDecoding;

        TensorPtr useDynamicTreeHost;

        void create(SizeType32 maxNumSequences, runtime::TllmRuntime const& runtime,
            runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);
    };

    Inputs engineInputs;

    class EngineOutputs
    {
    public:
        TensorPtr nextDraftTokens;
        TensorPtr nextDraftLens;
        TensorPtr nextDraftPaths;

        TensorPtr acceptedTokens;
        TensorPtr acceptedLens;
        TensorPtr acceptedPaths;
        TensorPtr chunkedContextNextTokens;

    } engineOutputs;

public:
    EagleBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime);

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig);

    void setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
        runtime::ITensor const& requestTypes, ITensor const& seqSlots, EagleBuffers::Inputs const& decoderBuffers,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void insertInputTensors(
        TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const;

private:
    template <typename T>
    void setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
        SizeType32 vocabSizePadded, ITensor const& seqSlots, EagleBuffers::Inputs const& draftBuffers,
        runtime::EagleModule const& eagleModule, runtime::BufferManager const& manager) const;

private:
    std::size_t scanTempStorageBytes{0};
    std::size_t reduceTempStorageBytes{0};
    float mDefaultPosteriorThreshold{0.09f};
    bool mDoGreedySampling{true};
    BufferPtr scanReduceTempStorage;
    TensorPtr cumSumGenerationLengths;
    TensorPtr maxGenerationLength;
    TensorPtr chunkedContextNextTokensHost;
    TensorPtr greedySamplingHost;
    TensorPtr posteriorAlphaHost;
    TensorPtr posteriorThresholdHost;
};

}
