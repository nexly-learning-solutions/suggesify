
#pragma once

#include "../executor/executor.h"
#include "explicitDraftTokensModule.h"
#include "iBuffer.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "tllmRuntime.h"
#include "worldConfig.h"

#include <cstddef>

namespace suggestify::runtime
{

class ExplicitDraftTokensBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using ITensor = runtime::ITensor;
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    class Inputs
    {
    public:
        TensorPtr temperatures;
        TensorPtr positionIdsBase;
        TensorPtr generationLengths;
        TensorPtr randomDataSample;
        TensorPtr randomDataValidation;
        TensorPtr draftTokens;
        TensorPtr draftIndices;
        TensorPtr draftProbs;
        TensorPtr packedMasks;
        TensorPtr positionIds;
        TensorPtr maxGenLengthHost;
        TensorPtr generationLengthsHost;
        TensorPtr useSpecDecoding;

        void create(SizeType32 maxNumSequences, runtime::TllmRuntime const& runtime,
            runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);
    };

    class EngineInputs : public Inputs
    {
    public:
        TensorPtr requestTypesDevice;
        TensorPtr positionOffsets;
    } engineInputs;

    class EngineOutputs
    {
    public:
        TensorPtr nextGenerationLengths;
        TensorPtr nextPositionOffsets;
        TensorPtr masks;

        TensorPtr nextDraftTokens;
        TensorPtr nextDraftIndices;
        TensorPtr nextDraftProbs;

        TensorPtr nextFlatTokens;
        TensorPtr bestPathLengths;
        TensorPtr bestPathIndices;
        TensorPtr maxGenToken;
        TensorPtr totalGenToken;
        TensorPtr packedPositionIds;
    } engineOutputs;

public:
    ExplicitDraftTokensBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime);

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig);

    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ITensor const& requestTypes,
        ITensor const& seqSlots, ExplicitDraftTokensBuffers::Inputs const& decoderBuffers,
        ITensor const& contextPositionIds, runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void insertInputTensors(
        TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const;

private:
    template <typename T>
    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 vocabSizePadded,
        ITensor const& seqSlots, ExplicitDraftTokensBuffers::Inputs const& draftBuffers,
        ITensor const& contextPositionIds, runtime::ExplicitDraftTokensModule const& explicitDraftTokensModule,
        runtime::CudaStream const& stream) const;

public:
    std::size_t scanTempStorageBytes{0};
    BufferPtr scanTempStorage;
    TensorPtr cumSumGenerationLengths;
};

}
