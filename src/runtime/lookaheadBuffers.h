
#pragma once

#include "../executor/executor.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "tllmRuntime.h"
#include "worldConfig.h"

namespace suggestify::runtime
{

class LookaheadDecodingBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using ITensor = suggestify::runtime::ITensor;
    LookaheadDecodingBuffers(
        SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, runtime::BufferManager const& bufferManager);
    TensorPtr generationLengths;
    TensorPtr positionOffsets;
    TensorPtr packedMasks;
    TensorPtr positionIds;
};

class LookaheadRuntimeBuffers
{
public:
    using SizeType32 = suggestify::runtime::SizeType32;
    using ITensor = suggestify::runtime::ITensor;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    LookaheadRuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime);

    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ITensor const& requestTypes,
        ITensor const& seqSlots, LookaheadDecodingBuffers const& decoderLookaheadBuffers,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 tokensPerStep);

    void insertInputTensors(
        TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const;

    void enableLookaheadDecoding(SizeType32 maxBatchSize, SizeType32 tokensPerStep);

    void disableLookaheadDecoding();

public:
    TensorPtr cumSumLength;
    TensorPtr packedMasksDevice;
    TensorPtr generationLengthsDevice;
    TensorPtr positionOffsetsDevice;
    TensorPtr positionIdsDevice;

    TensorPtr packedMaskHost;
    TensorPtr generationLengthsHost;
    TensorPtr positionOffsetsHost;
    TensorPtr positionIdsHost;

    TensorPtr packedMaskHostCopy;
    TensorPtr generationLengthsHostCopy;
    TensorPtr positionOffsetsHostCopy;
    TensorPtr positionIdsHostCopy;
    TensorPtr useSpecDecoding;

    TensorPtr batchSlotsHostCopy;
};

}
