
#pragma once

#include "bufferManager.h"
#include "common.h"
#include "generationConfig.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "tllmRuntime.h"
#include "worldConfig.h"

namespace suggestify::runtime
{

class RuntimeBuffers;

class RnnStateBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TensorMap = StringPtrMap<ITensor>;

    TensorPtr rnnStates;
    TensorPtr convStates;
    TensorPtr convStatesAlt;

    std::vector<TensorPtr> rnnState;
    std::vector<TensorPtr> convState;
    std::vector<TensorPtr> convStateAlt;

    TensorPtr slotMappingHost;
    TensorPtr slotMappingDevice;
    TensorPtr rnnStatePtrs;
    TensorPtr convStatePtrs;

    std::vector<TensorPtr> rnnStatePtr;
    std::vector<TensorPtr> convStatePtr;

    RnnStateBuffers();

    RnnStateBuffers(
        TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void reshape(SizeType32 batchSize);
    void reshape(
        GenerationConfig const& generationConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    RnnStateBuffers sliceTo(SizeType32 offset, SizeType32 size);

    void prepareContextStep(RuntimeBuffers* runtimeBuffers, BufferManager& manager);

    void postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
        BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
        SizeType32 const step, TensorPtr const& inputIds, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

protected:
    void tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void fillStatePtrs();

private:
    SizeType32 mConvKernel = 0;
    SizeType32 mStateSize = 0;
    SizeType32 mRnnHiddenSize = 0;
    SizeType32 mRnnHeadSize = 0;
    SizeType32 mRnnConvDimSize = 0;

    int mLocalNbLayers = 0;
    int mMaxBeamWidth = 0;

    bool mUseMambaConv1dPlugin = true;

    bool mIsRecurrentGemma = false;
};

}
