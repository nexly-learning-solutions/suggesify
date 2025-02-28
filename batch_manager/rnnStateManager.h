
#pragma once

#include "../runtime/bufferManager.h"
#include "../runtime/iTensor.h"
#include "../runtime/modelConfig.h"
#include "../runtime/worldConfig.h"

namespace sugesstify::batch_manager::rnn_state_manager
{

class RnnStateManager
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = sugesstify::runtime::SizeType32;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    RnnStateManager(SizeType32 maxNumSequences, sugesstify::runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, sugesstify::runtime::BufferManager const& bufferManager);

    void getPtrBuffers(TensorMap& inputBuffers, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void fillSlotMapping(
        runtime::ITensor& dstPointers, SizeType32 dstSlotOffset, SizeType32 seqSlotIdx, SizeType32 beamWidth) const;

private:
    TensorPtr pagedRnnStates;
    TensorPtr pagedConvStates;

    TensorPtr rnnStatePtrs;
    TensorPtr convStatePtrs;

    std::vector<TensorPtr> rnnStatePtr;
    std::vector<TensorPtr> convStatePtr;

    SizeType32 mMaxNumSequences = 0;
    SizeType32 mMaxBeamWidth = 0;
    SizeType32 mBeamSlotsPerSequence = 0;
};

}
