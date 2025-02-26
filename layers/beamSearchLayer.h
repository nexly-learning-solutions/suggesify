
#pragma once

#include "baseLayer.h"
#include "decodingParams.h"
#include "suggestify/runtime/common.h"

namespace suggestify::layers
{

template <typename T>
class BeamSearchLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    BeamSearchLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

    void setup(runtime::SizeType32 const batchSize, runtime::SizeType32 const beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

private:
    void allocateBuffer();
    void configureBeamSearchLayer();

private:
    using Base::mDecoderDomain;

    size_t mByteMaxSharedMemoryPerBlock{0};
    size_t mByteSharedMemoryStage1{0};
    size_t mByteSharedMemoryStage3{0};
    size_t mVPart{0};
    size_t mWorkspaceSize{0};
    bool mV2{false};

    TensorPtr mBeamSearchDiversityRateDevice;
    TensorPtr mLengthPenaltyDevice;
    TensorPtr mEarlyStoppingDevice;
    TensorPtr mBeamSearchDiversityRateHost;
    TensorPtr mLengthPenaltyHost;
    TensorPtr mEarlyStoppingHost;
};

}
