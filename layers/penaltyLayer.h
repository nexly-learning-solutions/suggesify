
#pragma once

#include <curand_kernel.h>

#include "../types.h"
#include "baseLayer.h"
#include "decodingParams.h"

namespace suggestify::layers
{

template <typename T>
class PenaltyLayer : public BaseLayer
{
public:
    PenaltyLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void initialize();
    void allocateWorkspace();
    void allocateBuffer();

private:
    using BaseLayer::mDecoderDomain;

    executor::DecodingMode mDecodingMode;

    size_t mWorkspaceSize{};
    TensorPtr mTemperatureDevice;
    TensorPtr mRepetitionPenaltyDevice;
    TensorPtr mPresencePenaltyDevice;
    TensorPtr mFrequencyPenaltyDevice;
    TensorPtr mMinLengthDevice;

    TensorPtr mTemperature;
    TensorPtr mRepetitionPenalty;
    TensorPtr mPresencePenalty;
    TensorPtr mFrequencyPenalty;
    TensorPtr mMinLength;

    bool mUseTemperature{false};
    bool mUseRepetitionPenalty{false};
    bool mUsePresencePenalty{false};
    bool mUseFrequencyPenalty{false};
    bool mUseMinLength{false};

    runtime::SizeType32 mCyclicStep{0};
    runtime::SizeType32 mRuntimeMaxSeqLen{0};
    runtime::SizeType32 mConfiguredBeamWidth{-1};

    BufferPtr mPenaltyWorkspaceDevice;
    BufferPtr mPenaltyWorkspacePrevDevice;
    TensorPtr mLogitsPtrsHost;
};

}
