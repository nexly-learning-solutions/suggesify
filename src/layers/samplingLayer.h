
#pragma once

#include "../types.h"
#include "baseLayer.h"
#include "decodingParams.h"
#include "common.h"

#include <curand_kernel.h>

namespace suggestify::layers
{

template <typename T>
class SamplingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;

    SamplingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    using Base::mDecoderDomain;

    executor::DecodingMode mDecodingMode;

    size_t mWorkspaceSize{0};
    size_t mSetupWorkspaceSize{0};

    TensorPtr mCurandStatesDevice;
    TensorPtr mSkipDecodeDevice;

    TensorPtr mSkipDecodeHost;
    bool mSkipAny{false};

    bool mOutputLogProbs{false};
    bool mCumLogProbs{false};

    std::vector<std::unique_ptr<BaseLayer>> mSamplingLayers;

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
};

}
