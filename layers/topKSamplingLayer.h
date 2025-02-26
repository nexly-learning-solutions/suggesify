
#pragma once

#include "baseLayer.h"
#include "common.h"

namespace suggestify::layers
{

template <typename T>
class TopKSamplingLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    TopKSamplingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

protected:
    bool mNormalizeLogProbs{true};
    size_t mWorkspaceSize{0};
    size_t mSetupWorkspaceSize{0};
    TensorPtr mRuntimeTopKDevice;
    TensorPtr mRuntimeTopPDevice;
    TensorPtr mSkipDecodeDevice;
    TensorPtr mRuntimeTopKHost;
    TensorPtr mSkipDecodeHost;

    using Base::mDecoderDomain;

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
};

}
