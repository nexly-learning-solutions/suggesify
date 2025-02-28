
#pragma once

#include "baseLayer.h"
#include "common.h"

namespace suggestify::layers
{

template <typename T>
class TopPSamplingLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    TopPSamplingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager,
        bool isDeterministic = true, bool isAirTopP = true);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;
    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

protected:
    TensorPtr mRuntimeTopKDevice;
    TensorPtr mRuntimeTopPDevice;
    TensorPtr mInitialTopPDevice;
    TensorPtr mTopPDecayDevice;
    TensorPtr mTopPMinDevice;
    TensorPtr mTopPResetIdsDevice;

    TensorPtr mSkipDecodeDevice;
    TensorPtr mSkipDecodeHost;
    size_t mWorkspaceSize{0};
    size_t mSetupWorkspaceSize{0};

    cudaDeviceProp mDeviceProp;
    runtime::SizeType32 mAirTopPBlockNum{0};
    bool mIsDeterministic{true};
    bool mIsAirTopP{false};

    using Base::mDecoderDomain;

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
};

}
