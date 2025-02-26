
#pragma once

#include "suggestify/executor/types.h"
#include "baseLayer.h"
#include "decodingParams.h"
#include "common.h"

#include <curand_kernel.h>

namespace suggestify::layers
{

template <typename T>
class ExternalDraftTokensLayer : public BaseLayer
{
public:
    using Base = BaseLayer;

    ExternalDraftTokensLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager, bool isDeterministic = true, bool isAirTopP = true);

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
    TensorPtr mSkipTopKDecodeDevice;
    TensorPtr mSkipTopKDecodeHost;
    TensorPtr mSkipTopPDecodeDevice;
    TensorPtr mSkipTopPDecodeHost;

    TensorPtr mBatchIsAccepted;
    TensorPtr mRuntimeMultinomialDevice;

    TensorPtr mOutputIdsAfterSampling;
    TensorPtr mTargetOutputIds;
    TensorPtr mRuntimeTopKDevice;
    TensorPtr mRuntimeTopKHost;
    TensorPtr mRuntimeTopPDevice;
    TensorPtr mMaskBuffer;

    TensorPtr mTargetLogits;

    cudaDeviceProp mDeviceProp;
    runtime::SizeType32 mAirTopPBlockNum{0};
    bool mIsDeterministic{true};
    bool mIsAirTopP{false};

private:
    void allocateBuffer(runtime::SizeType32 batchSize);
    void acceptDraftTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void multinomialSampling(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void getAllTopKs(std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void getAllTopPs(std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void forwardAcceptedTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& baseInputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
};

}
