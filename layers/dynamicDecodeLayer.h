
#pragma once

#include "suggestify/executor/types.h"
#include "baseLayer.h"
#include "penaltyLayer.h"

namespace suggestify::layers
{

template <typename T>
class DynamicDecodeLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    DynamicDecodeLayer(executor::DecodingMode const& mode, DecoderDomain const& decodingDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth,
        runtime::ITensor::SharedConstPtr batchSlots, std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardSync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

    void disableLookahead(DecoderDomain const& decoderDomain, SizeType32 batchSize, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& baseSetupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

private:
    void allocateBuffer();

    void initialize();
    void initializeLayers();

    void prepareIdsPtrs(std::shared_ptr<BaseDecodingOutputs> const& outputs, BufferConstPtr batchSlots,
        runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen);
    void prepareOutputData(std::shared_ptr<BaseDecodingOutputs> const& outputs, BufferConstPtr batchSlots,
        runtime::SizeType32 batchSize, runtime::SizeType32 maxBatchSize, runtime::SizeType32 beamWidth,
        runtime::SizeType32 maxSeqLen, runtime::SizeType32 maxTokensPerStep, bool outputLogProbs, cudaStream_t stream);

private:
    using Base::mDecoderDomain;

    std::vector<std::unique_ptr<BaseLayer>> mLayers;

    executor::DecodingMode mDecodingMode;

    TensorPtr mZeroParentIdsDevice;
    TensorPtr mOutputIdsPtrHost;
    TensorPtr mParentIdsPtrHost;
    TensorPtr mOutputIdsPtrDevice;
    TensorPtr mParentIdsPtrDevice;

    bool mHasDiffRuntimeArgs{false};

    bool mOutputLogProbs{false};

    runtime::SizeType32 mCyclicStep{0};
    runtime::SizeType32 mRuntimeMaxSeqLen{0};
    runtime::SizeType32 mConfiguredBeamWidth{-1};
};

}
