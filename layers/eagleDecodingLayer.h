
#pragma once

#include "baseLayer.h"
#include "decodingParams.h"
#include "common.h"
#include "decodingLayerWorkspace.h"

#include <curand_kernel.h>

namespace suggestify::layers
{

template <typename T>
class EagleDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using PathsVec = std::vector<std::vector<std::vector<runtime::SizeType32>>>;

    EagleDecodingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void allocateBuffer();

    void fillContextBuffers(runtime::SizeType32 batchSize, BufferConstPtr batchSlots,
        EagleSetupParams const& setupParams, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void augmentBatchSlots(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void convertToPackedMask(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void packAcceptedPaths(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void unpackData(EagleOutputs const& outputs, EagleInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

private:
    using Base::mDecoderDomain;

    size_t mWorkspaceSize{0};

    TensorPtr mTemperature;

    TensorPtr mCurandStatesDevice;
    TensorPtr mTemperatureDevice;

    TensorPtr mEagleNetCtxRequestTypes;
    TensorPtr mEagleNetCtxContextLengths;
    TensorPtr mEagleNetCtxPastKeyValueLengths;
    TensorPtr mEagleNetGenRequestTypes;
    TensorPtr mEagleNetGenContextLengths;
    TensorPtr mEagleNetGenPastKeyValueLengths;
};

}
