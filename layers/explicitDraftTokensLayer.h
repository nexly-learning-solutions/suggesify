
#pragma once

#include "baseLayer.h"
#include "decodingParams.h"
#include "common.h"
#include "decodingLayerWorkspace.h"

#include <curand_kernel.h>

namespace suggestify::layers
{

template <typename T>
class ExplicitDraftTokensLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using PathsVec = std::vector<std::vector<std::vector<runtime::SizeType32>>>;

    ExplicitDraftTokensLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void allocateBuffer();

    void convertPackedMask(ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    void packAcceptedPaths(ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    template <typename Dtype>
    void fillContextBuffers(SizeType32 batchSize, BufferConstPtr batchSlots,
        ExplicitDraftTokensSetupParams const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

    template <typename Dtype>
    void splitInputDataToBatchSlots(ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

private:
    using Base::mDecoderDomain;

    SizeType32 mNumPaths;
    SizeType32 mMaxPathLength;

    size_t mScanWorkspaceSizeInBytes{0};
    size_t mReduceWorkspaceSizeInBytes{0};
    size_t mWorkspaceSize{0};

    TensorPtr mCurandStatesDevice;
    TensorPtr mGenerationLengthInclusiveSum;
    TensorPtr mMaxGenerationLength;
    TensorPtr mTemperatureDevice;
    TensorPtr mBestPathIndicesSlots;
    TensorPtr mLastDraftIndicesSlots;

    TensorPtr mTemperature;

    std::optional<nvinfer1::DataType> mDecoderDtype{std::nullopt};
};

}
