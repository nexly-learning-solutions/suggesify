
#pragma once

#include "suggestify/executor/types.h"
#include "baseLayer.h"
#include "decodingParams.h"

#include <curand_kernel.h>

namespace suggestify::layers
{

template <typename T>
class StopCriteriaLayer : public BaseLayer
{
public:
    StopCriteriaLayer(executor::DecodingMode const& mode, DecoderDomain const&,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    static void checkMaxLengthStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
        runtime::BufferManager const& bufferManager, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    static void checkStopWordsStopCriteria(std::shared_ptr<BaseDecodingOutputs>& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
        runtime::SizeType32 maxSeqLen, runtime::BufferManager const& bufferManager,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    static void checkEosToken(std::shared_ptr<BaseDecodingOutputs>& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, DecoderDomain const& decoderDomain,
        runtime::BufferManager const& bufferManager, std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

private:
    using BaseLayer::mDecoderDomain;

    executor::DecodingMode mDecodingMode;
    size_t mWorkspaceSize{0};
};

}
