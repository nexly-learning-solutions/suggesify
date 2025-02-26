
#pragma once

#include "../types.h"
#include "baseLayer.h"
#include "decodingParams.h"

#include <curand_kernel.h>

namespace suggestify::layers
{

template <typename T>
class BanWordsLayer : public BaseLayer
{

public:
    BanWordsLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
        std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& baseSetupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

private:
    void allocateBuffer();
    void banBadWords(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots,
        DecoderDomain const& decoderDomain, runtime::SizeType32 maxSeqLen);
    void banRepeatNGrams(TensorPtr const& logits, std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<DecodingInputs> const& inputs, BufferConstPtr const& batchSlots,
        BufferPtr noRepeatNgramSizeDevice, DecoderDomain const& decoderDomain, runtime::SizeType32 maxSeqLen,
        bool useNoRepeatNgramSize);

private:
    executor::DecodingMode mDecodingMode;

    TensorPtr mNoRepeatNgramSizeDevice;
    TensorPtr mNoRepeatNgramSize;
    bool mUseNoRepeatNgramSize{false};
};

}
