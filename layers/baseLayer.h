
#pragma once

#include <utility>

#include "decodingParams.h"
#include "suggestify/runtime/bufferManager.h"
#include "suggestify/runtime/common.h"
#include "suggestify/runtime/decodingLayerWorkspace.h"

namespace suggestify::layers
{

class BaseLayer
{
public:
    using SizeType32 = runtime::SizeType32;
    using TokenIdType = runtime::TokenIdType;
    using BufferConstPtr = runtime::IBuffer::SharedConstPtr;
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;
    using TensorPtr = runtime::ITensor::SharedPtr;

    BaseLayer(DecoderDomain decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager)
        : mBufferManager(std::move(bufferManager))
        , mDecoderDomain(std::move(decoderDomain))
    {
    }

    virtual ~BaseLayer() = default;

    [[nodiscard]] cudaStream_t getStream() const noexcept
    {
        return mBufferManager->getStream().get();
    }

    [[nodiscard]] virtual size_t getWorkspaceSize() const noexcept
    {
        return 0;
    };

    virtual void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
        = 0;

    virtual void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
        = 0;

    virtual void forwardSync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
    {
    }

protected:
    std::shared_ptr<runtime::BufferManager> mBufferManager;

    DecoderDomain mDecoderDomain;
};

}
