
#pragma once

#include "../executor/types.h"
#include "bufferManager.h"
#include "decodingInput.h"
#include "decodingOutput.h"
#include "request.h"
#include "samplingConfig.h"

#include <NvInferRuntime.h>
#include <curand_kernel.h>

#include <memory>

namespace suggestify
{

namespace layers
{
template <typename T>
class DynamicDecodeLayer;
}

namespace runtime
{

class SpeculativeDecodingModule;

class DecodingLayerWorkspace;

class IGptDecoder
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;

    virtual ~IGptDecoder() = default;

    virtual void setup(SamplingConfig const& samplingConfig, size_t batchSize, TensorConstPtr const& batchSlots,
        std::optional<DecodingOutput> const& output = std::nullopt,
        std::optional<std::vector<decoder_batch::Request> const> const& requests = std::nullopt)
        = 0;

    virtual void forwardAsync(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void forwardSync(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual SamplingConfig const& getSamplingConfig() = 0;

    virtual void disableLookahead(
        std::optional<SamplingConfig> const& samplingConfig, SizeType32 batchSize, TensorConstPtr batchSlots)
        = 0;

    static std::unique_ptr<IGptDecoder> create(executor::DecodingMode const& mode, nvinfer1::DataType dtype,
        size_t maxBatchSize, size_t maxBeamWidth, size_t vocabSize, size_t vocabSizePadded, size_t maxSequenceLength,
        BufferManager::CudaStreamPtr const& stream,
        std::shared_ptr<SpeculativeDecodingModule const> const& speculativeDecodingModule = nullptr);
};

template <typename T>
class GptDecoder : public virtual IGptDecoder
{

public:
    using CudaStreamPtr = BufferManager::CudaStreamPtr;
    using TensorPtr = std::shared_ptr<ITensor>;

    GptDecoder(executor::DecodingMode const& mode, size_t maxBatchSize, size_t maxBeamWidth, size_t vocabSize,
        size_t vocabSizePadded, size_t maxSequenceLength, CudaStreamPtr const& stream,
        std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModule = nullptr);

    void setup(SamplingConfig const& samplingConfig, size_t batchSize, TensorConstPtr const& batchSlots,
        std::optional<DecodingOutput> const& output = std::nullopt,
        std::optional<std::vector<decoder_batch::Request> const> const& requests = std::nullopt) override;

    void forwardAsync(DecodingOutput& output, DecodingInput const& input) override;

    void forwardSync(DecodingOutput& output, DecodingInput const& input) override;

    SamplingConfig const& getSamplingConfig() override
    {
        return mSamplingConfig;
    }

    void disableLookahead(
        std::optional<SamplingConfig> const& samplingConfig, SizeType32 batchSize, TensorConstPtr batchSlots) override;

private:
    std::shared_ptr<BufferManager> mManager;
    std::shared_ptr<suggestify::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
    std::shared_ptr<suggestify::runtime::DecodingLayerWorkspace> mDecodingLayerWorkspace;

    SamplingConfig mSamplingConfig;

    size_t mMaxBatchSize;
    size_t mVocabSize;
    size_t mVocabSizePadded;

    executor::DecodingMode mDecodingMode;
};

inline std::unique_ptr<IGptDecoder> IGptDecoder::create(executor::DecodingMode const& mode, nvinfer1::DataType dtype,
    size_t maxBatchSize, size_t maxBeamWidth, size_t vocabSize, size_t vocabSizePadded, size_t maxSequenceLength,
    BufferManager::CudaStreamPtr const& stream,
    std::shared_ptr<SpeculativeDecodingModule const> const& speculativeDecodingModule)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return std::make_unique<GptDecoder<float>>(mode, maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded,
            maxSequenceLength, stream, speculativeDecodingModule);
    case nvinfer1::DataType::kHALF:
        return std::make_unique<GptDecoder<half>>(mode, maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded,
            maxSequenceLength, stream, speculativeDecodingModule);
    default:
        THROW("Unsupported decoder data type: %d. Use either kFLOAT or kHALF.", static_cast<int>(dtype));
        return nullptr;
    }
}

inline runtime::ITensor::SharedConstPtr getDefaultBatchSlots(runtime::SizeType32 batchSize)
{
    auto defaultBatchSlots = runtime::BufferManager::pinnedPool(
        runtime::ITensor::makeShape({batchSize}), runtime::TRTDataType<runtime::SizeType32>::value);
    auto range = runtime::BufferRange<runtime::SizeType32>(*defaultBatchSlots);
    std::iota(range.begin(), range.end(), 0);
    return defaultBatchSlots;
}
}
}
