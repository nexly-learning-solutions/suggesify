
#include "decodingLayerWorkspace.h"

#include <utility>

suggestify::runtime::DecodingLayerWorkspace::DecodingLayerWorkspace(std::shared_ptr<BufferManager> bufferManager,
    suggestify::layers::DecoderDomain const& decoderDomain, nvinfer1::DataType logitsType,
    size_t workspaceBufferSizeInBytes)
    : mBufferManager(std::move(bufferManager))
    , mBatchSlotsDevice(
          mBufferManager->gpu(ITensor::makeShape({decoderDomain.getBatchSize()}), TRTDataType<SizeType32>::value))
    , mRuntimeLogitsDevice(
          mBufferManager->gpu(ITensor::makeShape({decoderDomain.getBatchSize(), decoderDomain.getMaxDecodingTokens(),
                                  decoderDomain.getBeamWidth(), decoderDomain.getVocabSizePadded()}),
              logitsType))
    , mCurandStatesDevice(
          mBufferManager->gpu(ITensor::makeShape({decoderDomain.getBatchSize(), sizeof(curandState_t)})))
    , mWorkspaceDeviceBuffer(mBufferManager->gpu(workspaceBufferSizeInBytes))
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("Creating decoding workspace for a maximum batch size of %i, with a scratch space of %lu bytes",
        decoderDomain.getBatchSize(), workspaceBufferSizeInBytes);
    mBufferManager->getStream().synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void* suggestify::runtime::DecodingLayerWorkspace::getRawWorkspaceDevicePtr() const
{
    return mWorkspaceDeviceBuffer->data();
}

suggestify::runtime::DecodingLayerWorkspace::BufferPtr
suggestify::runtime::DecodingLayerWorkspace::getWorkspaceDeviceBuffer() const
{
    return mWorkspaceDeviceBuffer;
}

void suggestify::runtime::DecodingLayerWorkspace::setDeviceBatchSlots(TensorConstPtr const& newBatchSlots)
{
    mBatchSlotsDevice->reshape(newBatchSlots->getShape());
    mBufferManager->copy(*newBatchSlots, *mBatchSlotsDevice);
}

suggestify::runtime::SizeType32 const* suggestify::runtime::DecodingLayerWorkspace::getDeviceBatchSlotsPtr() const
{
    return suggestify::runtime::bufferCast<suggestify::runtime::SizeType32>(*mBatchSlotsDevice);
}

suggestify::runtime::DecodingLayerWorkspace::TensorConstPtr
suggestify::runtime::DecodingLayerWorkspace::getDeviceBatchSlots() const
{
    return mBatchSlotsDevice;
}

suggestify::runtime::DecodingLayerWorkspace::TensorPtr
suggestify::runtime::DecodingLayerWorkspace::getDeviceRuntimeLogits() const
{
    return mRuntimeLogitsDevice;
}

void suggestify::runtime::DecodingLayerWorkspace::resize(size_t minSize)
{
    if (mWorkspaceDeviceBuffer->getSizeInBytes() < minSize)
    {
        mWorkspaceDeviceBuffer->resize(minSize);
    }
}

suggestify::runtime::DecodingLayerWorkspace::TensorPtr
suggestify::runtime::DecodingLayerWorkspace::getWorkspaceAsDeviceTensor(ITensor::Shape shape, nvinfer1::DataType type)
{
    auto const sizeInBytes = ITensor::volume(shape) * BufferDataType(type).getSize();
    return std::make_shared<GenericTensor<BorrowingAllocator<MemoryType::kGPU>>>(
        shape, type, BorrowingAllocator<MemoryType::kGPU>{mWorkspaceDeviceBuffer->data(), sizeInBytes});
}

void suggestify::runtime::DecodingLayerWorkspace::initializeDeviceCurandStates(
    std::optional<std::vector<uint64_t>> const& randomSeed, suggestify::runtime::SizeType32 batchSize,
    suggestify::runtime::DecodingLayerWorkspace::TensorConstPtr const& batchSlots,
    suggestify::runtime::DecodingLayerWorkspace::TensorPtr& statesDevice)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const* batchSlotsPtr = suggestify::runtime::bufferCast<suggestify::runtime::SizeType32>(*batchSlots);
    auto* curandStateDevicePtr = reinterpret_cast<curandState_t*>(statesDevice->data());
    if (randomSeed)
    {
        if (randomSeed->size() == 1)
        {
            suggestify::kernels::invokeCurandInitialize(
                curandStateDevicePtr, batchSlotsPtr, batchSize, randomSeed->front(), getStream());
        }
        else
        {
            TLLM_CHECK_WITH_INFO(static_cast<suggestify::runtime::SizeType32>(randomSeed->size()) == batchSize,
                "Random seed vector size mismatch.");
            auto randomSeedsDevice = copyToWorkspace(randomSeed.value());
            auto const* randomSeedsDevicePtr = suggestify::runtime::bufferCast<uint64_t>(*randomSeedsDevice);
            suggestify::kernels::invokeCurandBatchInitialize(
                curandStateDevicePtr, batchSlotsPtr, batchSize, randomSeedsDevicePtr, getStream());
        }
    }
    else
    {
        suggestify::kernels::invokeCurandInitialize(curandStateDevicePtr, batchSlotsPtr, batchSize, 0, getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

cudaStream_t suggestify::runtime::DecodingLayerWorkspace::getStream()
{
    return mBufferManager->getStream().get();
}
