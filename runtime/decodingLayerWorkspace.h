
#pragma once

#include <memory>

#include "suggestify/common/dataType.h"
#include "suggestify/common/workspace.h"
#include "suggestify/layers/decodingParams.h"
#include "bufferManager.h"
#include "iBuffer.h"
#include "iTensor.h"
#include "tllmBuffers.h"

namespace suggestify::runtime
{

class DecodingLayerWorkspace
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TensorUniquePtr = ITensor::UniquePtr;
    using TensorConstPtr = ITensor::SharedConstPtr;
    using BufferPtr = IBuffer::SharedPtr;

    DecodingLayerWorkspace(std::shared_ptr<BufferManager> bufferManager, layers::DecoderDomain const& decoderDomain,
        nvinfer1::DataType logitsType, size_t workspaceBufferSizeInBytes);

    DecodingLayerWorkspace() = delete;

    DecodingLayerWorkspace(DecodingLayerWorkspace const& decodingLayerWorkspace) = delete;

    [[nodiscard]] void* getRawWorkspaceDevicePtr() const;

    template <typename T>
    T* getWorkspaceDevicePtrAs() const
    {
        return reinterpret_cast<T*>(mWorkspaceDeviceBuffer->data());
    };

    [[nodiscard]] BufferPtr getWorkspaceDeviceBuffer() const;

    void setDeviceBatchSlots(TensorConstPtr const& newBatchSlots);

    [[nodiscard]] SizeType32 const* getDeviceBatchSlotsPtr() const;

    [[nodiscard]] TensorConstPtr getDeviceBatchSlots() const;

    [[nodiscard]] TensorPtr getDeviceRuntimeLogits() const;

    TensorPtr getWorkspaceAsDeviceTensor(ITensor::Shape shape, nvinfer1::DataType type);

    template <typename T, typename Alloc>
    static void copyToWorkspace(runtime::BufferManager const& bufferManager, std::vector<T, Alloc> const& src,
        runtime::IBuffer::SharedPtr workspace)
    {
        auto const sizeOfWorkspaceInBytes = workspace->getSizeInBytes();
        auto const sizeOfSrcInBytes = sizeof(T) * src.size();
        TLLM_CHECK_WITH_INFO(sizeOfSrcInBytes <= sizeOfWorkspaceInBytes,
            "The size of the workspace (%zu bytes) is insufficient for the data (%zu bytes)", sizeOfWorkspaceInBytes,
            sizeOfSrcInBytes);
        auto const sizePerWorkspaceElement = BufferDataType(workspace->getDataType()).getSize();
        TLLM_CHECK_WITH_INFO(sizePerWorkspaceElement == 1 || sizePerWorkspaceElement == sizeof(T),
            "Copy to typed workspace, but element size mismatched (src: %zu, workspace: %zu)", sizeof(T),
            sizePerWorkspaceElement);
        runtime::IBuffer::SharedPtr workspaceSlice
            = runtime::IBuffer::slice(workspace, 0, sizeOfSrcInBytes / sizePerWorkspaceElement);
        bufferManager.copy(src.data(), *workspaceSlice, runtime::MemoryType::kCPU);
    }

    template <typename T>
    TensorPtr copyToWorkspace(std::vector<T> const& src)
    {
        copyToWorkspace(*mBufferManager, src, mWorkspaceDeviceBuffer);
        return getWorkspaceAsDeviceTensor(
            ITensor::makeShape({static_cast<SizeType32>(src.size())}), TRTDataType<T>::value);
    }

    void resize(size_t minSize);

    template <typename... Args>
    size_t static calculateRequiredWorkspaceSize(Args&&... args)
    {
        size_t lastTensorOffset = 0;
        auto alignedSizeCalculator
            = [&lastTensorOffset](std::pair<ITensor::Shape, nvinfer1::DataType> const& tensorDescriptor)
        {
            auto const& [shape, type] = tensorDescriptor;
            auto const sizeInBytes = ITensor::volume(shape) * suggestify::common::getDTypeSize(type);
            auto const sliceEnd = lastTensorOffset + sizeInBytes;
            lastTensorOffset = suggestify::common::alignSize(sliceEnd, suggestify::common::kCudaMemAlign);
        };
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);
        forEach(alignedSizeCalculator, argTuple);
        return lastTensorOffset;
    }

    template <typename... Args>
    auto mirrorInWorkspace(Args&&... args)
    {
        auto* lastTensorEndPtr = reinterpret_cast<std::int8_t*>(mWorkspaceDeviceBuffer->data());
        auto tensorFactory = [&lastTensorEndPtr, this](auto const& tensor)
        {
            if (tensor == nullptr)
            {
                return std::unique_ptr<GenericTensor<BorrowingAllocator<MemoryType::kGPU>>>{};
            }
            auto const sizeInBytes = tensor->getSizeInBytes();
            auto const borrowingAllocator = BorrowingAllocator<MemoryType::kGPU>{lastTensorEndPtr, sizeInBytes};
            auto res = std::make_unique<GenericTensor<BorrowingAllocator<MemoryType::kGPU>>>(
                tensor->getShape(), tensor->getDataType(), borrowingAllocator);
            auto const sliceEnd = lastTensorEndPtr + sizeInBytes;
            lastTensorEndPtr = suggestify::common::alignPtr(sliceEnd, suggestify::common::kCudaMemAlign);
            mBufferManager->copy(*tensor, *res);
            return res;
        };
        auto argTuple = std::make_tuple(std::forward<Args>(args)...);

        auto res = transform(tensorFactory, argTuple);
        std::size_t const numArgs = sizeof...(Args);
        std::size_t const sizeInBytes
            = lastTensorEndPtr - reinterpret_cast<std::int8_t*>(mWorkspaceDeviceBuffer->data());
        TLLM_LOG_DEBUG("Borrowing %lu bytes of the workspace for %i tensors.", sizeInBytes, numArgs);
        return res;
    }

    void initializeDeviceCurandStates(std::optional<std::vector<uint64_t>> const& randomSeed,
        runtime::SizeType32 batchSize, TensorConstPtr const& batchSlots, TensorPtr& statesDevice);

private:
    std::shared_ptr<BufferManager> mBufferManager;
    TensorPtr mBatchSlotsDevice;
    TensorPtr mRuntimeLogitsDevice;
    TensorPtr
        mCurandStatesDevice;
    BufferPtr mWorkspaceDeviceBuffer;

    cudaStream_t getStream();

    template <typename Func, typename Tuple, std::size_t... I>
    auto static transformImpl(Func&& func, Tuple&& tuple, std::index_sequence<I...>)
    {
        return std::make_tuple(func(std::get<I>(tuple))...);
    }

    template <typename Func, typename... Args>
    auto static transform(Func&& func, std::tuple<Args...> const& tuple)
    {
        return transformImpl(std::forward<Func>(func), tuple, std::index_sequence_for<Args...>{});
    }

    template <typename Func, typename Tuple, std::size_t... I>
    void static forEachImpl(Func&& func, Tuple&& tuple, std::index_sequence<I...>)
    {
        (func(std::get<I>(tuple)), ...);
    }

    template <typename Func, typename... Args>
    void static forEach(Func&& func, std::tuple<Args...> const& tuple)
    {
        forEachImpl(std::forward<Func>(func), tuple, std::index_sequence_for<Args...>{});
    }
};

}
