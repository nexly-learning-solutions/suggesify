
#pragma once

#include "../common/assert.h"
#include "cudaStream.h"
#include "iBuffer.h"
#include "iTensor.h"
#include <NvInferRuntime.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

class BufferManagerTest;

namespace suggestify::runtime
{

class CudaMemPool;

class BufferManager
{
public:
    using IBufferPtr = IBuffer::UniquePtr;

    using ITensorPtr = ITensor::UniquePtr;

    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using CudaMemPoolPtr = std::shared_ptr<CudaMemPool>;

    explicit BufferManager(CudaStreamPtr stream, bool trimPool = false);

    ~BufferManager()
    {
        if (mTrimPool)
        {
            memoryPoolTrimTo(0);
        }
    }

    static auto constexpr kBYTE_TYPE = nvinfer1::DataType::kUINT8;

    [[nodiscard]] IBufferPtr gpu(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE) const;

    [[nodiscard]] ITensorPtr gpu(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE) const;

    [[nodiscard]] static IBufferPtr gpuSync(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static ITensorPtr gpuSync(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static IBufferPtr cpu(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static ITensorPtr cpu(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static IBufferPtr pinned(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static ITensorPtr pinned(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static IBufferPtr pinnedPool(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static ITensorPtr pinnedPool(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static IBufferPtr managed(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] static ITensorPtr managed(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    [[nodiscard]] IBufferPtr allocate(
        MemoryType memoryType, std::size_t size, nvinfer1::DataType type = kBYTE_TYPE) const;

    [[nodiscard]] ITensorPtr allocate(
        MemoryType memoryType, nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE) const;

    [[nodiscard]] IBufferPtr emptyBuffer(MemoryType memoryType, nvinfer1::DataType type = kBYTE_TYPE) const
    {
        return allocate(memoryType, 0, type);
    }

    [[nodiscard]] ITensorPtr emptyTensor(MemoryType memoryType, nvinfer1::DataType type = kBYTE_TYPE) const
    {
        return allocate(memoryType, ITensor::makeShape({}), type);
    }

    void setMem(IBuffer& buffer, int32_t value) const;

    void setZero(IBuffer& buffer) const;

    void copy(void const* src, IBuffer& dst, MemoryType srcType) const;

    void copy(IBuffer const& src, void* dst, MemoryType dstType) const;

    void copy(void const* src, IBuffer& dst) const
    {
        return copy(src, dst, IBuffer::memoryType(src));
    }

    void copy(IBuffer const& src, void* dst) const
    {
        return copy(src, dst, IBuffer::memoryType(dst));
    }

    void copy(IBuffer const& src, IBuffer& dst) const;

    [[nodiscard]] IBufferPtr copyFrom(IBuffer const& src, MemoryType memoryType) const;

    [[nodiscard]] ITensorPtr copyFrom(ITensor const& src, MemoryType memoryType) const;

    template <typename T>
    [[nodiscard]] IBufferPtr copyFrom(std::vector<T> const& src, MemoryType memoryType) const
    {
        auto buffer = allocate(memoryType, src.size(), TRTDataType<std::remove_cv_t<T>>::value);
        copy(src.data(), *buffer);
        return buffer;
    }

    template <typename T>
    [[nodiscard]] ITensorPtr copyFrom(T* src, nvinfer1::Dims dims, MemoryType memoryType) const
    {
        auto buffer = allocate(memoryType, dims, TRTDataType<std::remove_cv_t<T>>::value);
        copy(src, *buffer);
        return buffer;
    }

    template <typename T>
    [[nodiscard]] ITensorPtr copyFrom(std::vector<T> const& src, nvinfer1::Dims dims, MemoryType memoryType) const
    {
        CHECK_WITH_INFO(src.size() == ITensor::volumeNonNegative(dims),
            common::fmtstr("[TensorRT-LLM][ERROR] Incompatible size %lu and dims %s", src.size(),
                ITensor::toString(dims).c_str()));
        return copyFrom(src.data(), dims, memoryType);
    }

    [[nodiscard]] CudaStream const& getStream() const;

    [[nodiscard]] std::size_t memoryPoolReserved() const;

    [[nodiscard]] std::size_t memoryPoolUsed() const;

    [[nodiscard]] std::size_t memoryPoolFree() const;

    void memoryPoolTrimTo(std::size_t size);

private:
    friend class ::BufferManagerTest;

    CudaStreamPtr mStream;
    CudaMemPoolPtr mPool;
    bool const mTrimPool;
};

}
