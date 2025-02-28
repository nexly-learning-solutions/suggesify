
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"
#include "cudaMemPool.h"
#include "cudaStream.h"
#include "iBuffer.h"
#include "iTensor.h"
#include "memoryCounters.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <vector>

namespace suggestify::runtime
{

template <typename TDerived, MemoryType memoryType, bool count = true>
class BaseAllocator
{
public:
    using ValueType = void;
    using PointerType = ValueType*;
    static auto constexpr kMemoryType = memoryType;

    PointerType allocate(std::size_t n)
    {
        PointerType ptr{};
        static_cast<TDerived*>(this)->allocateImpl(&ptr, n);
        if constexpr (count)
        {
            MemoryCounters::getInstance().allocate<memoryType>(n);
        }
        return ptr;
    }

    void deallocate(PointerType ptr, std::size_t n)
    {
        if (ptr)
        {
            static_cast<TDerived*>(this)->deallocateImpl(ptr, n);
            if constexpr (count)
            {
                MemoryCounters::getInstance().deallocate<memoryType>(n);
            }
        }
    }

    [[nodiscard]] MemoryType constexpr getMemoryType() const
    {
        return memoryType;
    }
};

class CudaAllocator : public BaseAllocator<CudaAllocator, MemoryType::kGPU>
{
    friend class BaseAllocator<CudaAllocator, MemoryType::kGPU>;

public:
    CudaAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        CUDA_CHECK(::cudaMalloc(ptr, n));
    }

    void deallocateImpl(
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        CUDA_CHECK_FREE_RESOURCE(::cudaFree(ptr));
    }
};

class CudaAllocatorAsync : public BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>
{
    friend class BaseAllocator<CudaAllocatorAsync, MemoryType::kGPU>;

public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using CudaPoolPtr = std::shared_ptr<CudaMemPool>;

    explicit CudaAllocatorAsync(CudaStreamPtr stream, CudaPoolPtr memPool)
        : mCudaStream(std::move(stream))
        , mMemPool(std::move(memPool))
    {
        CHECK_WITH_INFO(static_cast<bool>(mCudaStream), "Undefined CUDA stream");
        CHECK_WITH_INFO(static_cast<bool>(mMemPool), "Undefined CUDA mem pool");
    }

    [[nodiscard]] CudaStreamPtr getCudaStream() const
    {
        return mCudaStream;
    }

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        CUDA_CHECK(::cudaMallocAsync(ptr, n, mMemPool->getPool(), mCudaStream->get()));
    }

    void deallocateImpl(PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        CUDA_CHECK_FREE_RESOURCE(::cudaFreeAsync(ptr, mCudaStream->get()));
    }

private:
    CudaStreamPtr mCudaStream;
    CudaPoolPtr mMemPool;
};

class UVMAllocator : public BaseAllocator<UVMAllocator, MemoryType::kUVM>
{
    friend class BaseAllocator<UVMAllocator, MemoryType::kUVM>;

public:
    using Base = BaseAllocator<UVMAllocator, MemoryType::kUVM>;
    UVMAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        CUDA_CHECK(::cudaMallocManaged(ptr, n));
    }

    void deallocateImpl(
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        CUDA_CHECK_FREE_RESOURCE(::cudaFree(ptr));
    }
};

class PinnedAllocator : public BaseAllocator<PinnedAllocator, MemoryType::kPINNED>
{
    friend class BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;

public:
    using Base = BaseAllocator<PinnedAllocator, MemoryType::kPINNED>;
    PinnedAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        CUDA_CHECK(::cudaHostAlloc(ptr, n, cudaHostAllocDefault));
    }

    void deallocateImpl(
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        CUDA_CHECK_FREE_RESOURCE(::cudaFreeHost(ptr));
    }
};

class HostAllocator : public BaseAllocator<HostAllocator, MemoryType::kCPU>
{
    friend class BaseAllocator<HostAllocator, MemoryType::kCPU>;

public:
    HostAllocator() noexcept = default;

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        *ptr = std::malloc(n);
        if (*ptr == nullptr)
        {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl(
        PointerType ptr, [[maybe_unused]] std::size_t n)
    {
        std::free(ptr);
    }
};

template <MemoryType memoryType>
class BorrowingAllocator : public BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>
{
    friend class BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;

public:
    using Base = BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;
    using PointerType = typename Base::PointerType;

    BorrowingAllocator(void* ptr, std::size_t capacity)
        : mPtr(ptr)
        , mCapacity(capacity)
    {
        CHECK_WITH_INFO(capacity == std::size_t(0) || static_cast<bool>(mPtr), "Undefined pointer");
    }

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        if (n <= mCapacity)
        {
            *ptr = mPtr;
        }
        else
        {
            throw std::bad_alloc();
        }
    }

    void deallocateImpl(
        [[maybe_unused]] PointerType ptr, [[maybe_unused]] std::size_t n)
    {
    }

private:
    PointerType mPtr;
    std::size_t mCapacity;
};

using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;
using GpuBorrowingAllocator = BorrowingAllocator<MemoryType::kGPU>;
using PinnedBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNED>;
using ManagedBorrowingAllocator = BorrowingAllocator<MemoryType::kUVM>;
using PinnedPoolBorrowingAllocator = BorrowingAllocator<MemoryType::kPINNEDPOOL>;


template <typename TAllocator>
class MemoryPool : public BaseAllocator<MemoryPool<TAllocator>, TAllocator::kMemoryType, false>
{
    friend class BaseAllocator<MemoryPool<TAllocator>, TAllocator::kMemoryType, false>;

public:
    using Base = BaseAllocator<MemoryPool<TAllocator>, TAllocator::kMemoryType, false>;
    using PointerType = typename Base::PointerType;

    using Allocator = TAllocator;
    static_assert(std::is_same_v<typename Allocator::PointerType, PointerType>);

    static std::size_t constexpr kInitialChunkSize{std::size_t{1} << 29};
    static std::size_t constexpr kAlignment{256};

    explicit MemoryPool(std::size_t chunkSize = kInitialChunkSize, Allocator allocator = Allocator{})
        : mChunkSize(chunkSize)
        , mAllocator{allocator}
    {
    }

    ~MemoryPool()
    {
        std::lock_guard<std::mutex> lock(mLock);
        LOG_DEBUG("MemoryPool: Deallocating %zu chunks", mAllocatedChunks.size());
        for (auto const& [ptr, size] : mAllocatedChunks)
        {
            LOG_DEBUG("MemoryPool: Deallocating %zu B", size);
            try
            {
                mAllocator.deallocate(ptr, size);
            }
            catch (std::exception const& e)
            {
                LOG_EXCEPTION(e);
            }
        }
        mAllocatedChunks.clear();
    }

    [[nodiscard]] std::size_t getChunkSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return mChunkSize;
    }

    void setChunkSize(std::size_t chunkSize)
    {
        std::lock_guard<std::mutex> lock(mLock);
        mChunkSize = chunkSize;
    }

    [[nodiscard]] std::size_t getUsedSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return std::accumulate(mMemorySegments.cbegin(), mMemorySegments.cend(), std::size_t{0},
            [](std::size_t sum, auto const& chunk) { return chunk.tag ? sum + chunk.size : sum; });
    }

    [[nodiscard]] std::size_t getReservedSize() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return std::accumulate(mAllocatedChunks.cbegin(), mAllocatedChunks.cend(), std::size_t{0},
            [](std::size_t sum, auto const& chunk) { return sum + std::get<1>(chunk); });
    }

    class MemorySegment
    {
    public:
        MemorySegment(PointerType basePointer, std::size_t size, std::size_t offset = 0, PointerType tag = nullptr)
            : basePointer{basePointer}
            , size{size}
            , offset{offset}
            , tag{tag}
        {
        }

        PointerType const basePointer;
        std::size_t size;
        std::size_t offset;
        PointerType tag;
    };

    std::list<MemorySegment> const& getMemorySegments() const
    {
        std::lock_guard<std::mutex> lock(mLock);
        return mMemorySegments;
    }

    void logSegments() const;

protected:
    void allocateImpl(PointerType* ptr, std::size_t requestedSize);

    void deallocateImpl(PointerType tag, std::size_t n);

private:
    std::size_t mChunkSize;
    TAllocator mAllocator;
    std::mutex mutable mLock{};

    std::list<MemorySegment> mMemorySegments = {};
    std::vector<std::tuple<PointerType, std::size_t>> mAllocatedChunks = {};

    void allocateChunk()
    {
        LOG_DEBUG("MemoryPool: Allocating %zu B", mChunkSize);
        auto basePointer = mAllocator.allocate(mChunkSize);
        mAllocatedChunks.emplace_back(basePointer, mChunkSize);
        mMemorySegments.push_back(MemorySegment{basePointer, mChunkSize});
    }
};

template <typename TAllocator>
void MemoryPool<TAllocator>::allocateImpl(MemoryPool::PointerType* ptr, std::size_t requestedSize)
{
    std::lock_guard<std::mutex> lock(mLock);

    std::size_t const alignedRequest{
        requestedSize == 0 ? kAlignment : common::ceilDiv(requestedSize, kAlignment) * kAlignment};

    LOG_DEBUG("MemoryPool: Requested to reserve %zu B (%zu B aligned)", requestedSize, alignedRequest);

    auto it = std::find_if(mMemorySegments.begin(), mMemorySegments.end(),
        [alignedRequest](auto const& ms) { return ms.tag == nullptr && ms.size >= alignedRequest; });

    if (it == mMemorySegments.end())
    {
        LOG_DEBUG("MemoryPool: Needs more space to accommodate request of %zu B", requestedSize);
        if (mChunkSize < alignedRequest)
        {
            mChunkSize = alignedRequest;
            LOG_DEBUG("MemoryPool: Increasing chunk size to %zu B", mChunkSize);
        }
        allocateChunk();
        it = std::prev(mMemorySegments.end());
    }

    auto const offset = it->offset;
    auto const basePointer = it->basePointer;

    it->offset += alignedRequest;
    it->size -= alignedRequest;
    if (it->size == 0)
    {
        it = mMemorySegments.erase(it);
    }

    *ptr = static_cast<PointerType>(static_cast<std::uint8_t*>(basePointer) + offset);

    mMemorySegments.insert(it, MemorySegment{basePointer, alignedRequest, offset, *ptr});
}

template <typename TAllocator>
void MemoryPool<TAllocator>::deallocateImpl(PointerType tag, std::size_t n)
{
    std::lock_guard<std::mutex> lock(mLock);
    auto it = std::find_if(mMemorySegments.begin(), mMemorySegments.end(),
        [&tag](MemorySegment const& segment) { return segment.tag == tag; });

    CHECK_WITH_INFO(it != mMemorySegments.end(), "MemoryPool free: Requested tag %p could not be found", tag);

    it->tag = nullptr;

    if (it->size < n)
    {
        LOG_WARNING("MemoryPool: Requested to free %zu B, but only %zu B available", n, it->size);
    }

    if (it != mMemorySegments.begin())
    {
        auto previousIt = std::prev(it);
        if (previousIt->tag == nullptr && previousIt->basePointer == it->basePointer)
        {
            previousIt->size += it->size;
            it = std::prev(mMemorySegments.erase(it));
        }
    }

    if (std::next(it) != mMemorySegments.end())
    {
        auto nextIt = std::next(it);
        if (nextIt->tag == nullptr && nextIt->basePointer == it->basePointer)
        {
            it->size += nextIt->size;
            mMemorySegments.erase(nextIt);
        }
    }
}

template <typename TAllocator>
void MemoryPool<TAllocator>::logSegments() const
{
    std::lock_guard<std::mutex> lock(mLock);
    LOG_DEBUG("MemoryPool segments:");
    for (auto ms : mMemorySegments)
    {
        LOG_DEBUG("* Segment size %zu, tag %p, basePointer %p", ms.size, ms.tag, ms.basePointer);
    }
}

template <typename TAllocator>
class PoolAllocator : public BaseAllocator<PoolAllocator<TAllocator>, TAllocator::kMemoryType, false>
{
    friend class BaseAllocator<PoolAllocator<TAllocator>, TAllocator::kMemoryType, false>;

public:
    using Base = BaseAllocator<PoolAllocator<TAllocator>, TAllocator::kMemoryType, false>;
    using PointerType = typename Base::PointerType;
    using PoolType = MemoryPool<TAllocator>;

    static PoolType& getPool();

protected:
    void allocateImpl(PointerType* ptr, std::size_t n)
    {
        *ptr = getPool().allocate(n);
    }

    void deallocateImpl(
        typename TAllocator::PointerType ptr, std::size_t n)
    {
        getPool().deallocate(ptr, n);
    }
};

using PinnedPoolAllocator = PoolAllocator<PinnedAllocator>;


template <typename TAllocator>
class GenericBuffer : virtual public IBuffer
{
public:
    using AllocatorType = TAllocator;

    explicit GenericBuffer(nvinfer1::DataType type, TAllocator allocator = {})
        : GenericBuffer{0, type, std::move(allocator)} {};

    explicit GenericBuffer(
        std::size_t size, nvinfer1::DataType type, TAllocator allocator = {})
        : GenericBuffer{size, size, type, std::move(allocator)} {};

    GenericBuffer(GenericBuffer&& buf) noexcept
        : mSize{buf.mSize}
        , mCapacity{buf.mCapacity}
        , mType{buf.mType}
        , mAllocator{std::move(buf.mAllocator)}
        , mBuffer{buf.mBuffer}
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf) noexcept
    {
        if (this != &buf)
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mAllocator = std::move(buf.mAllocator);
            mBuffer = buf.mBuffer;
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    void* data() override
    {
        return LIKELY(mSize > 0) ? mBuffer : nullptr;
    }

    [[nodiscard]] void const* data() const override
    {
        return LIKELY(mSize > 0) ? mBuffer : nullptr;
    }

    [[nodiscard]] std::size_t getSize() const override
    {
        return mSize;
    }

    [[nodiscard]] std::size_t getCapacity() const override
    {
        return mCapacity;
    }

    [[nodiscard]] nvinfer1::DataType getDataType() const override
    {
        return mType;
    }

    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return mAllocator.getMemoryType();
    }

    void resize(std::size_t newSize) override
    {
        if (mCapacity < newSize)
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mBuffer = mAllocator.allocate(toBytes(newSize));
            mCapacity = newSize;
        }
        mSize = newSize;
    }

    void release() override
    {
        mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        mSize = 0;
        mCapacity = 0;
        mBuffer = nullptr;
    }

    ~GenericBuffer() override
    {
        try
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        }
        catch (std::exception const& e)
        {
            LOG_EXCEPTION(e);
        }
    }

protected:
    explicit GenericBuffer(std::size_t size, std::size_t capacity, nvinfer1::DataType type, TAllocator allocator = {})
        : mSize{size}
        , mCapacity{capacity}
        , mType{type}
        , mAllocator{std::move(allocator)}
        , mBuffer{capacity > 0 ? mAllocator.allocate(toBytes(capacity)) : nullptr}
    {
        CHECK(size <= capacity);
        CHECK(capacity == 0 || size > 0);
    }

private:
    std::size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    TAllocator mAllocator;
    void* mBuffer;
};

using DeviceBuffer = GenericBuffer<CudaAllocatorAsync>;
using StaticDeviceBuffer = GenericBuffer<CudaAllocator>;
using HostBuffer = GenericBuffer<HostAllocator>;
using PinnedBuffer = GenericBuffer<PinnedAllocator>;
using PinnedPoolBuffer = GenericBuffer<PinnedPoolAllocator>;
using UVMBuffer = GenericBuffer<UVMAllocator>;

template <typename T>
typename std::make_unsigned<T>::type nonNegative(T value)
{
    CHECK_WITH_INFO(value >= 0, "Value must be non-negative");
    return static_cast<typename std::make_unsigned<T>::type>(value);
}

template <typename TAllocator>
class GenericTensor : virtual public ITensor, public GenericBuffer<TAllocator>
{
public:
    using Base = GenericBuffer<TAllocator>;

    explicit GenericTensor(nvinfer1::DataType type, TAllocator allocator = {})
        : Base{type, std::move(allocator)}
    {
        mDims.nbDims = 0;
    }

    explicit GenericTensor(nvinfer1::Dims const& dims, nvinfer1::DataType type, TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), type, std::move(allocator)}
        , mDims{dims}
    {
    }

    explicit GenericTensor(
        nvinfer1::Dims const& dims, std::size_t capacity, nvinfer1::DataType type, TAllocator allocator = {})
        : Base{nonNegative(volume(dims)), capacity, type, std::move(allocator)}
        , mDims{dims}
    {
    }

    GenericTensor(GenericTensor&& tensor) noexcept
        : Base{std::move(tensor)}
        , mDims{tensor.dims}
    {
        tensor.mDims.nbDims = 0;
    }

    GenericTensor& operator=(GenericTensor&& tensor) noexcept
    {
        if (this != &tensor)
        {
            Base::operator=(std::move(tensor));
            mDims = tensor.dims;
            tensor.mDims.nbDims = 0;
        }
        return *this;
    }

    [[nodiscard]] nvinfer1::Dims const& getShape() const override
    {
        return mDims;
    }

    void reshape(nvinfer1::Dims const& dims) override
    {
        Base::resize(nonNegative(volume(dims)));
        mDims = dims;
    }

    void resize(std::size_t newSize) override
    {
        ITensor::resize(newSize);
    }

    void release() override
    {
        Base::release();
        mDims.nbDims = 0;
    }

private:
    nvinfer1::Dims mDims{};
};

using DeviceTensor = GenericTensor<CudaAllocatorAsync>;
using StaticDeviceTensor = GenericTensor<CudaAllocator>;
using HostTensor = GenericTensor<HostAllocator>;
using PinnedTensor = GenericTensor<PinnedAllocator>;
using PinnedPoolTensor = GenericTensor<PinnedPoolAllocator>;
using UVMTensor = GenericTensor<UVMAllocator>;

}
