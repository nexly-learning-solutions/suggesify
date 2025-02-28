
#pragma once

#include "../common/arrayView.h"
#include "../common/dataType.h"
#include "../src/decodingCommon.h"
#include "../src/kvCacheIndex.h"
#include "common.h"

#include <NvInferRuntime.h>

#include <cstdint>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <cstddef>
#include <cuda_fp16.h>
#include <memory>
#include <optional>
#include <ostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace suggestify::runtime
{

enum class MemoryType : std::int32_t
{
    kGPU = 0,
    kCPU = 1,
    kPINNED = 2,
    kUVM = 3,
    kPINNEDPOOL = 4
};

template <MemoryType T>
struct MemoryTypeString
{
};

template <>
struct MemoryTypeString<MemoryType::kGPU>
{
    static auto constexpr value = "GPU";
};

template <>
struct MemoryTypeString<MemoryType::kCPU>
{
    static auto constexpr value = "CPU";
};

template <>
struct MemoryTypeString<MemoryType::kPINNED>
{
    static auto constexpr value = "PINNED";
};

template <>
struct MemoryTypeString<MemoryType::kUVM>
{
    static auto constexpr value = "UVM";
};

template <>
struct MemoryTypeString<MemoryType::kPINNEDPOOL>
{
    static auto constexpr value = "PINNEDPOOL";
};

template <nvinfer1::DataType kDataType, bool kIsUnsigned = false, bool kIsPointer = false>
struct DataTypeTraits
{
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kFLOAT>
{
    using type = float;
    static char constexpr name[] = "float";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kHALF>
{
    using type = half;
    static char constexpr name[] = "half";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT8>
{
    using type = std::int8_t;
    static char constexpr name[] = "int8";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT32>
{
    using type = std::int32_t;
    static char constexpr name[] = "int32";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT64>
{
    using type = std::int64_t;
    static char constexpr name[] = "int64";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT32, true>
{
    using type = std::uint32_t;
    static char constexpr name[] = "uint32";
    static auto constexpr size = sizeof(type);
};

template <>
struct DataTypeTraits<nvinfer1::DataType::kINT64, true>
{
    using type = std::uint64_t;
    static char constexpr name[] = "uint64";
    static auto constexpr size = sizeof(type);
};

template <bool kUnsigned>
struct DataTypeTraits<nvinfer1::DataType::kBOOL, kUnsigned>
{
    using type = bool;
    static char constexpr name[] = "bool";
    static auto constexpr size = sizeof(type);
};

template <bool kUnsigned>
struct DataTypeTraits<nvinfer1::DataType::kUINT8, kUnsigned>
{
    using type = std::uint8_t;
    static char constexpr name[] = "uint8";
    static auto constexpr size = sizeof(type);
};

#ifdef ENABLE_BF16
template <>
struct DataTypeTraits<nvinfer1::DataType::kBF16>
{
    using type = __nv_bfloat16;
    static char constexpr name[] = "bfloat16";
    static auto constexpr size = sizeof(type);
};
#endif

#ifdef ENABLE_FP8
template <>
struct DataTypeTraits<nvinfer1::DataType::kFP8>
{
    using type = __nv_fp8_e4m3;
    static char constexpr name[] = "fp8";
    static auto constexpr size = sizeof(type);
};
#endif

template <nvinfer1::DataType kDataType, bool kUnsigned>
struct DataTypeTraits<kDataType, kUnsigned, true>
{
    using type = typename DataTypeTraits<kDataType, kUnsigned, false>::type*;
    static char constexpr name[] = "*";
    static auto constexpr size = sizeof(type);
};

class BufferDataType
{
public:
    constexpr BufferDataType(
        nvinfer1::DataType dataType, bool _unsigned = false, bool pointer = false)
        : mDataType{dataType}
        , mUnsigned{_unsigned}
        , mPointer{pointer}
    {
    }

    static auto constexpr kTrtPointerType = nvinfer1::DataType::kINT64;

    constexpr operator nvinfer1::DataType() const noexcept
    {
        return mPointer ? kTrtPointerType : mDataType;
    }

    [[nodiscard]] constexpr nvinfer1::DataType getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] constexpr bool isPointer() const noexcept
    {
        return mPointer;
    }

    [[nodiscard]] constexpr bool isUnsigned() const
    {
        switch (mDataType)
        {
        case nvinfer1::DataType::kBOOL: [[fallthrough]];
        case nvinfer1::DataType::kUINT8: return true;
        default: return mUnsigned;
        }
    }

    [[nodiscard]] constexpr std::size_t getSize() const noexcept
    {
        return suggestify::common::getDTypeSize(static_cast<nvinfer1::DataType>(*this));
    }

private:
    nvinfer1::DataType mDataType;
    bool mUnsigned;
    bool mPointer;
};

template <typename T, bool = false>
struct TRTDataType
{
};

template <>
struct TRTDataType<float>
{
    static constexpr auto value = nvinfer1::DataType::kFLOAT;
};

template <>
struct TRTDataType<half>
{
    static constexpr auto value = nvinfer1::DataType::kHALF;
};

template <>
struct TRTDataType<std::int8_t>
{
    static constexpr auto value = nvinfer1::DataType::kINT8;
};

template <>
struct TRTDataType<std::int32_t>
{
    static constexpr auto value = nvinfer1::DataType::kINT32;
};

template <>
struct TRTDataType<std::uint32_t>
{
    static constexpr auto value = BufferDataType{nvinfer1::DataType::kINT32, true};
};

template <>
struct TRTDataType<std::int64_t>
{
    static constexpr auto value = nvinfer1::DataType::kINT64;
};

template <>
struct TRTDataType<std::uint64_t>
{
    static constexpr auto value = BufferDataType{nvinfer1::DataType::kINT64, true};
};

template <>
struct TRTDataType<bool>
{
    static constexpr auto value = nvinfer1::DataType::kBOOL;
};

template <>
struct TRTDataType<std::uint8_t>
{
    static constexpr auto value = nvinfer1::DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <>
struct TRTDataType<__nv_bfloat16>
{
    static constexpr auto value = nvinfer1::DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TRTDataType<__nv_fp8_e4m3>
{
    static constexpr auto value = nvinfer1::DataType::kFP8;
};
#endif

template <>
struct TRTDataType<kernels::KVCacheIndex>
{
    static constexpr auto value = TRTDataType<kernels::KVCacheIndex::UnderlyingType>::value;
};

template <>
struct TRTDataType<kernels::FinishedState>
{
    static constexpr auto value = TRTDataType<kernels::FinishedState::UnderlyingType>::value;
};

template <>
struct TRTDataType<runtime::RequestType>
{
    static constexpr auto value = TRTDataType<std::underlying_type_t<runtime::RequestType>>::value;
};

template <>
struct TRTDataType<void*>
{
    static constexpr auto value = BufferDataType::kTrtPointerType;
};

template <typename T>
struct TRTDataType<T*>
{
private:
    static auto constexpr kUnderlyingType = BufferDataType{TRTDataType<std::remove_const_t<T>, false>::value};

public:
    static auto constexpr value = BufferDataType{kUnderlyingType.getDataType(), kUnderlyingType.isUnsigned(), true};
};

template <typename T>
using PointerElementType = typename std::remove_reference_t<T>::element_type;

template <typename T>
std::shared_ptr<std::remove_const_t<T>> constPointerCast(std::shared_ptr<T> const& ptr) noexcept
{
    return std::const_pointer_cast<std::remove_const_t<T>>(ptr);
}

template <typename T, typename D>
std::shared_ptr<std::remove_const_t<T>> constPointerCast(std::unique_ptr<T, D>&& ptr) noexcept
{
    return std::const_pointer_cast<std::remove_const_t<T>>(std::shared_ptr(std::move(ptr)));
}

class IBuffer
{
public:
    using UniquePtr = std::unique_ptr<IBuffer>;
    using SharedPtr = std::shared_ptr<IBuffer>;
    using UniqueConstPtr = std::unique_ptr<IBuffer const>;
    using SharedConstPtr = std::shared_ptr<IBuffer const>;
    using DataType = nvinfer1::DataType;

    [[nodiscard]] virtual void* data() = 0;

    [[nodiscard]] virtual void const* data() const = 0;

    [[nodiscard]] virtual void* data(std::size_t index)
    {
        auto* const dataPtr = this->data();
        return dataPtr ? static_cast<std::uint8_t*>(dataPtr) + toBytes(index) : nullptr;
    }

    [[nodiscard]] virtual void const* data(std::size_t index) const
    {
        auto const* const dataPtr = this->data();
        return dataPtr ? static_cast<std::uint8_t const*>(dataPtr) + toBytes(index) : nullptr;
    }

    [[nodiscard]] virtual std::size_t getSize() const = 0;

    [[nodiscard]] virtual std::size_t getSizeInBytes() const
    {
        return toBytes(getSize());
    }

    [[nodiscard]] virtual std::size_t getCapacity() const = 0;

    [[nodiscard]] virtual DataType getDataType() const = 0;

    [[nodiscard]] virtual char const* getDataTypeName() const;

    [[nodiscard]] virtual MemoryType getMemoryType() const = 0;

    [[nodiscard]] virtual char const* getMemoryTypeName() const;

    virtual void resize(std::size_t newSize) = 0;

    virtual void release() = 0;

    virtual ~IBuffer() = default;

    IBuffer(IBuffer const&) = delete;

    IBuffer& operator=(IBuffer const&) = delete;

    static UniquePtr slice(SharedPtr buffer, std::size_t offset, std::size_t size);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset, std::size_t size)
    {
        return IBuffer::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset, size);
    }

    static UniquePtr slice(SharedPtr buffer, std::size_t offset)
    {
        auto const size = buffer->getSize() - offset;
        return IBuffer::slice(std::move(buffer), offset, size);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset)
    {
        return IBuffer::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset);
    }

    static UniquePtr view(SharedPtr tensor)
    {
        auto constexpr offset = 0;
        return IBuffer::slice(std::move(tensor), offset);
    }

    static UniquePtr view(SharedPtr tensor, std::size_t size)
    {
        auto v = IBuffer::view(std::move(tensor));
        v->resize(size);
        return v;
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr view(TConstPtr&& tensor, std::size_t size)
    {
        return IBuffer::view(constPointerCast(std::forward<TConstPtr>(tensor)), size);
    }

    static UniquePtr wrap(void* data, DataType type, std::size_t size, std::size_t capacity);

    static UniquePtr wrap(void* data, DataType type, std::size_t size)
    {
        return wrap(data, type, size, size);
    }

    template <typename T>
    static UniquePtr wrap(T* data, std::size_t size, std::size_t capacity)
    {
        return wrap(data, TRTDataType<T>::value, size, capacity);
    }

    template <typename T>
    static UniquePtr wrap(T* data, std::size_t size)
    {
        return wrap<T>(data, size, size);
    }

    template <typename T>
    static UniquePtr wrap(std::vector<T>& v)
    {
        return wrap<T>(v.data(), v.size(), v.capacity());
    }

    static MemoryType memoryType(void const* data);

protected:
    IBuffer() = default;

    [[nodiscard]] std::size_t toBytes(std::size_t size) const
    {
        return size * BufferDataType(getDataType()).getSize();
    }
};

template <typename T>
T const* bufferCast(IBuffer const& buffer)
{
    if (TRTDataType<typename std::remove_cv<T>::type>::value != buffer.getDataType())
    {
        throw std::bad_cast();
    }
    return static_cast<T const*>(buffer.data());
}

template <typename T>
T* bufferCast(IBuffer& buffer)
{
    if (TRTDataType<typename std::remove_cv<T>::type>::value != buffer.getDataType())
    {
        throw std::bad_cast();
    }
    return static_cast<T*>(buffer.data());
}

template <typename T>
T* bufferCastOrNull(IBuffer::SharedPtr const& bufferPtr)
{
    if (bufferPtr)
    {
        return bufferCast<T>(*bufferPtr);
    }

    return static_cast<T*>(nullptr);
}

template <typename T>
T const* bufferCastOrNull(IBuffer::SharedConstPtr const& bufferPtr)
{
    if (bufferPtr)
    {
        return bufferCast<T>(*bufferPtr);
    }

    return static_cast<T const*>(nullptr);
}

template <typename T>
T* bufferCastOrNull(std::optional<IBuffer::SharedPtr> const& optionalBufferPtr)
{
    if (optionalBufferPtr)
    {
        return bufferCast<T>(*optionalBufferPtr.value());
    }

    return static_cast<T*>(nullptr);
}

template <typename T>
T const* bufferCastOrNull(std::optional<IBuffer::SharedConstPtr> const& optionalBufferPtr)
{
    if (optionalBufferPtr)
    {
        return bufferCast<T>(*optionalBufferPtr.value());
    }

    return static_cast<T const*>(nullptr);
}

template <typename T>
class BufferRange : public suggestify::common::ArrayView<T>
{
public:
    using Base = suggestify::common::ArrayView<T>;
    using typename Base::size_type;

    BufferRange(T* data, size_type size)
        : Base{data, size}
    {
    }

    template <typename U = T, std::enable_if_t<!std::is_const_v<U>, bool> = true>
    explicit BufferRange(IBuffer& buffer)
        : BufferRange(bufferCast<U>(buffer), buffer.getSize())
    {
    }

    template <typename U = T, std::enable_if_t<std::is_const_v<U>, bool> = true>
    explicit BufferRange(IBuffer const& buffer)
        : BufferRange(bufferCast<U>(buffer), buffer.getSize())
    {
    }
};

std::ostream& operator<<(std::ostream& output, IBuffer const& buffer);

}
