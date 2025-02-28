
#pragma once

#include "../common/assert.h"
#include "common.h"
#include "iBuffer.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <type_traits>

namespace nvinfer1
{
class IExecutionContext;
}

namespace suggestify::runtime
{

class ITensor : virtual public IBuffer
{
public:
    friend class ITensorBindings;

    using UniquePtr = std::unique_ptr<ITensor>;
    using SharedPtr = std::shared_ptr<ITensor>;
    using UniqueConstPtr = std::unique_ptr<ITensor const>;
    using SharedConstPtr = std::shared_ptr<ITensor const>;
    using Shape = nvinfer1::Dims;
    using DimType64 = std::remove_reference_t<decltype(Shape::d[0])>;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    static_assert(std::is_same_v<DimType64, std::int64_t>, "This version of TRT-LLM requires TensorRT 10.0 or later.");

    ~ITensor() override = default;

    [[nodiscard]] virtual Shape const& getShape() const = 0;

    template <SizeType32 n>
    [[nodiscard]] DimType64 getDimension() const
    {
        auto const shape = getShape();
        static_assert(n < shape.MAX_DIMS && n >= -shape.MAX_DIMS,
            "Trying to access the dimension of a tensor, when its maximal shape cannot have that dimension.");
        if constexpr (n < 0)
        {
            return shape.d[shape.nbDims + n];
        }
        else
        {
            return shape.d[n];
        }
    }

    virtual void reshape(Shape const& dims) = 0;

    void resize(std::size_t newSize) override
    {
        if (newSize == getSize())
            return;

        reshape(makeShape({castSize(newSize)}));
    }

    ITensor(ITensor const&) = delete;

    ITensor& operator=(ITensor const&) = delete;

    static std::int64_t volume(Shape const& dims)
    {
        {
            return dims.nbDims < 0 ? -1
                : dims.nbDims == 0
                ? 0
                : std::accumulate(dims.d, dims.d + dims.nbDims, std::int64_t{1}, std::multiplies<>{});
        }
    }

    static std::size_t volumeNonNegative(Shape const& shape)
    {
        auto const vol = volume(shape);
        CHECK_WITH_INFO(0 <= vol, "Invalid tensor shape");
        return static_cast<std::size_t>(vol);
    }

    static Shape strides(Shape const& dims)
    {
        auto const nbDims = dims.nbDims;
        Shape strides{};
        strides.nbDims = nbDims;
        if (nbDims > 0)
        {
            strides.d[nbDims - 1] = 1;
        }
        for (int i = nbDims - 2; i >= 0; i--)
        {
            strides.d[i] = dims.d[i + 1] * strides.d[i + 1];
        }
        return strides;
    }

    static Shape squeeze(Shape const& shape, SizeType32 dim);

    static Shape unsqueeze(Shape const& shape, SizeType32 dim);

    void squeeze(SizeType32 dim)
    {
        reshape(squeeze(getShape(), dim));
    }

    void unsqueeze(SizeType32 dim)
    {
        reshape(unsqueeze(getShape(), dim));
    }

    static UniquePtr slice(SharedPtr tensor, std::size_t offset, std::size_t size);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset, std::size_t size)
    {
        return ITensor::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset, size);
    }

    static UniquePtr slice(SharedPtr tensor, std::size_t offset)
    {
        auto const dims = tensor->getShape();
        auto const size = (dims.nbDims > 0 ? dims.d[0] : 0) - offset;
        return ITensor::slice(std::move(tensor), offset, size);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::size_t offset)
    {
        return ITensor::slice(constPointerCast(std::forward<TConstPtr>(tensor)), offset);
    }

    static UniquePtr slice(SharedPtr tensor, Shape const& offsetDims, DimType64 size);

    static UniquePtr slice(SharedPtr tensor, std::initializer_list<DimType64> const& offsetDims, DimType64 size)
    {
        return slice(std::move(tensor), makeShape(offsetDims), size);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, Shape const& offsetDims, std::size_t size)
    {
        return slice(constPointerCast(std::forward<TConstPtr>(tensor)), offsetDims, size);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(
        TConstPtr&& tensor, std::initializer_list<DimType64> const& offsetDims, std::size_t size)
    {
        return slice(constPointerCast(std::forward<TConstPtr>(tensor)), offsetDims, size);
    }

    static UniquePtr slice(SharedPtr tensor, Shape const& offsetDims)
    {
        auto const dims = tensor->getShape();
        auto const nbDims = offsetDims.nbDims;
        auto const size = (dims.nbDims > 0 && nbDims > 0) ? dims.d[nbDims - 1] - offsetDims.d[nbDims - 1] : 0;
        return ITensor::slice(std::move(tensor), offsetDims, size);
    }

    static UniquePtr slice(SharedPtr tensor, std::initializer_list<DimType64> const& offsetDims)
    {
        return slice(std::move(tensor), makeShape(offsetDims));
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, Shape const& offsetDims)
    {
        return slice(constPointerCast(std::forward<TConstPtr>(tensor)), offsetDims);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr slice(TConstPtr&& tensor, std::initializer_list<DimType64> const& offsetDims)
    {
        return slice(constPointerCast(std::forward<TConstPtr>(tensor)), offsetDims);
    }

    static UniquePtr at(SharedPtr tensor, Shape const& offsetDims)
    {
        auto result = slice(std::move(tensor), offsetDims, 1);
        if (result->getShape().nbDims > 1)
        {
            result->squeeze(0);
        }
        return result;
    }

    static UniquePtr at(SharedPtr tensor, std::initializer_list<DimType64> const& offsetDims)
    {
        return at(std::move(tensor), makeShape(offsetDims));
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr at(TConstPtr&& tensor, Shape const& offsetDims)
    {
        return at(constPointerCast(std::forward<TConstPtr>(tensor)), offsetDims);
    }

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static ITensor::UniqueConstPtr at(TConstPtr&& tensor, std::initializer_list<DimType64> const& offsetDims)
    {
        return at(constPointerCast(std::forward<TConstPtr>(tensor)), offsetDims);
    }

    static UniquePtr view(IBuffer::SharedPtr buffer, Shape const& dims);

    template <typename TConstPtr, std::enable_if_t<std::is_const_v<PointerElementType<TConstPtr>>, int> = 0>
    static UniqueConstPtr view(TConstPtr&& tensor, Shape const& dims)
    {
        return ITensor::view(constPointerCast(std::forward<TConstPtr>(tensor)), dims);
    }

    static UniquePtr view(SharedPtr tensor)
    {
        auto shapes = tensor->getShape();
        return ITensor::view(std::move(tensor), shapes);
    }

    static UniquePtr flattenN(SharedPtr tensor, std::int64_t sliceN = -1)
    {
        UniquePtr flatten = ITensor::view(tensor, ITensor::makeShape({ITensor::volume(tensor->getShape()), 1}));
        if (sliceN > 0)
        {
            flatten = ITensor::slice(std::move(flatten), 0, sliceN);
        }
        return flatten;
    }

    static UniquePtr wrap(void* data, nvinfer1::DataType type, Shape const& shape, std::size_t capacity);

    static UniquePtr wrap(void* data, nvinfer1::DataType type, Shape const& shape)
    {
        return wrap(data, type, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(T* data, Shape const& shape, std::size_t capacity)
    {
        return wrap(data, TRTDataType<T>::value, shape, capacity);
    }

    template <typename T>
    static UniquePtr wrap(T* data, Shape const& shape)
    {
        return wrap<T>(data, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(std::vector<T>& v, Shape const& shape)
    {
        return wrap<T>(v.data(), shape, v.capacity());
    }

    static Shape makeShape(std::initializer_list<DimType64> const& dims);

    static std::string toString(Shape const& dims);

    static bool shapeEquals(Shape const& lhs, Shape const& rhs)
    {
        return shapeEquals(lhs, rhs.d, rhs.nbDims);
    }

    template <typename T>
    static bool shapeEquals(Shape const& lhs, T const* dims, SizeType32 count)
    {
        return lhs.nbDims == count && std::equal(lhs.d, lhs.d + lhs.nbDims, dims);
    }

    [[nodiscard]] bool shapeEquals(Shape const& other) const
    {
        return shapeEquals(getShape(), other);
    }

    [[nodiscard]] bool shapeEquals(std::initializer_list<SizeType32> const& other) const
    {
        return shapeEquals(getShape(), other.begin(), other.size());
    }

    template <typename T>
    bool shapeEquals(T const* dims, SizeType32 count) const
    {
        return shapeEquals(getShape(), dims, count);
    }

protected:
    ITensor() = default;

    static DimType64 castSize(size_t newSize)
    {
        CHECK_WITH_INFO(
            newSize <= std::numeric_limits<DimType64>::max(), "New size is too large. Use reshape() instead.");
        return static_cast<DimType64>(newSize);
    }
};

inline std::ostream& operator<<(std::ostream& output, ITensor::Shape const& dims)
{
    return output << ITensor::toString(dims);
}

std::ostream& operator<<(std::ostream& output, ITensor const& tensor);

template <typename T>
T const* bufferCastOrNull(ITensor::SharedConstPtr const& tensorPtr)
{
    return bufferCastOrNull<T>(static_cast<IBuffer::SharedConstPtr>(tensorPtr));
}

template <typename T>
T* bufferCastOrNull(ITensor::SharedPtr const& tensorPtr)
{
    return bufferCastOrNull<T>(static_cast<IBuffer::SharedPtr>(tensorPtr));
}

template <typename T>
T* bufferCastOrNull(std::optional<ITensor::SharedPtr> const& optionalTensorPtr)
{
    return bufferCastOrNull<T>(static_cast<std::optional<IBuffer::SharedPtr>>(optionalTensorPtr));
}

template <typename T>
T const* bufferCastOrNull(std::optional<ITensor::SharedConstPtr> const& optionalTensorPtr)
{
    return bufferCastOrNull<T>(static_cast<std::optional<IBuffer::SharedConstPtr>>(optionalTensorPtr));
}

}
