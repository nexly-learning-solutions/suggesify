
#pragma once

#include "../executor/types.h"

#include "../common/arrayView.h"
#include "../common/assert.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace suggestify::runtime
{
class ITensor;
class CudaStream;
}

namespace suggestify::executor
{

class Tensor;

namespace detail
{
std::shared_ptr<runtime::ITensor> const& toITensor(Tensor const& tensor);
Tensor ofITensor(std::shared_ptr<runtime::ITensor> tensor);
using DimType64 = int64_t;

}

class Shape : public suggestify::common::ArrayView<detail::DimType64 const>
{
public:
    using Base = suggestify::common::ArrayView<detail::DimType64 const>;
    using DimType64 = typename std::remove_cv_t<Base::value_type>;

    Shape()
        : Base{nullptr, 0} {};

    Shape(DimType64 const* data, Base::size_type size)
        : Base{data, size}
    {
    }

    Shape(std::initializer_list<DimType64> dims)
        : Base{dims.begin(), dims.size()}
    {
    }
};

class Tensor
{
public:
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;

    static Tensor cpu(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor cpu(Shape shape = {})
    {
        return Tensor::cpu(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToCpu(Tensor::CudaStreamPtr stream = nullptr) const;

    static Tensor pinned(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor pinned(Shape shape = {})
    {
        return Tensor::pinned(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToPinned(Tensor::CudaStreamPtr stream = nullptr) const;

    static Tensor pooledPinned(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor pooledPinned(Shape shape = {})
    {
        return Tensor::pooledPinned(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToPooledPinned(Tensor::CudaStreamPtr stream = nullptr) const;

    static Tensor managed(DataType dataType, Shape shape = {});

    template <typename T>
    static Tensor managed(Shape shape = {})
    {
        return Tensor::managed(getRuntimeType<T>(), shape);
    }

    [[nodiscard]] Tensor copyToManaged(Tensor::CudaStreamPtr stream = nullptr) const;

    static Tensor gpu(DataType dataType, CudaStreamPtr stream, Shape shape = {});

    template <typename T>
    static Tensor gpu(CudaStreamPtr stream, Shape shape = {})
    {
        return Tensor::gpu(getRuntimeType<T>(), std::move(stream), shape);
    }

    [[nodiscard]] Tensor copyToGpu(Tensor::CudaStreamPtr stream) const;

    static Tensor of(DataType dataType, void* data, Shape shape);

    template <typename T>
    static Tensor of(T* data, Shape shape)
    {
        return of(getRuntimeType<T>(), static_cast<void*>(data), shape);
    }

    template <typename T>
    static Tensor of(T& data)
    {
        using DimType64 = Shape::DimType64;
        if constexpr (!std::is_same_v<DimType64, decltype(data.size())>)
        {
            TLLM_CHECK(data.size() <= std::numeric_limits<DimType64>::max());
        }
        return of(data.data(), {static_cast<Shape::DimType64>(data.size())});
    }

    Tensor() noexcept = default;

    ~Tensor() = default;

    Tensor(Tensor const& other) noexcept = default;

    Tensor(Tensor&& other) noexcept = default;

    Tensor& operator=(Tensor const& other) noexcept = default;

    Tensor& operator=(Tensor&& other) noexcept = default;

    [[nodiscard]] void* getData();

    [[nodiscard]] void const* getData() const;

    [[nodiscard]] DataType getDataType() const;

    [[nodiscard]] MemoryType getMemoryType() const;

    [[nodiscard]] Shape getShape() const;

    [[nodiscard]] std::size_t getSize() const;

    [[nodiscard]] std::size_t getSizeInBytes() const;

    void setZero(CudaStreamPtr stream = nullptr);

    void setFrom(Tensor const& other, CudaStreamPtr stream = nullptr);

    explicit operator bool() const
    {
        return static_cast<bool>(mTensor);
    }

    bool operator==(Tensor const& rhs) const
    {
        return mTensor == rhs.mTensor;
    }

    bool operator!=(Tensor const& rhs) const
    {
        return !(rhs == *this);
    }

private:
    using Impl = runtime::ITensor;
    explicit Tensor(std::shared_ptr<runtime::ITensor> tensor);

    template <typename T>
    static DataType getRuntimeType()
    {
        return TypeTraits<std::remove_cv_t<T>>::value;
    }

    [[nodiscard]] Tensor copyTo(std::shared_ptr<Impl> tensor, CudaStreamPtr stream) const;

    std::shared_ptr<Impl> mTensor;

    friend std::shared_ptr<runtime::ITensor> const& detail::toITensor(Tensor const& tensor);
    friend Tensor detail::ofITensor(std::shared_ptr<runtime::ITensor> tensor);
    friend class Serialization;
};

}
