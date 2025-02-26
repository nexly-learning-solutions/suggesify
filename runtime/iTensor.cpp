
#include "iTensor.h"

#include "suggestify/common/memoryUtils.h"
#include "suggestify/common/stringUtils.h"
#include "bufferManager.h"
#include "tensorView.h"
#include "tllmBuffers.h"

#include <initializer_list>
#include <memory>

using namespace suggestify::runtime;

namespace tc = suggestify::common;

ITensor::UniquePtr ITensor::slice(SharedPtr tensor, std::size_t offset, std::size_t size)
{
    CHECK(tensor);
    return std::make_unique<TensorView>(std::move(tensor), offset, size);
}

ITensor::UniquePtr ITensor::slice(SharedPtr tensor, Shape const& offsetDims, ITensor::DimType64 size)
{
    auto shape = tensor->getShape();
    CHECK(offsetDims.nbDims >= 0);
    CHECK(shape.nbDims >= offsetDims.nbDims);
    CHECK(size >= 0);

    Shape strides = ITensor::strides(shape);
    DimType64 offset{0};
    for (SizeType32 di = 0; di < offsetDims.nbDims - 1; di++)
    {
        CHECK(0 <= offsetDims.d[di] && offsetDims.d[di] < shape.d[di]);
        offset += offsetDims.d[di] * strides.d[di];
    }

    if (LIKELY(offsetDims.nbDims > 0))
    {
        CHECK(offsetDims.d[offsetDims.nbDims - 1] + size <= shape.d[offsetDims.nbDims - 1]);
        offset += offsetDims.d[offsetDims.nbDims - 1] * strides.d[offsetDims.nbDims - 1];
    }
    else
    {
        CHECK(size >= 0 && size <= 1);
        CHECK(shape.nbDims == 0 ? size == 0 : true);
    }

    Shape dims;
    dims.nbDims = shape.nbDims - offsetDims.nbDims + 1;
    dims.d[0] = size;
    for (SizeType32 di = 1; di < dims.nbDims; di++)
    {
        dims.d[di] = shape.d[di - 1 + offsetDims.nbDims];
    }

    return std::make_unique<TensorView>(std::move(tensor), offset, volume(dims), dims);
}

ITensor::UniquePtr ITensor::view(IBuffer::SharedPtr buffer, nvinfer1::Dims const& dims)
{
    auto const size = buffer->getSize();
    return std::make_unique<TensorView>(std::move(buffer), 0, size, dims);
}

nvinfer1::Dims ITensor::makeShape(std::initializer_list<ITensor::DimType64> const& dims)
{
    CHECK_WITH_INFO(dims.size() <= nvinfer1::Dims::MAX_DIMS, "Number of dimensions is too large");
    nvinfer1::Dims shape{};
    shape.nbDims = static_cast<decltype(Shape::nbDims)>(dims.size());
    std::copy(dims.begin(), dims.end(), shape.d);
    return shape;
}

std::string ITensor::toString(nvinfer1::Dims const& dims)
{
    if (dims.nbDims < 0)
    {
        return "invalid";
    }
    else if (dims.nbDims == 0)
    {
        return "()";
    }
    else
    {
        return tc::arr2str(dims.d, dims.nbDims);
    }
}

ITensor::UniquePtr ITensor::wrap(void* data, nvinfer1::DataType type, nvinfer1::Dims const& shape, std::size_t capacity)
{
    auto const size = volumeNonNegative(shape);
    CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
    auto memoryType = IBuffer::memoryType(data);

    ITensor::UniquePtr result;
    auto const capacityInBytes = capacity * BufferDataType(type).getSize();
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericTensor<PinnedBorrowingAllocator>(
            shape, capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kPINNEDPOOL:
        result.reset(new GenericTensor<PinnedPoolBorrowingAllocator>(
            shape, capacity, type, PinnedPoolBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset(
            new GenericTensor<CpuBorrowingAllocator>(
                shape, capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset(
            new GenericTensor<GpuBorrowingAllocator>(
                shape, capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kUVM:
        result.reset(
            new GenericTensor<ManagedBorrowingAllocator>(
                shape, capacity, type, ManagedBorrowingAllocator(data, capacityInBytes)));
        break;
    default: THROW("Invalid memory type."); break;
    }
    return result;
}

ITensor::Shape ITensor::squeeze(Shape const& shape, SizeType32 dim)
{
    CHECK_WITH_INFO(0 < shape.nbDims, "Cannot squeeze 1-dimensional tensor");
    CHECK_WITH_INFO(
        dim < shape.nbDims, tc::fmtstr("Invalid index %d, tensor has %d dimensions", dim, shape.nbDims));
    CHECK_WITH_INFO(shape.d[dim] == 1, "Can only squeeze dimension of size 1");

    Shape newDims{shape.nbDims - 1};
    std::copy(shape.d, shape.d + dim, newDims.d);
    std::copy(shape.d + dim + 1, shape.d + shape.nbDims, newDims.d + dim);
    return newDims;
}

ITensor::Shape ITensor::unsqueeze(Shape const& shape, SizeType32 dim)
{
    CHECK_WITH_INFO(shape.nbDims < Shape::MAX_DIMS, "Too many dimensions to unsqueeze");
    CHECK_WITH_INFO(
        0 <= dim && dim <= shape.nbDims, common::fmtstr("Invalid dim %d, tensor has %d dimensions", dim, shape.nbDims));

    Shape newDims{shape.nbDims + 1};
    std::copy(shape.d, shape.d + dim, newDims.d);
    newDims.d[dim] = 1;
    std::copy(shape.d + dim, shape.d + shape.nbDims, newDims.d + dim + 1);
    return newDims;
}

namespace
{
template <typename T>
void printTensor(ITensor const& tensor, std::ostream& out)
{
    CHECK_WITH_INFO(tensor.getDataType() == TRTDataType<typename std::remove_cv<T>::type>::value,
        tc::fmtstr("Data type mismatch: %d vs %d", static_cast<std::int32_t>(tensor.getDataType()),
            static_cast<std::int32_t>(TRTDataType<typename std::remove_cv<T>::type>::value)));
    auto const& shape = tensor.getShape();
    out << "shape: " << shape << std::endl;
    out << "vals: " << std::endl;

    BufferManager::ITensorPtr host{};
    T const* hostData;
    if (tensor.getMemoryType() == MemoryType::kGPU)
    {
        auto streamPtr = std::make_shared<CudaStream>();
        BufferManager manager{streamPtr};
        host = manager.copyFrom(tensor, MemoryType::kCPU);
        streamPtr->synchronize();
        hostData = bufferCast<T>(*host);
    }
    else
    {
        hostData = bufferCast<T>(tensor);
    }

    using TOutput
        = std::conditional_t<std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>, std::int32_t, T>;
    if (shape.nbDims > 3)
    {
        out << "Not printing elements for more than 3 dims\n";
    }
    else if (shape.nbDims == 3 && shape.d[2] > 1)
    {
        for (int i = 0; i < shape.d[0]; ++i)
        {
            for (int j = 0; j < shape.d[1]; ++j)
            {
                out << "i=" << i << " j=" << j << ": ";
                tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, j, 0), shape.d[2]) << "\n";
            }
        }
    }
    else if (shape.nbDims >= 2 && shape.d[1] > 1)
    {
        for (int i = 0; i < shape.d[0]; ++i)
        {
            out << "i=" << i << ": ";
            tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, 0), shape.d[1]) << "\n";
        }
    }
    else
    {
        tc::arr2outCasted<TOutput>(out, hostData, shape.d[0]) << "\n";
    }
    out << std::flush;
}

}

std::ostream& suggestify::runtime::operator<<(std::ostream& out, ITensor const& tensor)
{
    switch (tensor.getDataType())
    {
    case nvinfer1::DataType::kFLOAT: printTensor<float>(tensor, out); break;
    case nvinfer1::DataType::kHALF: printTensor<half>(tensor, out); break;
    case nvinfer1::DataType::kBOOL: printTensor<bool>(tensor, out); break;
    case nvinfer1::DataType::kINT8: printTensor<std::int8_t>(tensor, out); break;
    case nvinfer1::DataType::kINT32: printTensor<std::int32_t>(tensor, out); break;
    case nvinfer1::DataType::kINT64: printTensor<std::int64_t>(tensor, out); break;
    case nvinfer1::DataType::kUINT8: printTensor<std::uint8_t>(tensor, out); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: printTensor<__nv_bfloat16>(tensor, out); break;
#endif
    default: THROW("Unsupported data type");
    }

    return out;
}
