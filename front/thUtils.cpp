
#include "thUtils.h"
#include <NvInferRuntime.h>
#include <array>

namespace torch_ext
{

suggestify::runtime::ITensor::Shape convert_shape(torch::Tensor tensor)
{
    constexpr auto trtMaxDims = nvinfer1::Dims::MAX_DIMS;
    auto const torchTensorNumDims = tensor.dim();
    CHECK_WITH_INFO(torchTensorNumDims <= trtMaxDims,
        "TensorRT supports at most %i tensor dimensions. Found a Torch tensor with %li dimensions.", trtMaxDims,
        torchTensorNumDims);
    auto result = nvinfer1::Dims{};
    result.nbDims = static_cast<int32_t>(torchTensorNumDims);
    for (int i = 0; i < torchTensorNumDims; i++)
    {
        result.d[i] = static_cast<int64_t>(tensor.size(i));
    }
    return result;
}

template <typename T>
suggestify::runtime::ITensor::UniquePtr convert_tensor(torch::Tensor tensor)
{
    return suggestify::runtime::ITensor::wrap(
        get_ptr<T>(tensor), suggestify::runtime::TRTDataType<T>::value, convert_shape(tensor));
}

template suggestify::runtime::ITensor::UniquePtr convert_tensor<int32_t*>(torch::Tensor tensor);
template suggestify::runtime::ITensor::UniquePtr convert_tensor<int32_t>(torch::Tensor tensor);
template suggestify::runtime::ITensor::UniquePtr convert_tensor<uint8_t>(torch::Tensor tensor);
template suggestify::runtime::ITensor::UniquePtr convert_tensor<int8_t>(torch::Tensor tensor);
template suggestify::runtime::ITensor::UniquePtr convert_tensor<float>(torch::Tensor tensor);
template suggestify::runtime::ITensor::UniquePtr convert_tensor<half>(torch::Tensor tensor);
#ifdef ENABLE_BF16
template suggestify::runtime::ITensor::UniquePtr convert_tensor<__nv_bfloat16>(torch::Tensor tensor);
#endif
template suggestify::runtime::ITensor::UniquePtr convert_tensor<bool>(torch::Tensor tensor);

std::optional<float> getFloatEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    try
    {
        float value = std::stof(env);
        return {value};
    }
    catch (std::invalid_argument const& e)
    {
        return std::nullopt;
    }
    catch (std::out_of_range const& e)
    {
        return std::nullopt;
    }
};

int nextPowerOfTwo(int v)
{
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}
}
