#include "quantizeTensorPlugin.h"
#include "../src/quantization.h"

using namespace nvinfer1;
using namespace suggestify::kernels;
using suggestify::plugins::QuantizeTensorPluginCreator;
using suggestify::plugins::QuantizeTensorPlugin;

static char const* QUANTIZE_TENSOR_PLUGIN_VERSION{"1"};
static char const* QUANTIZE_TENSOR_PLUGIN_NAME{"QuantizeTensor"};
PluginFieldCollection QuantizeTensorPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QuantizeTensorPluginCreator::mPluginAttributes;

QuantizeTensorPlugin::QuantizeTensorPlugin() {}

QuantizeTensorPlugin::QuantizeTensorPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* QuantizeTensorPlugin::clone() const noexcept
{
    return new QuantizeTensorPlugin(*this);
}

nvinfer1::DimsExprs QuantizeTensorPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        CHECK(nbInputs == 2);
        CHECK(outputIndex < 1);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool QuantizeTensorPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF
#ifdef ENABLE_BF16
                   || inOut[pos].type == nvinfer1::DataType::kBF16
#endif
                   )
            && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        CHECK(false);
        return false;
    }
}

void QuantizeTensorPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t QuantizeTensorPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int QuantizeTensorPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    int64_t numElts = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims; ++ii)
    {
        numElts *= inputDesc[0].dims.d[ii];
    }

    if (inputDesc[0].type == DataType::kFLOAT)
    {
        invokeQuantization<float>(reinterpret_cast<int8_t*>(outputs[0]), reinterpret_cast<float const*>(inputs[0]),
            numElts, reinterpret_cast<float const*>(inputs[1]), stream, mProp.maxGridSize[0]);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        invokeQuantization<half>(reinterpret_cast<int8_t*>(outputs[0]), reinterpret_cast<half const*>(inputs[0]),
            numElts, reinterpret_cast<float const*>(inputs[1]), stream, mProp.maxGridSize[0]);
    }
#ifdef ENABLE_BF16
    else if (inputDesc[0].type == DataType::kBF16)
    {
        invokeQuantization<__nv_bfloat16>(reinterpret_cast<int8_t*>(outputs[0]),
            reinterpret_cast<__nv_bfloat16 const*>(inputs[0]), numElts, reinterpret_cast<float const*>(inputs[1]),
            stream, mProp.maxGridSize[0]);
    }
#endif
    sync_check_cuda_error();
    return 0;
}

nvinfer1::DataType QuantizeTensorPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(nbInputs == 2);
    CHECK(index == 0);
    return nvinfer1::DataType::kINT8;
}


char const* QuantizeTensorPlugin::getPluginType() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_NAME;
}

char const* QuantizeTensorPlugin::getPluginVersion() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_VERSION;
}

int QuantizeTensorPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int QuantizeTensorPlugin::initialize() noexcept
{
    int deviceId = 0;
    suggestify::common::check_cuda_error(cudaGetDevice(&deviceId));
    suggestify::common::check_cuda_error(cudaGetDeviceProperties(&mProp, deviceId));
    return 0;
}

void QuantizeTensorPlugin::terminate() noexcept {}

size_t QuantizeTensorPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void QuantizeTensorPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    assert(d == a + getSerializationSize());
}

void QuantizeTensorPlugin::destroy() noexcept
{
    delete this;
}


QuantizeTensorPluginCreator::QuantizeTensorPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QuantizeTensorPluginCreator::getPluginName() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_NAME;
}

char const* QuantizeTensorPluginCreator::getPluginVersion() const noexcept
{
    return QUANTIZE_TENSOR_PLUGIN_VERSION;
}

PluginFieldCollection const* QuantizeTensorPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QuantizeTensorPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto* obj = new QuantizeTensorPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QuantizeTensorPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new QuantizeTensorPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
