#include "layernormQuantizationPlugin.h"
#include "pluginUtils.h"
#include "../src/layernormKernels.h"

using namespace nvinfer1;
using namespace suggestify::kernels;
using namespace suggestify::common;
using suggestify::plugins::LayernormQuantizationPluginCreator;
using suggestify::plugins::LayernormQuantizationPlugin;

static char const* LAYERNORM_QUANTIZATION_PLUGIN_VERSION{"1"};
static char const* LAYERNORM_QUANTIZATION_PLUGIN_NAME{"LayernormQuantization"};
PluginFieldCollection LayernormQuantizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LayernormQuantizationPluginCreator::mPluginAttributes;

LayernormQuantizationPlugin::LayernormQuantizationPlugin(
    float eps, bool useDiffOfSquares, bool dynamicActivationScaling, nvinfer1::DataType type)
    : mEps(eps)
    , mUseDiffOfSquares(useDiffOfSquares)
    , mDynActScaling(dynamicActivationScaling)
    , mType(type)
{
}

LayernormQuantizationPlugin::LayernormQuantizationPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mEps);
    read(d, mUseDiffOfSquares);
    read(d, mDynActScaling);
    read(d, mType);
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* LayernormQuantizationPlugin::clone() const noexcept
{
    auto* plugin = new LayernormQuantizationPlugin(mEps, mUseDiffOfSquares, mDynActScaling, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs LayernormQuantizationPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[outputIndex];
    }

    try
    {
        CHECK(outputIndex == 1);
        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims - 1; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        ret.d[ret.nbDims - 1] = exprBuilder.constant(1);
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LayernormQuantizationPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    int const totalPoses = 6 + static_cast<int>(mDynActScaling);
    CHECK(0 <= pos && pos < totalPoses);
    CHECK(nbInputs == 4);
    if (pos < nbInputs)
    {
        switch (pos)
        {
        case 0:
        case 1:
        case 2: return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        case 3: return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    if (pos == 4)
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void LayernormQuantizationPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t LayernormQuantizationPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int LayernormQuantizationPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    int64_t m64 = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m64 *= inputDesc[0].dims.d[i];
    }
    int const m = INT32_CAST(m64);
    int const n = INT32_CAST(inputDesc[1].dims.d[0]);

    float const* scale = reinterpret_cast<float const*>(inputs[3]);
    int8_t* output = reinterpret_cast<int8_t*>(outputs[0]);
    float* dynamic_scale = mDynActScaling ? reinterpret_cast<float*>(outputs[1]) : nullptr;

    if (mType == DataType::kHALF)
    {
        half const* input = reinterpret_cast<half const*>(inputs[0]);
        half const* weight = reinterpret_cast<half const*>(inputs[1]);
        half const* bias = reinterpret_cast<half const*>(inputs[2]);
        invokeGeneralLayerNorm(
            (half*) nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares, scale, dynamic_scale, output);
    }
    else if (mType == DataType::kFLOAT)
    {
        float const* input = reinterpret_cast<float const*>(inputs[0]);
        float const* weight = reinterpret_cast<float const*>(inputs[1]);
        float const* bias = reinterpret_cast<float const*>(inputs[2]);
        invokeGeneralLayerNorm(
            (float*) nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares, scale, dynamic_scale, output);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        __nv_bfloat16 const* input = reinterpret_cast<__nv_bfloat16 const*>(inputs[0]);
        __nv_bfloat16 const* weight = reinterpret_cast<__nv_bfloat16 const*>(inputs[1]);
        __nv_bfloat16 const* bias = reinterpret_cast<__nv_bfloat16 const*>(inputs[2]);
        invokeGeneralLayerNorm((__nv_bfloat16*) nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares,
            scale, dynamic_scale, output);
    }
#endif
    sync_check_cuda_error();
    return 0;
}

nvinfer1::DataType LayernormQuantizationPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert((mDynActScaling && index < 2) || (!mDynActScaling && index == 0));
    if (index == 0)
    {
        return nvinfer1::DataType::kINT8;
    }
    return nvinfer1::DataType::kFLOAT;
}


char const* LayernormQuantizationPlugin::getPluginType() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_NAME;
}

char const* LayernormQuantizationPlugin::getPluginVersion() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_VERSION;
}

int LayernormQuantizationPlugin::getNbOutputs() const noexcept
{
    return 1 + static_cast<int>(mDynActScaling);
}

int LayernormQuantizationPlugin::initialize() noexcept
{
    return 0;
}

void LayernormQuantizationPlugin::terminate() noexcept {}

size_t LayernormQuantizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mUseDiffOfSquares) + sizeof(mDynActScaling) + sizeof(mType);
}

void LayernormQuantizationPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mUseDiffOfSquares);
    write(d, mDynActScaling);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void LayernormQuantizationPlugin::destroy() noexcept
{
    delete this;
}


LayernormQuantizationPluginCreator::LayernormQuantizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mPluginAttributes.emplace_back(PluginField("use_diff_of_squares", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dyn_act_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* LayernormQuantizationPluginCreator::getPluginName() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_NAME;
}

char const* LayernormQuantizationPluginCreator::getPluginVersion() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_VERSION;
}

PluginFieldCollection const* LayernormQuantizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LayernormQuantizationPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    float eps;
    nvinfer1::DataType type;
    bool useDiffOfSquares;
    bool dynamicActivationScaling;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dyn_act_scaling"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            dynamicActivationScaling = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_diff_of_squares"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            useDiffOfSquares = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new LayernormQuantizationPlugin(eps, useDiffOfSquares, dynamicActivationScaling, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LayernormQuantizationPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new LayernormQuantizationPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
