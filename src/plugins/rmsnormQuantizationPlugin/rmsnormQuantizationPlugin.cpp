#include "rmsnormQuantizationPlugin.h"
#include "pluginUtils.h"
#include "../src/rmsnormKernels.h"

using namespace nvinfer1;
using namespace suggestify::kernels;
using namespace suggestify::common;
using suggestify::plugins::RmsnormQuantizationPluginCreator;
using suggestify::plugins::RmsnormQuantizationPlugin;

static char const* RMSNORM_QUANTIZATION_PLUGIN_VERSION{"1"};
static char const* RMSNORM_QUANTIZATION_PLUGIN_NAME{"RmsnormQuantization"};
PluginFieldCollection RmsnormQuantizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RmsnormQuantizationPluginCreator::mPluginAttributes;

RmsnormQuantizationPlugin::RmsnormQuantizationPlugin(float eps, bool dynamicActivationScaling, bool sumPerToken,
    bool clampValEnabled, QuantMode quantMode, nvinfer1::DataType type, nvinfer1::DataType outputType)
    : mEps(eps)
    , mDynActScaling(dynamicActivationScaling)
    , mSumPerToken(sumPerToken)
    , mClampValEnabled{clampValEnabled}
    , mQuantMode{quantMode}
    , mType(type)
    , mOutputType{outputType}
{
    CHECK_WITH_INFO(mOutputType == nvinfer1::DataType::kINT8 || mOutputType == nvinfer1::DataType::kFP8,
        "Only int8 or fp8 output type is allowed.");
    CHECK_WITH_INFO(mQuantMode.hasPerTokenScaling(), "The quant mode is not valid.");
}

RmsnormQuantizationPlugin::RmsnormQuantizationPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mEps);
    read(d, mDynActScaling);
    read(d, mSumPerToken);
    read(d, mClampValEnabled);
    read(d, mQuantMode);
    read(d, mType);
    read(d, mOutputType);
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different nexly version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* RmsnormQuantizationPlugin::clone() const noexcept
{
    auto* plugin = new RmsnormQuantizationPlugin(
        mEps, mDynActScaling, mSumPerToken, mClampValEnabled, mQuantMode, mType, mOutputType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RmsnormQuantizationPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[outputIndex];
    }

    try
    {
        if (outputIndex == 1)
        {
            CHECK(mDynActScaling);
        }
        else if (outputIndex == 2)
        {
            CHECK(mSumPerToken);
        }
        else
        {
            CHECK(false);
        }

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

bool RmsnormQuantizationPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    int const totalPoses
        = 6 + static_cast<int>(mClampValEnabled) + static_cast<int>(mDynActScaling) + static_cast<int>(mSumPerToken);
    CHECK(0 <= pos && pos < totalPoses);
    CHECK(nbInputs == 4 + static_cast<int>(mClampValEnabled));
    if (pos < nbInputs)
    {
        if (pos < 3)
        {
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
        else if (pos == 3)
        {
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
        else if (pos == 4 && mClampValEnabled)
        {
            return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }
    }
    else if (pos == 4 + int(mClampValEnabled))
    {
        return (inOut[pos].type == mOutputType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == 5 + int(mClampValEnabled))
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == 6 + int(mClampValEnabled))
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }

    CHECK_WITH_INFO(false, "The input/output is not supported.");
    return false;
}

void RmsnormQuantizationPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t RmsnormQuantizationPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

template <typename T, typename QuantT>
void RmsnormQuantizationPlugin::dispatchDataType(void* out, void const* input, void const* gamma, void const* beta,
    float const eps, int const tokens, int const hidden_dim, cudaStream_t stream, void const* clampValPtr,
    void const* scale, void* dynamic_scale, void* sum_per_token, void* normed_output_quant) noexcept
{

    invokeGeneralRmsNorm(reinterpret_cast<T*>(out), reinterpret_cast<T const*>(input),
        reinterpret_cast<T const*>(gamma), reinterpret_cast<T const*>(beta), eps, tokens, hidden_dim, mQuantMode,
        stream, reinterpret_cast<float const*>(clampValPtr), reinterpret_cast<float const*>(scale),
        reinterpret_cast<float*>(dynamic_scale), reinterpret_cast<float*>(sum_per_token),
        reinterpret_cast<QuantT*>(normed_output_quant));
}

int RmsnormQuantizationPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
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

    void const* input = inputs[0];
    void const* weight = inputs[1];
    void const* bias = inputs[2];
    void const* scale = inputs[3];
    void const* clampValPtr = mClampValEnabled ? inputs[4] : nullptr;
    void* output = outputs[0];
    void* dynamic_scale = mDynActScaling ? outputs[1] : nullptr;
    void* sum_per_token = mSumPerToken ? outputs[2] : nullptr;

    if (inputDesc[0].type == DataType::kFLOAT && mOutputType == DataType::kINT8)
    {
        dispatchDataType<float, int8_t>(
            nullptr, input, weight, bias, mEps, m, n, stream, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kFLOAT && mOutputType == DataType::kFP8)
    {
        dispatchDataType<float, __nv_fp8_e4m3>(
            nullptr, input, weight, bias, mEps, m, n, stream, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#endif
    else if (inputDesc[0].type == DataType::kHALF && mOutputType == DataType::kINT8)
    {
        dispatchDataType<half, int8_t>(
            nullptr, input, weight, bias, mEps, m, n, stream, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kHALF && mOutputType == DataType::kFP8)
    {
        dispatchDataType<half, __nv_fp8_e4m3>(
            nullptr, input, weight, bias, mEps, m, n, stream, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#endif
#ifdef ENABLE_BF16
    else if (inputDesc[0].type == DataType::kBF16 && mOutputType == DataType::kINT8)
    {
        dispatchDataType<__nv_bfloat16, int8_t>(
            nullptr, input, weight, bias, mEps, m, n, stream, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kBF16 && mOutputType == DataType::kFP8)
    {
        dispatchDataType<__nv_bfloat16, __nv_fp8_e4m3>(
            nullptr, input, weight, bias, mEps, m, n, stream, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#endif
#endif
    sync_check_cuda_error();
    return 0;
}

nvinfer1::DataType RmsnormQuantizationPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index <= 2);

    if (index == 0)
    {
        return mOutputType;
    }
    if (index == 1)
    {
        assert(mDynActScaling);
        return nvinfer1::DataType::kFLOAT;
    }
    {
        assert(mDynActScaling && mSumPerToken);
        return nvinfer1::DataType::kFLOAT;
    }
}


char const* RmsnormQuantizationPlugin::getPluginType() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_NAME;
}

char const* RmsnormQuantizationPlugin::getPluginVersion() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_VERSION;
}

int RmsnormQuantizationPlugin::getNbOutputs() const noexcept
{
    return 1 + static_cast<int>(mDynActScaling) + static_cast<int>(mSumPerToken);
}

int RmsnormQuantizationPlugin::initialize() noexcept
{
    return 0;
}

void RmsnormQuantizationPlugin::terminate() noexcept {}

size_t RmsnormQuantizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mOutputType) + sizeof(mClampValEnabled) + sizeof(mEps) + sizeof(mDynActScaling) + sizeof(mSumPerToken)
        + sizeof(mType) + sizeof(mQuantMode);
}

void RmsnormQuantizationPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mDynActScaling);
    write(d, mSumPerToken);
    write(d, mClampValEnabled);
    write(d, mQuantMode);
    write(d, mType);
    write(d, mOutputType);
    assert(d == a + getSerializationSize());
}

void RmsnormQuantizationPlugin::destroy() noexcept
{
    delete this;
}


RmsnormQuantizationPluginCreator::RmsnormQuantizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mPluginAttributes.emplace_back(PluginField("dyn_act_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sum_per_token", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clamp_enabled", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quant_mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("out_type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* RmsnormQuantizationPluginCreator::getPluginName() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_NAME;
}

char const* RmsnormQuantizationPluginCreator::getPluginVersion() const noexcept
{
    return RMSNORM_QUANTIZATION_PLUGIN_VERSION;
}

PluginFieldCollection const* RmsnormQuantizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RmsnormQuantizationPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType outputType;
    QuantMode quantMode;
    bool clampValEnabled = false;
    float eps;
    nvinfer1::DataType type;
    bool dynamicActivationScaling;
    bool sumPerToken;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "quant_mode"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            quantMode = QuantMode(*(static_cast<int32_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "out_type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            outputType = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "clamp_enabled"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            clampValEnabled = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "eps"))
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
        else if (!strcmp(attrName, "sum_per_token"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            sumPerToken = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new RmsnormQuantizationPlugin(
            eps, dynamicActivationScaling, sumPerToken, clampValEnabled, quantMode, type, outputType);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RmsnormQuantizationPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new RmsnormQuantizationPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
