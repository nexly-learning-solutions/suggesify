
#include "gemmSwigluPlugin.h"
#include "cutlass_extensions/gemm_configs.h"

#include <NvInferRuntimeBase.h>
#include <numeric>

using namespace nvinfer1;
using namespace suggestify::common;
using namespace suggestify::kernels::cutlass_kernels;
using suggestify::plugins::GemmSwigluPluginCreator;
using suggestify::plugins::GemmSwigluPlugin;
using suggestify::plugins::GemmSwigluPluginProfiler;
using suggestify::plugins::read;
using suggestify::plugins::write;

static char const* GEMM_SWIGLU_PLUGIN_VERSION{"1"};
static char const* GEMM_SWIGLU_PLUGIN_NAME{"GemmSwiglu"};
PluginFieldCollection GemmSwigluPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GemmSwigluPluginCreator::mPluginAttributes;

size_t GemmSwigluPluginProfiler::getBytePerElement(nvinfer1::DataType type)
{
    size_t bpe;
    if (type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kBF16)
    {
        bpe = 2;
    }
    else if (type == nvinfer1::DataType::kINT8 || type == nvinfer1::DataType::kFP8)
    {
        bpe = 1;
    }
    else
    {
        TLLM_THROW("Not recognized/implemented");
    }
    return bpe;
}

void GemmSwigluPluginProfiler::setQuantMode(suggestify::common::QuantMode const& quantMode)
{
    mQuantMode = quantMode;
}

void GemmSwigluPluginProfiler::runTactic(
    int m, int n, int k, GemmSwigluPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    size_t bpe = getBytePerElement(mType);

    size_t wsSizeRunner = mRunner->getWorkspaceSize(m, n, k);

    size_t wsByteOffset = 0;
    int8_t* wsBytePointer = reinterpret_cast<int8_t*>(workspace);
    void* aTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * k * bpe));
    void* bTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, n * k * bpe));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, 1 * n * bpe));
    void* dTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * (n / 2) * bpe));
    char* workspaceTmp = reinterpret_cast<char*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, wsSizeRunner));

    mRunner->gemm(
        dTmp, aTmp, bTmp, cTmp, mQuantMode, m, n, k, 1.0, 1.0, 1.0, tactic, workspaceTmp, wsSizeRunner, stream);
}

int GemmSwigluPluginProfiler::getMaxProfileM() const
{
    return 32768;
}

void GemmSwigluPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    std::vector<size_t> workspaces = {
        maxM * k * getBytePerElement(mType),
        n * k * getBytePerElement(mType),
        1 * n * getBytePerElement(mType),
        maxM * (n / 2) * getBytePerElement(mType),
        mRunner->getWorkspaceSize(maxM, n, k)
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<GemmSwigluPluginProfiler::Config> GemmSwigluPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

GemmSwigluPlugin::GemmSwigluPlugin(QuantMode quantMode, nvinfer1::DataType type, bool hasBias, float scale_d0,
    float scale_d1, float scale_output, GemmSwigluPlugin::PluginProfilerPtr const& pluginProfiler)
    : mQuantMode(quantMode)
    , mHasBias(hasBias)
    , mScaleD0(scale_d0)
    , mScaleD1(scale_d1)
    , mScaleOutput(scale_output)
    , mPluginProfiler(pluginProfiler)
{
    init(type);
}

GemmSwigluPlugin::GemmSwigluPlugin(
    void const* data, size_t length, GemmSwigluPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    unsigned int quantMode;
    read(d, quantMode);
    read(d, type);
    read(d, mHasBias);
    read(d, mScaleD0);
    read(d, mScaleD1);
    read(d, mScaleOutput);
    read(d, mDims);

    mQuantMode = QuantMode(quantMode);

    init(type);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK(d == a + length);
}

void GemmSwigluPlugin::init(nvinfer1::DataType type)
{
    mType = type;
    if (mType == nvinfer1::DataType::kFP8)
    {
        mGemmRunner = std::make_shared<CutlassFusedGatedGemmRunner<__nv_fp8_e4m3>>();
    }
    else
    {
        TLLM_THROW("Gemm Swiglu plugin only supports fp8 now");
    }

    mPluginProfiler->setQuantMode(mQuantMode);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::IPluginV2DynamicExt* GemmSwigluPlugin::clone() const noexcept
{
    auto* plugin = new GemmSwigluPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs GemmSwigluPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 3);
        TLLM_CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue() / 2);
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool GemmSwigluPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    case 3:
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        TLLM_CHECK(false);
        return false;
    }
}

void GemmSwigluPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];
    int const maxN = in[1].max.d[1];
    int const minK = in[0].min.d[in[0].min.nbDims - 1];
    int const minN = in[1].min.d[1];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mType};

    mWorkspaceMaxSize = mGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t GemmSwigluPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mWorkspaceMaxSize;
}

int GemmSwigluPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    int const n = inputDesc[1].dims.d[1];
    int const k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    size_t const wsSize = mGemmRunner->getWorkspaceSize(m, n, k);

    auto const bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid GEMM tactic");
    mGemmRunner->gemm(outputs[0], inputs[0], inputs[1], inputs[2], mQuantMode, m, n, k, mScaleD0, mScaleD1,
        mScaleOutput, *bestTactic, reinterpret_cast<char*>(workspace), wsSize, stream);

    return 0;
}

nvinfer1::DataType GemmSwigluPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}


char const* GemmSwigluPlugin::getPluginType() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_NAME;
}

char const* GemmSwigluPlugin::getPluginVersion() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_VERSION;
}

int GemmSwigluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int GemmSwigluPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void GemmSwigluPlugin::terminate() noexcept {}

size_t GemmSwigluPlugin::getSerializationSize() const noexcept
{
    return sizeof(unsigned int) +
        sizeof(nvinfer1::DataType) +
        sizeof(bool) +
        sizeof(float) * 3 +
        sizeof(mDims) +
        mPluginProfiler->getSerializationSize(mGemmId);
}

void GemmSwigluPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mHasBias);
    write(d, mScaleD0);
    write(d, mScaleD1);
    write(d, mScaleOutput);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    TLLM_CHECK(d == a + getSerializationSize());
}

void GemmSwigluPlugin::destroy() noexcept
{
    delete this;
}

void GemmSwigluPlugin::configGemm()
{
    mPluginProfiler->profileTactics(mGemmRunner, mType, mDims, mGemmId);
}


GemmSwigluPluginCreator::GemmSwigluPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_bias", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("scale_d0", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(PluginField("scale_d1", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(PluginField("scale_output", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GemmSwigluPluginCreator::getPluginName() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_NAME;
}

char const* GemmSwigluPluginCreator::getPluginVersion() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_VERSION;
}

PluginFieldCollection const* GemmSwigluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GemmSwigluPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    TLLM_CHECK(fc->nbFields == 5);
    nvinfer1::DataType type;
    bool hasBias;
    float scale_d0;
    float scale_d1;
    float scale_output;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "has_bias"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            hasBias = static_cast<bool>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "scale_d0"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            scale_d0 = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "scale_d1"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            scale_d1 = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "scale_output"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            scale_output = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
    }
    try
    {
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler( false);
        QuantMode quantMode = QuantMode::fromDescription();
        auto* obj = new GemmSwigluPlugin(quantMode, type, hasBias, scale_d0, scale_d1, scale_output, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GemmSwigluPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler( true);
        auto* obj = new GemmSwigluPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
