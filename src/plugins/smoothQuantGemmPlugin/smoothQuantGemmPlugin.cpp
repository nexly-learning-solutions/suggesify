#include "smoothQuantGemmPlugin.h"
#include "../src/weightOnlyBatchedGemv/int8SQ.h"
#include <numeric>

using namespace nvinfer1;
using namespace suggestify::common;
using namespace suggestify::kernels::cutlass_kernels;
using suggestify::plugins::SmoothQuantGemmPluginCreator;
using suggestify::plugins::SmoothQuantGemmPlugin;
using suggestify::plugins::SmoothQuantGemmPluginProfiler;
using suggestify::plugins::read;
using suggestify::plugins::write;

static char const* SQ_GEMM_PLUGIN_VERSION{"1"};
static char const* SQ_GEMM_PLUGIN_NAME{"SmoothQuantGemm"};
PluginFieldCollection SmoothQuantGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SmoothQuantGemmPluginCreator::mPluginAttributes;

void SmoothQuantGemmPluginProfiler::runTactic(int m, int n, int k, SmoothQuantGemmPluginProfiler::Config const& tactic,
    char* workspace, cudaStream_t const& stream)
{
    int8_t* aTmp = reinterpret_cast<int8_t*>(workspace);
    int8_t* bTmp = nextWorkspacePtr(aTmp, m * k * sizeof(int8_t));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(bTmp, n * k * sizeof(int8_t)));
    float* alphaRowTmp = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(cTmp), m * n * (mType == nvinfer1::DataType::kFLOAT ? 4 : 2)));
    float* alphaColTmp
        = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaRowTmp), m * sizeof(float)));
    char* workspaceTmp
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaColTmp), n * sizeof(float)));

    int const wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(
        aTmp, bTmp, mQuantMode, alphaColTmp, alphaRowTmp, cTmp, m, n, k, tactic, workspaceTmp, wsSize, stream);
}

void SmoothQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(int8_t),
        n * k * sizeof(int8_t),
        maxM * n * (mType == nvinfer1::DataType::kFLOAT ? 4u : 2u),
        maxM * sizeof(float),
        n * sizeof(float),
        mRunner->getWorkspaceSize(maxM, n, k)
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<SmoothQuantGemmPluginProfiler::Config> SmoothQuantGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

SmoothQuantGemmPlugin::SmoothQuantGemmPlugin(
    QuantMode quantMode, nvinfer1::DataType type, SmoothQuantGemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mQuantMode(quantMode)
    , mPluginProfiler(pluginProfiler)
{
    init(type);
}

SmoothQuantGemmPlugin::SmoothQuantGemmPlugin(
    void const* data, size_t length, SmoothQuantGemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    bool perChannelScaling = false, perTokenScaling = false;
    nvinfer1::DataType type;
    unsigned int quantMode;
    read(d, quantMode);
    read(d, type);
    read(d, mDims);

    mQuantMode = QuantMode(quantMode);

    init(type);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different nexly version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void SmoothQuantGemmPlugin::init(nvinfer1::DataType type)
{
    mType = type;
    if (mType == nvinfer1::DataType::kHALF)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<half>>();
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<float>>();
    }
    else if (mType == nvinfer1::DataType::kINT32)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<int32_t>>();
    }
#ifdef ENABLE_BF16
    else if (mType == nvinfer1::DataType::kBF16)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<__nv_bfloat16>>();
    }
#endif

    mPluginProfiler->setQuantMode(mQuantMode);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::IPluginV2DynamicExt* SmoothQuantGemmPlugin::clone() const noexcept
{
    auto* plugin = new SmoothQuantGemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs SmoothQuantGemmPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        CHECK(nbInputs == 4);
        CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        CHECK(nbDimsA >= 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = inputs[1].d[0];
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SmoothQuantGemmPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
    case 3:
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 4:
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        assert(false);
        return false;
    }
}

void SmoothQuantGemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];
    int const maxN = in[1].max.d[0];
    int const minK = in[0].min.d[in[0].min.nbDims - 1];
    int const minN = in[1].min.d[0];

    CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mType};

    m_workspaceMaxSize = m_sqGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t SmoothQuantGemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int SmoothQuantGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = INT32_CAST(m64);
    int const n = INT32_CAST(inputDesc[1].dims.d[0]);
    int const k = INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);
    int const wsSize = m_sqGemmRunner->getWorkspaceSize(m, n, k);
    if (m <= 4)
    {
        suggestify::kernels::smooth_quant::Params params(reinterpret_cast<int8_t const*>(inputs[0]),
            reinterpret_cast<int8_t const*>(inputs[1]), reinterpret_cast<float const*>(inputs[2]),
            reinterpret_cast<float const*>(inputs[3]), reinterpret_cast<void*>(outputs[0]), m, n, k, mQuantMode);
        if (mType == nvinfer1::DataType::kHALF)
        {
            suggestify::kernels::smooth_quant::int8_sq_launcher<half>(params, stream);
        }
        else if (mType == nvinfer1::DataType::kFLOAT)
        {
            suggestify::kernels::smooth_quant::int8_sq_launcher<float>(params, stream);
        }
#ifdef ENABLE_BF16
        else if (mType == nvinfer1::DataType::kBF16)
        {
            suggestify::kernels::smooth_quant::int8_sq_launcher<__nv_bfloat16>(params, stream);
        }
#endif
        else if (mType == nvinfer1::DataType::kINT32)
        {
            suggestify::kernels::smooth_quant::int8_sq_launcher<int>(params, stream);
        }
    }
    else
    {
        auto const& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        CHECK_WITH_INFO(bestTactic, "No valid SQ GEMM tactic");
        m_sqGemmRunner->gemm(reinterpret_cast<int8_t const*>(inputs[0]), reinterpret_cast<int8_t const*>(inputs[1]),
            mQuantMode, reinterpret_cast<float const*>(inputs[3]), reinterpret_cast<float const*>(inputs[2]),
            reinterpret_cast<void*>(outputs[0]), m, n, k, *bestTactic, reinterpret_cast<char*>(workspace), wsSize,
            stream);
    }

    return 0;
}

nvinfer1::DataType SmoothQuantGemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return mType;
}


char const* SmoothQuantGemmPlugin::getPluginType() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

char const* SmoothQuantGemmPlugin::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

int SmoothQuantGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int SmoothQuantGemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void SmoothQuantGemmPlugin::terminate() noexcept {}

size_t SmoothQuantGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(unsigned int) +
        sizeof(nvinfer1::DataType) +
        sizeof(mDims) +
        mPluginProfiler->getSerializationSize(mGemmId);
}

void SmoothQuantGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void SmoothQuantGemmPlugin::destroy() noexcept
{
    delete this;
}

void SmoothQuantGemmPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_sqGemmRunner, mType, mDims, mGemmId);
}


SmoothQuantGemmPluginCreator::SmoothQuantGemmPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("has_per_channel_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_per_token_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* SmoothQuantGemmPluginCreator::getPluginName() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

char const* SmoothQuantGemmPluginCreator::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* SmoothQuantGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SmoothQuantGemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    bool perTokenScaling, perChannelScaling;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "has_per_channel_scaling"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            perChannelScaling = static_cast<bool>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "has_per_token_scaling"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            perTokenScaling = static_cast<bool>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( false);
        QuantMode quantMode = QuantMode::fromDescription(true, true, perTokenScaling, perChannelScaling);
        auto* obj = new SmoothQuantGemmPlugin(quantMode, type, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SmoothQuantGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( true);
        auto* obj = new SmoothQuantGemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
