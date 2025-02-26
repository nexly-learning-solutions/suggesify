
#include "fp8RowwiseGemmPlugin.h"
#include "cutlass_extensions/gemm_configs.h"

#include <NvInferRuntimeBase.h>
#include <numeric>

using namespace nvinfer1;
using namespace suggestify::common;
using namespace suggestify::kernels::cutlass_kernels;
using suggestify::plugins::Fp8RowwiseGemmPluginCreator;
using suggestify::plugins::Fp8RowwiseGemmPlugin;
using suggestify::plugins::Fp8RowwiseGemmPluginProfiler;

static char const* FP8_ROWWISE_GEMM_PLUGIN_VERSION{"1"};
static char const* FP8_ROWWISE_GEMM_PLUGIN_NAME{"Fp8RowwiseGemm"};
PluginFieldCollection Fp8RowwiseGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> Fp8RowwiseGemmPluginCreator::mPluginAttributes;

size_t Fp8RowwiseGemmPluginProfiler::getBytePerElement(nvinfer1::DataType type)
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

void Fp8RowwiseGemmPluginProfiler::setQuantMode(suggestify::common::QuantMode const& quantMode)
{
    mQuantMode = quantMode;
}

void Fp8RowwiseGemmPluginProfiler::runTactic(int m, int n, int k, Fp8RowwiseGemmPluginProfiler::Config const& tactic,
    char* workspace, cudaStream_t const& stream)
{
    size_t bpeIn = getBytePerElement(nvinfer1::DataType::kFP8);
    size_t bpeOut = getBytePerElement(mType);

    size_t wsSizeRunner = mRunner->getWorkspaceSize(m, n, k);

    size_t wsByteOffset = 0;
    int8_t* wsBytePointer = reinterpret_cast<int8_t*>(workspace);
    void* aTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * k * bpeIn));
    void* bTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, n * k * bpeIn));
    void* dTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * n * bpeOut));
    float* scaleD0Tmp = reinterpret_cast<float*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * sizeof(float)));
    float* scaleD1Tmp = reinterpret_cast<float*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, n * sizeof(float)));
    char* workspaceTmp = reinterpret_cast<char*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, wsSizeRunner));

    mRunner->gemm(dTmp, aTmp, bTmp, nullptr, mQuantMode, m, n, k, scaleD0Tmp, scaleD1Tmp, tactic, workspaceTmp,
        wsSizeRunner, stream);
    sync_check_cuda_error();
}

int Fp8RowwiseGemmPluginProfiler::getMaxProfileM() const
{
    return 16384;
}

void Fp8RowwiseGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    std::vector<size_t> workspaces = {
        maxM * k * getBytePerElement(nvinfer1::DataType::kFP8),
        n * k * getBytePerElement(nvinfer1::DataType::kFP8),
        maxM * n * getBytePerElement(mType),
        maxM * sizeof(float),
        n * sizeof(float),
        maxM * sizeof(float),
        mRunner->getWorkspaceSize(maxM, n, k)
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<Fp8RowwiseGemmPluginProfiler::Config> Fp8RowwiseGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

Fp8RowwiseGemmPlugin::Fp8RowwiseGemmPlugin(
    QuantMode quantMode, nvinfer1::DataType type, Fp8RowwiseGemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mQuantMode(quantMode)
    , mPluginProfiler(pluginProfiler)
{
    init(type);
}

Fp8RowwiseGemmPlugin::Fp8RowwiseGemmPlugin(
    void const* data, size_t length, Fp8RowwiseGemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    unsigned int quantMode;
    read(d, quantMode);
    read(d, type);
    read(d, mDims);

    mQuantMode = QuantMode(quantMode);

    init(type);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK(d == a + length);
}

void Fp8RowwiseGemmPlugin::init(nvinfer1::DataType type)
{
    mType = type;
    if (mType == nvinfer1::DataType::kHALF)
    {
        mGemmRunner = std::make_shared<CutlassFp8RowwiseGemmRunner<half>>();
    }
#ifdef ENABLE_BF16
    else if (mType == nvinfer1::DataType::kBF16)
    {
        mGemmRunner = std::make_shared<CutlassFp8RowwiseGemmRunner<__nv_bfloat16>>();
    }
#endif
    else
    {
        TLLM_THROW("Fp8 Rowwise Gemm plugin doesn't support this type now");
    }

    mPluginProfiler->setQuantMode(mQuantMode);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::IPluginV2DynamicExt* Fp8RowwiseGemmPlugin::clone() const noexcept
{
    auto* plugin = new Fp8RowwiseGemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs Fp8RowwiseGemmPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 4);
        TLLM_CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
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

bool Fp8RowwiseGemmPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return inOut[pos].type == nvinfer1::DataType::kFP8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[pos].type == nvinfer1::DataType::kFP8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
    case 3:
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 4:
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        return false;
    }
}

void Fp8RowwiseGemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];
    int const maxN = in[1].max.d[0];
    int const minK = in[0].min.d[in[0].min.nbDims - 1];
    int const minN = in[1].min.d[0];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mType};

    mWorkspaceMaxSize = mGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t Fp8RowwiseGemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mWorkspaceMaxSize;
}

int Fp8RowwiseGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    int const n = inputDesc[1].dims.d[0];
    int const k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    size_t const wsSize = mGemmRunner->getWorkspaceSize(m, n, k);

    auto const bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid GEMM tactic");
    mGemmRunner->gemm(outputs[0], inputs[0], inputs[1], nullptr, mQuantMode, m, n, k,
        reinterpret_cast<float const*>(inputs[2]), reinterpret_cast<float const*>(inputs[3]), *bestTactic,
        reinterpret_cast<char*>(workspace), wsSize, stream);
    sync_check_cuda_error();

    return 0;
}

nvinfer1::DataType Fp8RowwiseGemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}


char const* Fp8RowwiseGemmPlugin::getPluginType() const noexcept
{
    return FP8_ROWWISE_GEMM_PLUGIN_NAME;
}

char const* Fp8RowwiseGemmPlugin::getPluginVersion() const noexcept
{
    return FP8_ROWWISE_GEMM_PLUGIN_VERSION;
}

int Fp8RowwiseGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int Fp8RowwiseGemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void Fp8RowwiseGemmPlugin::terminate() noexcept {}

size_t Fp8RowwiseGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(unsigned int) +
        sizeof(nvinfer1::DataType) +
        sizeof(mDims) +
        mPluginProfiler->getSerializationSize(mGemmId);
}

void Fp8RowwiseGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    TLLM_CHECK(d == a + getSerializationSize());
}

void Fp8RowwiseGemmPlugin::destroy() noexcept
{
    delete this;
}

void Fp8RowwiseGemmPlugin::configGemm()
{
    mPluginProfiler->profileTactics(mGemmRunner, mType, mDims, mGemmId);
}

Fp8RowwiseGemmPluginCreator::Fp8RowwiseGemmPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("has_per_channel_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_per_token_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* Fp8RowwiseGemmPluginCreator::getPluginName() const noexcept
{
    return FP8_ROWWISE_GEMM_PLUGIN_NAME;
}

char const* Fp8RowwiseGemmPluginCreator::getPluginVersion() const noexcept
{
    return FP8_ROWWISE_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* Fp8RowwiseGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Fp8RowwiseGemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    TLLM_CHECK(fc->nbFields == 3);
    bool perTokenScaling, perChannelScaling;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "has_per_channel_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            perChannelScaling = static_cast<bool>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "has_per_token_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            perTokenScaling = static_cast<bool>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler( false);
        QuantMode quantMode = QuantMode::fromDescription();
        auto* obj = new Fp8RowwiseGemmPlugin(quantMode, type, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* Fp8RowwiseGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler( true);
        auto* obj = new Fp8RowwiseGemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
