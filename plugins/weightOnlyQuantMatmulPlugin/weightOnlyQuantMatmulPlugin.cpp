#include "weightOnlyQuantMatmulPlugin.h"

#include <numeric>

using namespace nvinfer1;
using namespace suggestify::common;
using namespace suggestify::kernels::cutlass_kernels;
using suggestify::plugins::WeightOnlyQuantMatmulPluginCreator;
using suggestify::plugins::WeightOnlyQuantMatmulPlugin;
using suggestify::plugins::WeightOnlyQuantGemmPluginProfiler;
using suggestify::plugins::read;
using suggestify::plugins::write;

static char const* WOQ_MATMUL_PLUGIN_VERSION{"1"};
static char const* WOQ_MATMUL_PLUGIN_NAME{"WeightOnlyQuantMatmul"};
PluginFieldCollection WeightOnlyQuantMatmulPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> WeightOnlyQuantMatmulPluginCreator::mPluginAttributes;

void WeightOnlyQuantGemmPluginProfiler::runTactic(int m, int n, int k,
    WeightOnlyQuantGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    int const originalN = n * getWeightTypeMultiplier(mWeightTypeId);
    half* actPtr = reinterpret_cast<half*>(workspace);
    int8_t* weightPtr
        = reinterpret_cast<int8_t*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
    half* scalesPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(int8_t)));
    half* outputPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scalesPtr), originalN * sizeof(half)));
    char* workspacePtr
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

    int const wsSize = mRunner->getWorkspaceSize(m, originalN, k);

    if (mWeightTypeId == WeightTypeId::INT8)
    {
        mRunner->gemm(actPtr, weightPtr, scalesPtr, outputPtr, m, originalN, k, tactic, workspacePtr, wsSize, stream);
    }
    else
    {
        mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), scalesPtr, outputPtr, m, originalN, k,
            tactic, workspacePtr, wsSize, stream);
    }
}

void WeightOnlyQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    int const originalN = n * getWeightTypeMultiplier(mWeightTypeId);
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),
        n * k * sizeof(int8_t),
        originalN * sizeof(half),
        maxM * originalN * sizeof(half),
        mRunner->getWorkspaceSize(maxM, originalN, k)
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyQuantGemmPluginProfiler::Config> WeightOnlyQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId,
    WeightOnlyQuantMatmulPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    init(type, weightTypeId);
}

WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(
    void const* data, size_t length, WeightOnlyQuantMatmulPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    WeightTypeId weightTypeId;
    read(d, type);
    read(d, weightTypeId);
    read(d, mDims);

    init(type, weightTypeId);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void WeightOnlyQuantMatmulPlugin::init(nvinfer1::DataType type, WeightTypeId weightTypeId)
{
    mArch = suggestify::common::getSMVersion();
    mType = type;
    mWeightTypeId = weightTypeId;

    if (mWeightTypeId == WeightTypeId::INT8)
    {
        if (mType == nvinfer1::DataType::kHALF)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::FP16Int8PerChannel);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::FP16Int8PerChannel;
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::BF16Int8PerChannel);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::BF16Int8PerChannel;
        }
#endif
        else
        {
            CHECK(false);
        }
    }
    else if (mWeightTypeId == WeightTypeId::INT4)
    {
        if (mType == nvinfer1::DataType::kHALF)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::FP16Int4PerChannel);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::FP16Int4PerChannel;
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            m_weightOnlyGemmRunner = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t,
                cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::BF16Int4PerChannel);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::BF16Int4PerChannel;
        }
#endif
        else
        {
            CHECK(false);
        }
    }
    else
    {
        CHECK(false);
    }

    mPluginProfiler->setWeightTypeId(mWeightTypeId);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::IPluginV2DynamicExt* WeightOnlyQuantMatmulPlugin::clone() const noexcept
{
    auto* plugin = new WeightOnlyQuantMatmulPlugin(*this);
    return plugin;
}

void WeightOnlyQuantMatmulPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_weightOnlyGemmRunner, mType, mDims, mGemmId);
}

nvinfer1::DimsExprs WeightOnlyQuantMatmulPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{

    try
    {
        CHECK(nbInputs == 3);
        CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        int const nbDimsB = inputs[1].nbDims;
        CHECK(nbDimsA >= 2);
        CHECK(nbDimsB == 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        if (mWeightTypeId == WeightTypeId::INT8)
        {
            ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue());
        }
        else
        {
            ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue() * INT8_INT4_RATIO);
        }
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool WeightOnlyQuantMatmulPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        return inOut[0].type == mType && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        return inOut[1].type == nvinfer1::DataType::kINT8 && inOut[1].format == TensorFormat::kLINEAR;
    case 2:
        return inOut[2].type == mType && inOut[2].format == TensorFormat::kLINEAR;
    case 3:
        return inOut[3].type == mType && inOut[3].format == TensorFormat::kLINEAR;
    default:
        assert(false);
        return false;
    }
}

void WeightOnlyQuantMatmulPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];
    int const maxN = in[1].max.d[1] * getWeightTypeMultiplier(mWeightTypeId);

    auto const K = maxK;
    auto const N = maxN / getWeightTypeMultiplier(mWeightTypeId);

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }

    mGemmId = {N, K, mType};

    m_workspaceMaxSize = m_weightOnlyGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t WeightOnlyQuantMatmulPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int WeightOnlyQuantMatmulPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = INT32_CAST(m64);
    int const n = INT32_CAST(inputDesc[1].dims.d[1]);
    int const k = INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);

    if (m == 0)
        return 0;

    bool const use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
#if defined(ENABLE_BF16)
    CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16,
        "No valid weightOnlyQuantMatmul configuration");
#else
    CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF, "No valid weightOnlyQuantMatmul configuration");
#endif
    int real_n = mWeightTypeId == WeightTypeId::INT4 ? n * INT8_INT4_RATIO : n;
    if (use_cuda_kernel)
    {
        void const* cuda_kernel_act_ptr = inputs[0];
        void const* cuda_kernel_weight_ptr = inputs[1];
        void const* cuda_kernel_scales_ptr = inputs[2];
        void* cuda_kernel_out_ptr = outputs[0];
        suggestify::kernels::weight_only::Params params(cuda_kernel_act_ptr, nullptr, cuda_kernel_weight_ptr,
            cuda_kernel_scales_ptr, nullptr, nullptr, cuda_kernel_out_ptr, 1.f, m, real_n, k, 0, mCudaKernelType);
        suggestify::kernels::weight_only::kernel_launcher(mArch, params, stream);
    }
    else
    {
        int const ws_size = m_weightOnlyGemmRunner->getWorkspaceSize(m, real_n, k);

        auto const& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        CHECK_WITH_INFO(bestTactic,
            "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
            "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
            "engine.)");

        m_weightOnlyGemmRunner->gemm(inputs[0], inputs[1], inputs[2], outputs[0], m, real_n, k, *bestTactic,
            reinterpret_cast<char*>(workspace), ws_size, stream);
    }

    return 0;
}

nvinfer1::DataType WeightOnlyQuantMatmulPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return mType;
}


char const* WeightOnlyQuantMatmulPlugin::getPluginType() const noexcept
{
    return WOQ_MATMUL_PLUGIN_NAME;
}

char const* WeightOnlyQuantMatmulPlugin::getPluginVersion() const noexcept
{
    return WOQ_MATMUL_PLUGIN_VERSION;
}

int WeightOnlyQuantMatmulPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int WeightOnlyQuantMatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void WeightOnlyQuantMatmulPlugin::terminate() noexcept {}

size_t WeightOnlyQuantMatmulPlugin::getSerializationSize() const noexcept
{
    return sizeof(mWeightTypeId) +
        sizeof(nvinfer1::DataType) +
        sizeof(mDims) +
        mPluginProfiler->getSerializationSize(mGemmId);
}

void WeightOnlyQuantMatmulPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mWeightTypeId);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void WeightOnlyQuantMatmulPlugin::destroy() noexcept
{
    delete this;
}


WeightOnlyQuantMatmulPluginCreator::WeightOnlyQuantMatmulPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weight_type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* WeightOnlyQuantMatmulPluginCreator::getPluginName() const noexcept
{
    return WOQ_MATMUL_PLUGIN_NAME;
}

char const* WeightOnlyQuantMatmulPluginCreator::getPluginVersion() const noexcept
{
    return WOQ_MATMUL_PLUGIN_VERSION;
}

PluginFieldCollection const* WeightOnlyQuantMatmulPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WeightOnlyQuantMatmulPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    WeightTypeId weightTypeId;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "weight_type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            weightTypeId = static_cast<WeightTypeId>(*(static_cast<int const*>(fields[i].data)));
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
        auto* obj = new WeightOnlyQuantMatmulPlugin(type, weightTypeId, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WeightOnlyQuantMatmulPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( true);
        auto* obj = new WeightOnlyQuantMatmulPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
