#include "weightOnlyGroupwiseQuantMatmulPlugin.h"

#include <numeric>

using namespace nvinfer1;
using namespace suggestify::common;
using namespace suggestify::kernels::cutlass_kernels;
using suggestify::plugins::WeightOnlyGroupwiseQuantMatmulPluginCreator;
using suggestify::plugins::WeightOnlyGroupwiseQuantMatmulPlugin;
using suggestify::plugins::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using suggestify::plugins::WeightOnlyGemmRunnerPtr;

static constexpr int BIAS = int(1) << 0;
static constexpr int ZERO = int(1) << 1;
static constexpr int PRE_QUANT_SCALE = int(1) << 2;
static constexpr int FP8_ALPHA = int(1) << 3;
static constexpr int INT8_WEIGHT = int(1) << 4;
using suggestify::plugins::read;
using suggestify::plugins::write;

static char const* WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION{"1"};
static char const* WOQ_GROUPWISE_MATMUL_PLUGIN_NAME{"WeightOnlyGroupwiseQuantMatmul"};
PluginFieldCollection WeightOnlyGroupwiseQuantMatmulPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> WeightOnlyGroupwiseQuantMatmulPluginCreator::mPluginAttributes;

void WeightOnlyGroupwiseQuantGemmPluginProfiler::runTactic(int m, int n, int k,
    WeightOnlyGroupwiseQuantGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    int const originalN = mQuantAlgo & INT8_WEIGHT ? n * FP16_INT8_RATIO : n * FP16_INT4_RATIO;
    half* actPtr = reinterpret_cast<half*>(workspace);
    void* weightPtr = nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half));
    half* inputScalesPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(float)));
    half* zerosPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(inputScalesPtr), k * originalN * sizeof(half) / mGroupSize));
    half* biasesPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(zerosPtr), k * originalN * sizeof(half) / mGroupSize));
    half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(biasesPtr), m * sizeof(half)));
    char* workspacePtr
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

    if ((mQuantAlgo & ZERO) == 0)
    {
        zerosPtr = nullptr;
    }

    if ((mQuantAlgo & BIAS) == 0)
    {
        biasesPtr = nullptr;
    }

    int const wsSize = mRunner->getWorkspaceSize(m, originalN, k);
    if (mQuantAlgo & INT8_WEIGHT)
    {
        mRunner->gemm(actPtr, reinterpret_cast<int8_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr, outputPtr, m,
            originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    }
    else
    {
        mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr,
            outputPtr, m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    }
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    int const originalN = mQuantAlgo & INT8_WEIGHT ? n * FP16_INT8_RATIO : n * FP16_INT4_RATIO;
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),
        k * n * sizeof(float),
        k * originalN * sizeof(half) / mGroupSize,
        k * originalN * sizeof(half) / mGroupSize,
        maxM * sizeof(half),
        maxM * originalN * sizeof(half),
        mRunner->getWorkspaceSize(maxM, originalN, k)
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyGroupwiseQuantGemmPluginProfiler::Config> WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(nvinfer1::DataType type, int quant_algo,
    int group_size, WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    init(type, quant_algo, group_size);
}

WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(
    void const* data, size_t length, WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    int quant_algo = 0;
    int group_size = 0;
    read(d, type);
    read(d, quant_algo);
    read(d, group_size);
    read(d, mDims);

    init(type, quant_algo, group_size);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

template <typename ActivationType, typename WeightType, typename OutputType, typename ScaleZeroType,
    cutlass::WeightOnlyQuantOp QuantOp>
using GemmRunner = suggestify::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp,
    ScaleZeroType, OutputType, OutputType>;

template <typename ActivationType, typename WeightType, typename OutputType, typename ScaleZeroType = OutputType>
WeightOnlyGemmRunnerPtr selectGemmRunnerForZERO(int quant_algo)
{
    if (quant_algo & ZERO)
    {
        return std::make_shared<GemmRunner<ActivationType, WeightType, OutputType, ScaleZeroType,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
    }
    else
    {
        return std::make_shared<GemmRunner<ActivationType, WeightType, OutputType, ScaleZeroType,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
    }
}

template <typename ActivationType>
WeightOnlyGemmRunnerPtr selectGemmRunnerForWeightType(int quant_algo)
{
    if (quant_algo & INT8_WEIGHT)
    {
        return selectGemmRunnerForZERO<ActivationType, uint8_t, ActivationType>(quant_algo);
    }
    else
    {
        return selectGemmRunnerForZERO<ActivationType, cutlass::uint4b_t, ActivationType>(quant_algo);
    }
}

void WeightOnlyGroupwiseQuantMatmulPlugin::init(nvinfer1::DataType type, int quant_algo, int group_size)
{
    mArch = suggestify::common::getSMVersion();
    mType = type;
    mQuantAlgo = quant_algo;
    mGroupSize = group_size;

    mPreQuantScaleInputIdx = (quant_algo & PRE_QUANT_SCALE) ? 1 : 0;
    mWeightInputIdx = mPreQuantScaleInputIdx + 1;
    mScalesInputIdx = mWeightInputIdx + 1;
    mZerosInputIdx = (quant_algo & ZERO) ? mScalesInputIdx + 1 : mScalesInputIdx;
    mBiasesInputIdx = (quant_algo & BIAS) ? mZerosInputIdx + 1 : mZerosInputIdx;
    mAlphaInputIdx = (quant_algo & FP8_ALPHA) ? mBiasesInputIdx + 1 : mBiasesInputIdx;

    if (mType == nvinfer1::DataType::kHALF)
    {
        if (quant_algo & FP8_ALPHA)
        {
            if (mArch < 89)
            {
                THROW("W4A(fp)8 kernel is unsupported on pre-Ada (sm<89) architectures!");
            }
            assert(!(quant_algo & INT8_WEIGHT) && "W4A(fp)8 kernel requires INT4 weight!");
            m_weightOnlyGroupwiseGemmRunner
                = selectGemmRunnerForZERO<__nv_fp8_e4m3, cutlass::uint4b_t, half>(quant_algo);
        }
        else
        {
            m_weightOnlyGroupwiseGemmRunner = selectGemmRunnerForWeightType<half>(quant_algo);
        }
        if (quant_algo & INT8_WEIGHT)
        {
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::FP16Int8Groupwise);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::FP16Int8Groupwise;
        }
        else
        {
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::FP16Int4Groupwise);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::FP16Int4Groupwise;
        }
    }
#if defined(ENABLE_BF16)
    else if (mType == nvinfer1::DataType::kBF16)
    {
        if (quant_algo & FP8_ALPHA)
        {
            if (mArch < 89)
            {
                THROW("W4A(fp)8 kernel is unsupported on pre-Ada (sm<89) architectures!");
            }
            assert(!(quant_algo & INT8_WEIGHT) && "W4A(fp)8 kernel requires INT4 weight!");
            m_weightOnlyGroupwiseGemmRunner
                = selectGemmRunnerForZERO<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, half>(quant_algo);
        }
        else
        {
            m_weightOnlyGroupwiseGemmRunner = selectGemmRunnerForWeightType<__nv_bfloat16>(quant_algo);
        }
        if (quant_algo & INT8_WEIGHT)
        {
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::BF16Int8Groupwise);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::BF16Int8Groupwise;
        }
        else
        {
            mCudaKernelEnabled = suggestify::kernels::weight_only::is_supported(
                mArch, suggestify::kernels::weight_only::KernelType::BF16Int4Groupwise);
            mCudaKernelType = suggestify::kernels::weight_only::KernelType::BF16Int4Groupwise;
        }
    }
#endif
    else
    {
        THROW("Unsupported data type");
    }
    mPluginProfiler->setQuantAlgo(mQuantAlgo);
    mPluginProfiler->setGroupSize(mGroupSize);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::IPluginV2DynamicExt* WeightOnlyGroupwiseQuantMatmulPlugin::clone() const noexcept
{
    auto* plugin = new WeightOnlyGroupwiseQuantMatmulPlugin(*this);
    return plugin;
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_weightOnlyGroupwiseGemmRunner, mType, mDims, mGemmId);
}

nvinfer1::DimsExprs WeightOnlyGroupwiseQuantMatmulPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{


    try
    {
        CHECK(nbInputs == mAlphaInputIdx + 1);
        CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        int const nbDimsB = inputs[mWeightInputIdx].nbDims;
        CHECK(nbDimsA >= 2);
        CHECK(nbDimsB == 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }

        int const weight_multiplier = mQuantAlgo & INT8_WEIGHT ? FP16_INT8_RATIO : FP16_INT4_RATIO;
        ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[mWeightInputIdx].d[1]->getConstantValue() * weight_multiplier);

        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool WeightOnlyGroupwiseQuantMatmulPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos < mAlphaInputIdx + 2)
    {
        if (pos == mWeightInputIdx)
        {
            return inOut[mWeightInputIdx].type == mType && inOut[mWeightInputIdx].format == TensorFormat::kLINEAR;
        }
        else if ((mQuantAlgo & FP8_ALPHA) && pos == mAlphaInputIdx)
        {
            return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }
        else
        {
            return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
        }
    }
    else
    {
        assert(false);
        return false;
    }
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];

    int const weight_multiplier = mQuantAlgo & INT8_WEIGHT ? FP16_INT8_RATIO : FP16_INT4_RATIO;
    int const maxN = in[mWeightInputIdx].max.d[1] * weight_multiplier;

    auto const K = maxK;
    auto const N = maxN / weight_multiplier;

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }
    mGemmId = {N, K, mType};

    size_t smoothedActSize = static_cast<size_t>(maxM) * static_cast<size_t>(maxK)
        * (in[0].desc.type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(half));
    m_workspaceMaxSize = smoothedActSize + m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = INT32_CAST(m64);
    int const n = INT32_CAST(inputDesc[mWeightInputIdx].dims.d[1]);
    int const k = INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);

    bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
    bool use_pre_quant_scale = mQuantAlgo & PRE_QUANT_SCALE;

    half const* zeros_ptr = (mQuantAlgo & ZERO) ? reinterpret_cast<half const*>(inputs[mZerosInputIdx]) : nullptr;
    half const* biases_ptr = (mQuantAlgo & BIAS) ? reinterpret_cast<half const*>(inputs[mBiasesInputIdx]) : nullptr;
    half const* act_ptr = reinterpret_cast<half const*>(inputs[0]);
    float alpha = 1.0;
    if (mQuantAlgo & FP8_ALPHA)
    {
        cudaMemcpy(&alpha, const_cast<void*>(inputs[mAlphaInputIdx]), sizeof(float), cudaMemcpyDeviceToHost);
    }

    if (use_pre_quant_scale && !use_cuda_kernel)
    {
        act_ptr = reinterpret_cast<half const*>(workspace);
        if (mType == nvinfer1::DataType::kHALF)
        {
            if (mQuantAlgo & FP8_ALPHA)
            {
                suggestify::kernels::apply_per_channel_scale_kernel_launcher<half, __nv_fp8_e4m3>(
                    reinterpret_cast<__nv_fp8_e4m3*>(workspace), reinterpret_cast<half const*>(inputs[0]),
                    reinterpret_cast<half const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
            else
            {
                suggestify::kernels::apply_per_channel_scale_kernel_launcher<half, half>(
                    reinterpret_cast<half*>(workspace), reinterpret_cast<half const*>(inputs[0]),
                    reinterpret_cast<half const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            if (mQuantAlgo & FP8_ALPHA)
            {
                suggestify::kernels::apply_per_channel_scale_kernel_launcher<__nv_bfloat16, __nv_fp8_e4m3>(
                    reinterpret_cast<__nv_fp8_e4m3*>(workspace), reinterpret_cast<__nv_bfloat16 const*>(inputs[0]),
                    reinterpret_cast<__nv_bfloat16 const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
            else
            {
                suggestify::kernels::apply_per_channel_scale_kernel_launcher<__nv_bfloat16, __nv_bfloat16>(
                    reinterpret_cast<__nv_bfloat16*>(workspace), reinterpret_cast<__nv_bfloat16 const*>(inputs[0]),
                    reinterpret_cast<__nv_bfloat16 const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
        }
#endif
    }

#if defined(ENABLE_BF16)
    CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16,
        "No valid weightOnlyGropwiseQuantMatmul configuration");
#else
    CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF, "No valid weightOnlyGropwiseQuantMatmul configuration");
#endif

    int real_n = mQuantAlgo & INT8_WEIGHT ? n * FP16_INT8_RATIO : n * FP16_INT4_RATIO;

    if (use_cuda_kernel)
    {
        void const* pre_quant_scale_ptr = nullptr;
        if (use_pre_quant_scale)
            pre_quant_scale_ptr = inputs[mPreQuantScaleInputIdx];
        void const* cuda_kernel_act_ptr = inputs[0];
        void const* cuda_kernel_act_scale_ptr = pre_quant_scale_ptr;
        void const* cuda_kernel_weight_ptr = inputs[mWeightInputIdx];
        void const* cuda_kernel_scales_ptr = inputs[mScalesInputIdx];
        void const* cuda_kernel_zeros_ptr = zeros_ptr;
        void const* cuda_kernel_bias_ptr = biases_ptr;
        void* cuda_kernel_out_ptr = outputs[0];
        suggestify::kernels::weight_only::Params params{cuda_kernel_act_ptr, cuda_kernel_act_scale_ptr,
            cuda_kernel_weight_ptr, cuda_kernel_scales_ptr, cuda_kernel_zeros_ptr, cuda_kernel_bias_ptr,
            cuda_kernel_out_ptr, alpha, m, real_n, k, mGroupSize, mCudaKernelType,
            static_cast<bool>(mQuantAlgo & FP8_ALPHA)};
        suggestify::kernels::weight_only::kernel_launcher(mArch, params, stream);
    }
    else
    {
        int const ws_bytes = m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(m, real_n, k);

        int32_t* weight_ptr = const_cast<int32_t*>(reinterpret_cast<int32_t const*>(inputs[mWeightInputIdx]));

        auto const& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        CHECK_WITH_INFO(bestTactic,
            "No valid weight only groupwise GEMM tactic(It is usually caused by the failure to execute all "
            "candidate "
            "configurations of the CUTLASS kernel, please pay attention to the warning information when building "
            "the "
            "engine.)");
        m_weightOnlyGroupwiseGemmRunner->gemm(act_ptr, weight_ptr, inputs[mScalesInputIdx], zeros_ptr, biases_ptr,
            alpha, outputs[0], m, real_n, k, mGroupSize, *bestTactic,
            reinterpret_cast<char*>(workspace) + m * k * sizeof(half), ws_bytes, stream);
    }
    return 0;
}

nvinfer1::DataType WeightOnlyGroupwiseQuantMatmulPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return mType;
}


char const* WeightOnlyGroupwiseQuantMatmulPlugin::getPluginType() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_NAME;
}

char const* WeightOnlyGroupwiseQuantMatmulPlugin::getPluginVersion() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void WeightOnlyGroupwiseQuantMatmulPlugin::terminate() noexcept {}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getSerializationSize() const noexcept
{
    return sizeof(nvinfer1::DataType) +
        sizeof(int) +
        sizeof(int) +
        sizeof(mDims) +
        mPluginProfiler->getSerializationSize(mGemmId);
}

void WeightOnlyGroupwiseQuantMatmulPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mQuantAlgo);
    write(d, mGroupSize);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void WeightOnlyGroupwiseQuantMatmulPlugin::destroy() noexcept
{
    delete this;
}


WeightOnlyGroupwiseQuantMatmulPluginCreator::WeightOnlyGroupwiseQuantMatmulPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quant_algo", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* WeightOnlyGroupwiseQuantMatmulPluginCreator::getPluginName() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_NAME;
}

char const* WeightOnlyGroupwiseQuantMatmulPluginCreator::getPluginVersion() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION;
}

PluginFieldCollection const* WeightOnlyGroupwiseQuantMatmulPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WeightOnlyGroupwiseQuantMatmulPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    int QuantAlgo;
    int GroupSize;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "quant_algo"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            QuantAlgo = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "group_size"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            GroupSize = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
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
        auto* obj = new WeightOnlyGroupwiseQuantMatmulPlugin(type, QuantAlgo, GroupSize, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WeightOnlyGroupwiseQuantMatmulPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler( true);
        auto* obj = new WeightOnlyGroupwiseQuantMatmulPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
