
#include "qserveGemmPlugin.h"
#include "../src/qserveGemm.h"
#include <cassert>
#include <numeric>

using namespace nvinfer1;
using namespace suggestify::common;
using suggestify::plugins::QServeGemmPluginCreator;
using suggestify::plugins::QServeGemmPlugin;
using suggestify::plugins::read;
using suggestify::plugins::write;
using namespace suggestify::kernels::qserve;

static char const* QSERVE_GEMM_PLUGIN_VERSION{"1"};
static char const* QSERVE_GEMM_PLUGIN_NAME{"QServeGemm"};

PluginFieldCollection QServeGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QServeGemmPluginCreator::mPluginAttributes;

namespace suggestify::plugins
{

QServeGemmPlugin::QServeGemmPlugin(
    nvinfer1::DataType dtype, int groupSize)
{
    init(dtype, groupSize);
}

QServeGemmPlugin::QServeGemmPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;

    nvinfer1::DataType type;
    unsigned int quantMode;
    int groupSize;

    read(d, quantMode);
    read(d, type);
    read(d, groupSize);

    read(d, mDims);


    init(type, groupSize);

    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void QServeGemmPlugin::init(nvinfer1::DataType dtype, int groupSize)
{
    if (groupSize <= 0)
        groupSize = -1;
    mGroupSize = groupSize;
    mType = dtype;
    mRunner = std::make_shared<QServeGemmRunner>();

    int arch = suggestify::common::getSMVersion();

    if (arch < 80)
    {
        THROW("QServe W4A8 is unsupported on pre-Ampere (sm<80) architectures!");
    }
}

nvinfer1::IPluginV2DynamicExt* QServeGemmPlugin::clone() const noexcept
{
    auto* plugin = new QServeGemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs QServeGemmPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        CHECK(nbInputs == 6);
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

bool QServeGemmPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (mGroupSize != -1)
    {
        switch (pos)
        {
        case 0:
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 1:
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 2:
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 3:
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 4:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 5:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 6:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        default: return false;
        }
    }

    else
    {
        switch (pos)
        {
        case 0:
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 1:
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 2:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 3:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 4:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 5:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 6:
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        default: return false;
        }
    }
}

void QServeGemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
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
    m_workspaceMaxSize = mRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t QServeGemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int QServeGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{




    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = INT32_CAST(m64);
    int const n = INT32_CAST(inputDesc[1].dims.d[0]);
    int const k = INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);


    if (mGroupSize != -1)
    {
        ParamsPerGroup params = {reinterpret_cast<int8_t const*>(inputs[0]),
            reinterpret_cast<int8_t const*>(inputs[1]),
            reinterpret_cast<int8_t const*>(inputs[2]),
            reinterpret_cast<int8_t const*>(inputs[3]),
            reinterpret_cast<half const*>(inputs[4]),
            reinterpret_cast<half const*>(inputs[5]),
            reinterpret_cast<half*>(outputs[0]),
            m, n, k};
        mRunner->gemmPerGroup(params, stream);
    }
    else
    {
        ParamsPerChannel params = {reinterpret_cast<int8_t const*>(inputs[0]),
            reinterpret_cast<int8_t const*>(inputs[1]),
            reinterpret_cast<half const*>(inputs[2]),
            reinterpret_cast<half const*>(inputs[3]),
            reinterpret_cast<half const*>(inputs[4]),
            reinterpret_cast<half const*>(inputs[5]),
            reinterpret_cast<half*>(outputs[0]),
            m, n, k};
        mRunner->gemmPerChannel(params, stream);
    }

    return 0;
}

nvinfer1::DataType QServeGemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return mType;
}


char const* QServeGemmPlugin::getPluginType() const noexcept
{
    return QSERVE_GEMM_PLUGIN_NAME;
}

char const* QServeGemmPlugin::getPluginVersion() const noexcept
{
    return QSERVE_GEMM_PLUGIN_VERSION;
}

int QServeGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int QServeGemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void QServeGemmPlugin::terminate() noexcept {}

size_t QServeGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mQuantMode) +
        sizeof(mType) +
        sizeof(mGroupSize) +
        sizeof(mDims);
}

void QServeGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mGroupSize);
    write(d, mDims);

    assert(d == a + getSerializationSize());
}

void QServeGemmPlugin::destroy() noexcept
{
    delete this;
}

void QServeGemmPlugin::configGemm() {}


QServeGemmPluginCreator::QServeGemmPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.push_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.push_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QServeGemmPluginCreator::getPluginName() const noexcept
{
    return QSERVE_GEMM_PLUGIN_NAME;
}

char const* QServeGemmPluginCreator::getPluginVersion() const noexcept
{
    return QSERVE_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* QServeGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QServeGemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{

    PluginField const* fields = fc->fields;

    DataType dtype;
    int group_size = -1;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            dtype = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
            assert(dtype == nvinfer1::DataType::kHALF);
        }
        else if (!strcmp(attrName, "group_size"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            group_size = *static_cast<int const*>(fields[i].data);
            assert(group_size == -1 || group_size == 128);
        }
    }
    try
    {
        auto* obj = new QServeGemmPlugin(dtype, group_size);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QServeGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new QServeGemmPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
}
