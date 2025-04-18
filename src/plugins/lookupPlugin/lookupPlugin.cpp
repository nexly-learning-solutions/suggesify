
#include <cstdio>

#include "lookupPlugin.h"
#include "../src/lookupKernels.h"
#include "../plugins/common/plugin.h"

using namespace nvinfer1;
using namespace suggestify::kernels;
using namespace suggestify::common;
using suggestify::plugins::LookupPluginCreator;
using suggestify::plugins::LookupPlugin;

static char const* LOOKUP_PLUGIN_VERSION{"1"};
static char const* LOOKUP_PLUGIN_NAME{"Lookup"};
PluginFieldCollection LookupPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LookupPluginCreator::mPluginAttributes;

LookupPlugin::LookupPlugin(nvinfer1::DataType type, int rank)
    : mType(type)
    , mRank(rank)
{
    mArch = suggestify::common::getSMVersion();
}

LookupPlugin::LookupPlugin(void const* data, size_t length)
{
    mArch = suggestify::common::getSMVersion();
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mRank);
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different nexly version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* LookupPlugin::clone() const noexcept
{
    auto* plugin = new LookupPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs LookupPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        CHECK(nbInputs == 2 || nbInputs == 3);
        CHECK(outputIndex == 0);
        DimsExprs ret;
        int const nbDimsInput = inputs[0].nbDims;
        int const nbDimsWeight = inputs[1].nbDims;
        ret.nbDims = nbDimsInput + 1;

        for (int i = 0; i < nbDimsInput; ++i)
        {
            ret.d[i] = inputs[0].d[i];
        }
        ret.d[nbDimsInput] = inputs[1].d[nbDimsWeight - 1];

        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LookupPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    bool res = false;
    if (nbInputs == 2)
    {
        switch (pos)
        {
        case 0: res = ((inOut[0].type == DataType::kINT32) && (inOut[0].format == TensorFormat::kLINEAR)); break;
        case 1: res = ((inOut[1].type == mType) && (inOut[1].format == TensorFormat::kLINEAR)); break;
        case 2: res = ((inOut[2].type == mType) && (inOut[2].format == TensorFormat::kLINEAR)); break;
        default:
            res = false;
        }
    }
    else
    {
        CHECK_WITH_INFO(mArch == 90, "int8 weight only lookupPlugin is only supported in SM 90 now.");
        switch (pos)
        {
        case 0: res = ((inOut[0].type == DataType::kINT32) && (inOut[0].format == TensorFormat::kLINEAR)); break;
        case 1:
            res = ((inOut[1].type == DataType::kINT8 || inOut[1].type == mType)
                && (inOut[1].format == TensorFormat::kLINEAR));
            break;
        case 2: res = ((inOut[2].type == mType) && (inOut[2].format == TensorFormat::kLINEAR)); break;
        case 3: res = ((inOut[3].type == mType) && (inOut[3].format == TensorFormat::kLINEAR)); break;
        default:
            res = false;
        }
    }
    return res;
}

void LookupPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    mNbInputs = nbInputs;
}

size_t LookupPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int LookupPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{

    int64_t tokenNum = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        tokenNum *= inputDesc[0].dims.d[i];
    }

    int const localVocabSize = inputDesc[1].dims.d[0];
    int const hidden = inputDesc[1].dims.d[inputDesc[1].dims.nbDims - 1];
    int const* input = reinterpret_cast<int const*>(inputs[0]);

    int offset = mRank * localVocabSize;

    if (mNbInputs == 3)
    {
        int8_t const* weight = reinterpret_cast<int8_t const*>(inputs[1]);
        if (mType == DataType::kHALF)
        {
            half const* per_token_scales = reinterpret_cast<half const*>(inputs[2]);
            half* output = reinterpret_cast<half*>(outputs[0]);
            invokeLookUp<half, int8_t, int>(
                output, input, weight, tokenNum, offset, localVocabSize, hidden, per_token_scales, stream);
        }
        else if (mType == DataType::kFLOAT)
        {
            float const* per_token_scales = reinterpret_cast<float const*>(inputs[2]);
            float* output = reinterpret_cast<float*>(outputs[0]);
            invokeLookUp<float, int8_t, int>(
                output, input, weight, tokenNum, offset, localVocabSize, hidden, per_token_scales, stream);
        }
        else if (mType == DataType::kBF16)
        {
            __nv_bfloat16 const* per_token_scales = reinterpret_cast<__nv_bfloat16 const*>(inputs[2]);
            __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
            invokeLookUp<__nv_bfloat16, int8_t, int>(
                output, input, weight, tokenNum, offset, localVocabSize, hidden, per_token_scales, stream);
        }
    }
    else
    {
        if (mType == DataType::kHALF)
        {
            half const* weight = reinterpret_cast<half const*>(inputs[1]);
            half* output = reinterpret_cast<half*>(outputs[0]);
            invokeLookUp<half, half, int>(
                output, input, weight, tokenNum, offset, localVocabSize, hidden, nullptr, stream);
        }
        else if (mType == DataType::kFLOAT)
        {
            float const* weight = reinterpret_cast<float const*>(inputs[1]);
            float* output = reinterpret_cast<float*>(outputs[0]);
            invokeLookUp<float, float, int>(
                output, input, weight, tokenNum, offset, localVocabSize, hidden, nullptr, stream);
        }
        else if (mType == DataType::kBF16)
        {
            __nv_bfloat16 const* weight = reinterpret_cast<__nv_bfloat16 const*>(inputs[1]);
            __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
            invokeLookUp<__nv_bfloat16, __nv_bfloat16, int>(
                output, input, weight, tokenNum, offset, localVocabSize, hidden, nullptr, stream);
        }
    }
    sync_check_cuda_error();

    return 0;
}

nvinfer1::DataType LookupPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return mType;
}


char const* LookupPlugin::getPluginType() const noexcept
{
    return LOOKUP_PLUGIN_NAME;
}

char const* LookupPlugin::getPluginVersion() const noexcept
{
    return LOOKUP_PLUGIN_VERSION;
}

int LookupPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int LookupPlugin::initialize() noexcept
{
    return 0;
}

void LookupPlugin::destroy() noexcept
{
    delete this;
}

size_t LookupPlugin::getSerializationSize() const noexcept
{
    return sizeof(mType) + sizeof(mRank);
}

void LookupPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mRank);

    assert(d == a + getSerializationSize());
}

void LookupPlugin::terminate() noexcept {}


LookupPluginCreator::LookupPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rank", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* LookupPluginCreator::getPluginName() const noexcept
{
    return LOOKUP_PLUGIN_NAME;
}

char const* LookupPluginCreator::getPluginVersion() const noexcept
{
    return LOOKUP_PLUGIN_VERSION;
}

PluginFieldCollection const* LookupPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LookupPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    int rank;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "rank"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            rank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new LookupPlugin(type, rank);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LookupPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new LookupPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
