
#include "topkLastDimPlugin.h"
#include "../common/assert.h"

using namespace nvinfer1;
using namespace suggestify::kernels;
using namespace suggestify::common;
using suggestify::plugins::TopkLastDimPluginCreator;
using suggestify::plugins::TopkLastDimPlugin;

static char const* TOPK_LAST_DIM_PLUGIN_VERSION{"1"};
static char const* TOPK_LAST_DIM_PLUGIN_NAME{"TopkLastDim"};
PluginFieldCollection TopkLastDimPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> TopkLastDimPluginCreator::mPluginAttributes;

TopkLastDimPlugin::TopkLastDimPlugin(nvinfer1::DataType type, int32_t k, bool is_largest)
    : mType(type)
    , mK(k)
    , mIsLargest(is_largest)
{
    CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
    CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF)
            || (mType == DataType::kINT32),
        "Only support int, float, half, and bfloat16.");
}

TopkLastDimPlugin::TopkLastDimPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mK);
    read(d, mIsLargest);
    CHECK(d == a + length);
    CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
    CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF)
            || (mType == DataType::kINT32),
        "Only support int, float, half, and bfloat16.");
}

nvinfer1::IPluginV2DynamicExt* TopkLastDimPlugin::clone() const noexcept
{
    auto* plugin = new TopkLastDimPlugin(mType, mK, mIsLargest);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs TopkLastDimPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    CHECK_WITH_INFO(outputIndex < 2, "Only 2 outputs.");
    nvinfer1::DimsExprs output(inputs[0]);
    int numDim = output.nbDims;
    output.d[numDim - 1] = exprBuilder.constant(mK);
    return output;
}

bool TopkLastDimPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    bool res = inOut[pos].format == TensorFormat::kLINEAR;
    if (pos < 2)
    {
        res = res && inOut[pos].type == mType;
    }
    else if (pos == 2)
    {
        res = res && inOut[pos].type == DataType::kINT32;
    }
    return res;
}

void TopkLastDimPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t TopkLastDimPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    auto const batchSize = inputs[getInputTensorIdx()].dims.d[0];
    auto const inputLength = inputs[getInputTensorIdx()].dims.d[1];
    size_t tempStorageBytes;
    if (mType == DataType::kINT32)
    {
        tempStorageBytes = invokeComputeTopkLastDimWorkspaceSize<int>(batchSize, inputLength, mK, mIsLargest);
    }
    else if (mType == DataType::kHALF)
    {
        tempStorageBytes = invokeComputeTopkLastDimWorkspaceSize<half>(batchSize, inputLength, mK, mIsLargest);
    }
    else if (mType == DataType::kFLOAT)
    {
        tempStorageBytes = invokeComputeTopkLastDimWorkspaceSize<float>(batchSize, inputLength, mK, mIsLargest);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        tempStorageBytes = invokeComputeTopkLastDimWorkspaceSize<__nv_bfloat16>(batchSize, inputLength, mK, mIsLargest);
    }
#endif
    return tempStorageBytes;
}

template <typename T>
int TopkLastDimPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    auto const batchSize = inputDesc[getInputTensorIdx()].dims.d[0];
    auto const inputLength = inputDesc[getInputTensorIdx()].dims.d[1];
    if (batchSize == 0)
    {
        return 0;
    }

    invokeTopkLastDim<T>(
        batchSize, inputLength, mK, mIsLargest, inputs[getInputTensorIdx()], outputs[0], outputs[1], workspace, stream);

    sync_check_cuda_error();
    return 0;
}

int TopkLastDimPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (mType == DataType::kINT32)
    {
        return enqueueImpl<int>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#endif
    return 0;
}

nvinfer1::DataType TopkLastDimPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK_WITH_INFO(index < 2, "Only 2 outputs.");
    nvinfer1::DataType data_type;
    if (index == 1)
    {
        data_type = DataType::kINT32;
    }
    else
    {
        data_type = inputTypes[getInputTensorIdx()];
    }
    return data_type;
}


char const* TopkLastDimPlugin::getPluginType() const noexcept
{
    return TOPK_LAST_DIM_PLUGIN_NAME;
}

char const* TopkLastDimPlugin::getPluginVersion() const noexcept
{
    return TOPK_LAST_DIM_PLUGIN_VERSION;
}

int TopkLastDimPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int TopkLastDimPlugin::initialize() noexcept
{
    return 0;
}

void TopkLastDimPlugin::terminate() noexcept {}

size_t TopkLastDimPlugin::getSerializationSize() const noexcept
{
    return sizeof(mType) + sizeof(mK) + sizeof(mIsLargest);
}

void TopkLastDimPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mK);
    write(d, mIsLargest);
    assert(d == a + getSerializationSize());
}

void TopkLastDimPlugin::destroy() noexcept
{
    delete this;
}


TopkLastDimPluginCreator::TopkLastDimPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("is_largest", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* TopkLastDimPluginCreator::getPluginName() const noexcept
{
    return TOPK_LAST_DIM_PLUGIN_NAME;
}

char const* TopkLastDimPluginCreator::getPluginVersion() const noexcept
{
    return TOPK_LAST_DIM_PLUGIN_VERSION;
}

PluginFieldCollection const* TopkLastDimPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* TopkLastDimPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    int32_t k;
    bool is_largest;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "k"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            k = static_cast<int32_t>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "is_largest"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            is_largest = static_cast<int32_t>(*(static_cast<int const*>(fields[i].data))) != 0;
        }
    }
    try
    {
        auto* obj = new TopkLastDimPlugin(type, k, is_largest);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* TopkLastDimPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new TopkLastDimPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
