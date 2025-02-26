
#include "cumsumLastDimPlugin.h"
#include "suggestify/common/assert.h"

using namespace nvinfer1;
using namespace suggestify::kernels;
using namespace suggestify::common;
using suggestify::plugins::CumsumLastDimPluginCreator;
using suggestify::plugins::CumsumLastDimPlugin;

static char const* CUMSUM_LAST_DIM_PLUGIN_VERSION{"1"};
static char const* CUMSUM_LAST_DIM_PLUGIN_NAME{"CumsumLastDim"};
PluginFieldCollection CumsumLastDimPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CumsumLastDimPluginCreator::mPluginAttributes;

static constexpr SizeType32 LENGTH_LIMIT_FOR_BLOCKSCAN = 4096;

CumsumLastDimPlugin::CumsumLastDimPlugin(SizeType32 inputLength, nvinfer1::DataType type, size_t temp_storage_bytes)
    : mInputLength(inputLength)
    , mTempStorageBytes(temp_storage_bytes)
    , mType(type)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF)
            || (mType == DataType::kINT32),
        "Only support int, float, half, and bfloat16.");
    if (mTempStorageBytes == 0)
    {
        mTempStorageBytes = getWorkspaceSizeNeeded(inputLength, type);
    }
}

CumsumLastDimPlugin::CumsumLastDimPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mInputLength);
    read(d, mTempStorageBytes);
    read(d, mType);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF)
            || (mType == DataType::kINT32),
        "Only support int, float, half, and bfloat16.");
}

nvinfer1::IPluginV2DynamicExt* CumsumLastDimPlugin::clone() const noexcept
{
    auto* plugin = new CumsumLastDimPlugin(mInputLength, mType, mTempStorageBytes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs CumsumLastDimPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK_WITH_INFO(outputIndex == 0, "Only one output.");
    return inputs[getInputTensorIdx()];
}

bool CumsumLastDimPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void CumsumLastDimPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t CumsumLastDimPlugin::getWorkspaceSizeNeeded(SizeType32 inputLength, nvinfer1::DataType type)
{
    size_t tempStorageBytes;
    if (inputLength < LENGTH_LIMIT_FOR_BLOCKSCAN)
    {
        tempStorageBytes = 0;
    }
    else if (type == DataType::kINT32)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<int>(inputLength);
    }
    else if (type == DataType::kHALF)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<half>(inputLength);
    }
    else if (type == DataType::kFLOAT)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<float>(inputLength);
    }
#ifdef ENABLE_BF16
    else if (type == DataType::kBF16)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<__nv_bfloat16>(inputLength);
    }
#endif
    return tempStorageBytes;
}

size_t CumsumLastDimPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mTempStorageBytes;
}

template <typename T>
int CumsumLastDimPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    auto const batchSize = inputDesc[getInputTensorIdx()].dims.d[0];
    auto const inputLength = inputDesc[getInputTensorIdx()].dims.d[1];
    void* wp = inputLength < LENGTH_LIMIT_FOR_BLOCKSCAN || batchSize > 2 ? nullptr : workspace;
    invokeCumsumLastDim<T>(
        batchSize, inputLength, inputs[getInputTensorIdx()], outputs[0], wp, mTempStorageBytes, stream);

    sync_check_cuda_error();
    return 0;
}

int CumsumLastDimPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
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

nvinfer1::DataType CumsumLastDimPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK_WITH_INFO(index == 0, "Only one output.");
    return inputTypes[getInputTensorIdx()];
}


char const* CumsumLastDimPlugin::getPluginType() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_NAME;
}

char const* CumsumLastDimPlugin::getPluginVersion() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_VERSION;
}

int CumsumLastDimPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int CumsumLastDimPlugin::initialize() noexcept
{
    return 0;
}

void CumsumLastDimPlugin::terminate() noexcept {}

size_t CumsumLastDimPlugin::getSerializationSize() const noexcept
{
    return sizeof(mInputLength) + sizeof(mTempStorageBytes) + sizeof(mType);
}

void CumsumLastDimPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mInputLength);
    write(d, mTempStorageBytes);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void CumsumLastDimPlugin::destroy() noexcept
{
    delete this;
}


CumsumLastDimPluginCreator::CumsumLastDimPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("input_length", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CumsumLastDimPluginCreator::getPluginName() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_NAME;
}

char const* CumsumLastDimPluginCreator::getPluginVersion() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_VERSION;
}

PluginFieldCollection const* CumsumLastDimPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* CumsumLastDimPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int inputLength;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "input_length"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            inputLength = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new CumsumLastDimPlugin(inputLength, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* CumsumLastDimPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new CumsumLastDimPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
