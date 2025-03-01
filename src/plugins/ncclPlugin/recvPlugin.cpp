#include "recvPlugin.h"

#include "../common/logger.h"
#include "../common/mpiUtils.h"

#include <nccl.h>

using namespace nvinfer1;
using suggestify::plugins::RecvPluginCreator;
using suggestify::plugins::RecvPlugin;

static char const* RECV_PLUGIN_VERSION{"1"};
static char const* RECV_PLUGIN_NAME{"Recv"};
PluginFieldCollection RecvPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RecvPluginCreator::mPluginAttributes;

RecvPlugin::RecvPlugin(int srcRank, nvinfer1::DataType type)
    : mSrcRank(srcRank)
    , mType(type)
{
}

RecvPlugin::RecvPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mSrcRank);
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different nexly version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* RecvPlugin::clone() const noexcept
{
    auto* plugin = new RecvPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RecvPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool RecvPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RecvPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t RecvPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RecvPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    size_t size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }
    LOG_DEBUG("start ncclRecv with size %d", size);
    NCCLCHECK(ncclRecv(outputs[0], size, (*getDtypeMap())[inputDesc[0].type], 0, mComm, stream));
    LOG_DEBUG("end ncclRecv with size %d", size);

    return 0;
}

nvinfer1::DataType RecvPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}


char const* RecvPlugin::getPluginType() const noexcept
{
    return RECV_PLUGIN_NAME;
}

char const* RecvPlugin::getPluginVersion() const noexcept
{
    return RECV_PLUGIN_VERSION;
}

int RecvPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int RecvPlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    ncclUniqueId id;
    COMM_SESSION.recvValue(id, mSrcRank, 0);
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    NCCLCHECK(ncclCommInitRank(&mComm, 2, id, 1));
    return 0;
}

void RecvPlugin::terminate() noexcept
{
    if (isBuilding())
    {
        return;
    }
    NCCLCHECK(ncclCommDestroy(mComm));
}

size_t RecvPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSrcRank) + sizeof(mType);
}

void RecvPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mSrcRank);
    assert(d == a + getSerializationSize());
}

void RecvPlugin::destroy() noexcept
{
    delete this;
}


RecvPluginCreator::RecvPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("src_rank", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* RecvPluginCreator::getPluginName() const noexcept
{
    return RECV_PLUGIN_NAME;
}

char const* RecvPluginCreator::getPluginVersion() const noexcept
{
    return RECV_PLUGIN_VERSION;
}

PluginFieldCollection const* RecvPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RecvPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int srcRank;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "src_rank"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            srcRank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new RecvPlugin(srcRank, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RecvPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new RecvPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
