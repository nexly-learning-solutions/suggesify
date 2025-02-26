#include "allgatherPlugin.h"
#include "suggestify/common/mpiUtils.h"

#include <nccl.h>

using namespace nvinfer1;
using suggestify::plugins::AllgatherPluginCreator;
using suggestify::plugins::AllgatherPlugin;

static char const* ALLGATHER_PLUGIN_VERSION{"1"};
static char const* ALLGATHER_PLUGIN_NAME{"AllGather"};
PluginFieldCollection AllgatherPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> AllgatherPluginCreator::mPluginAttributes;

AllgatherPlugin::AllgatherPlugin(std::set<int> group, nvinfer1::DataType type)
    : mGroup(std::move(group))
    , mType(type)
{
}

AllgatherPlugin::AllgatherPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    mGroup.clear();
    int groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mGroup.insert(groupItem);
    }
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* AllgatherPlugin::clone() const noexcept
{
    auto* plugin = new AllgatherPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AllgatherPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    auto ret = inputs[0];
    auto groupSize = exprBuilder.constant(mGroup.size());
    ret.d[0] = exprBuilder.operation(DimensionOperation::kPROD, *ret.d[0], *groupSize);
    return ret;
}

bool AllgatherPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{

    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void AllgatherPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t AllgatherPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int AllgatherPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
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

    TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
    NCCLCHECK(ncclAllGather(inputs[0], outputs[0], size, (*getDtypeMap())[inputDesc[0].type], *mNcclComm, stream));

    return 0;
}

nvinfer1::DataType AllgatherPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}


char const* AllgatherPlugin::getPluginType() const noexcept
{
    return ALLGATHER_PLUGIN_NAME;
}

char const* AllgatherPlugin::getPluginVersion() const noexcept
{
    return ALLGATHER_PLUGIN_VERSION;
}

int AllgatherPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int AllgatherPlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    mNcclComm = getComm(mGroup);
    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    return 0;
}

void AllgatherPlugin::terminate() noexcept {}

size_t AllgatherPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType);
}

void AllgatherPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void AllgatherPlugin::destroy() noexcept
{
    delete this;
}


AllgatherPluginCreator::AllgatherPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* AllgatherPluginCreator::getPluginName() const noexcept
{
    return ALLGATHER_PLUGIN_NAME;
}

char const* AllgatherPluginCreator::getPluginVersion() const noexcept
{
    return ALLGATHER_PLUGIN_VERSION;
}

PluginFieldCollection const* AllgatherPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AllgatherPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            auto const* r = static_cast<int const*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new AllgatherPlugin(group, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AllgatherPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new AllgatherPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
