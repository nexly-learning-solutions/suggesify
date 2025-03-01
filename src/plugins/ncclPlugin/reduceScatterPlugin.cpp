#include "reduceScatterPlugin.h"

#include <cassert>
#include <nccl.h>

using namespace nvinfer1;
using suggestify::plugins::ReduceScatterPluginCreator;
using suggestify::plugins::ReduceScatterPlugin;

static char const* REDUCE_SCATTER_PLUGIN_VERSION{"1"};
static char const* REDUCE_SCATTER_PLUGIN_NAME{"ReduceScatter"};
PluginFieldCollection ReduceScatterPluginCreator::mFC{};
std::vector<PluginField> ReduceScatterPluginCreator::mPluginAttributes;

ReduceScatterPlugin::ReduceScatterPlugin(std::set<int> group, nvinfer1::DataType type)
    : mGroup(std::move(group))
    , mType(type)
{
}

ReduceScatterPlugin::ReduceScatterPlugin(void const* data, size_t length)
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
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different nexly version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* ReduceScatterPlugin::clone() const noexcept
{
    auto* plugin = new ReduceScatterPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs ReduceScatterPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    auto output = inputs[0];
    output.d[0]
        = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *output.d[0], *exprBuilder.constant(mGroup.size()));
    return output;
}

bool ReduceScatterPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void ReduceScatterPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t ReduceScatterPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int ReduceScatterPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    size_t size = 1;
    for (int i = 0; i < outputDesc[0].dims.nbDims; ++i)
    {
        size *= outputDesc[0].dims.d[i];
    }

    CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
    NCCLCHECK(ncclReduceScatter(
        inputs[0], outputs[0], size, (*getDtypeMap())[inputDesc[0].type], ncclSum, *mNcclComm, stream));

    return 0;
}

nvinfer1::DataType ReduceScatterPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}


char const* ReduceScatterPlugin::getPluginType() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_NAME;
}

char const* ReduceScatterPlugin::getPluginVersion() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_VERSION;
}

int ReduceScatterPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int ReduceScatterPlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    mNcclComm = getComm(mGroup);
    return 0;
}

void ReduceScatterPlugin::terminate() noexcept {}

size_t ReduceScatterPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType);
}

void ReduceScatterPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void ReduceScatterPlugin::destroy() noexcept
{
    delete this;
}


ReduceScatterPluginCreator::ReduceScatterPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* ReduceScatterPluginCreator::getPluginName() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_NAME;
}

char const* ReduceScatterPluginCreator::getPluginVersion() const noexcept
{
    return REDUCE_SCATTER_PLUGIN_VERSION;
}

PluginFieldCollection const* ReduceScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ReduceScatterPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            auto const* r = static_cast<int const*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new ReduceScatterPlugin(group, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* ReduceScatterPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new ReduceScatterPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
