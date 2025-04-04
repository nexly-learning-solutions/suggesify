
#include "cudaStreamPlugin.h"
#include "../runtime/iBuffer.h"

#include <cassert>

using namespace nvinfer1;
using suggestify::plugins::CudaStreamPluginCreator;
using suggestify::plugins::CudaStreamPlugin;

static char const* CUDA_STREAM_PLUGIN_VERSION{"1"};
static char const* CUDA_STREAM_PLUGIN_NAME{"CudaStream"};
PluginFieldCollection CudaStreamPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CudaStreamPluginCreator::mPluginAttributes;

CudaStreamPlugin::CudaStreamPlugin(int sideStreamId, int nbInputs, nvinfer1::DataType type)
    : mSideStreamId(sideStreamId)
    , mNbInputs(nbInputs)
    , mType(type)
{
    init();
}

CudaStreamPlugin::CudaStreamPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mSideStreamId);
    read(d, mNbInputs);
    read(d, mType);

    init();

    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different nexly version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

CudaStreamPlugin::CudaStreamPlugin(CudaStreamPlugin const& other)
    : mSideStreamId(other.mSideStreamId)
    , mNbInputs(other.mNbInputs)
    , mType(other.mType)
{
    init();
}

void CudaStreamPlugin::init()
{
    mSideStreamPtr = nullptr;
}

nvinfer1::IPluginV2DynamicExt* CudaStreamPlugin::clone() const noexcept
{
    auto* plugin = new CudaStreamPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs CudaStreamPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    assert(outputIndex == 0);
    return inputs[outputIndex];
}

bool CudaStreamPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    CHECK_WITH_INFO(nbInputs == mNbInputs, "CudaStreamPlugin only accepts mNbInputs inputs");
    CHECK_WITH_INFO(nbOutputs == 1, "CudaStreamPlugin only accepts 1 output");

    auto const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (pos > 0 && pos < nbInputs)
    {
        return true;
    }
    return desc.type == mType;
}

void CudaStreamPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t CudaStreamPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int CudaStreamPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (!mSideStreamPtr)
    {
        auto const resource_name = nvinfer1::pluginInternal::SideStream::getResourceKey(mSideStreamId);
        nvinfer1::pluginInternal::SideStream side_stream{};
        mSideStreamPtr = reinterpret_cast<nvinfer1::pluginInternal::SideStream*>(
            getPluginRegistry()->acquirePluginResource(resource_name, &side_stream));
    }
    mSideStreamPtr->waitSideStreamOnMainStream(stream);
    size_t count = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        count *= inputDesc[0].dims.d[i];
    }
    count *= suggestify::runtime::BufferDataType(inputDesc[0].type).getSize();
    CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], count, cudaMemcpyDeviceToDevice, stream));

    return 0;
}

nvinfer1::DataType CudaStreamPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return mType;
}


char const* CudaStreamPlugin::getPluginType() const noexcept
{
    return CUDA_STREAM_PLUGIN_NAME;
}

char const* CudaStreamPlugin::getPluginVersion() const noexcept
{
    return CUDA_STREAM_PLUGIN_VERSION;
}

int CudaStreamPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int CudaStreamPlugin::initialize() noexcept
{
    return 0;
}

void CudaStreamPlugin::terminate() noexcept
{
    if (mSideStreamPtr)
    {
        auto const resource_name = nvinfer1::pluginInternal::SideStream::getResourceKey(mSideStreamId);
        getPluginRegistry()->releasePluginResource(resource_name);
        mSideStreamPtr = nullptr;
    }
}

size_t CudaStreamPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSideStreamId) + sizeof(mNbInputs) + sizeof(mType);
}

void CudaStreamPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mSideStreamId);
    write(d, mNbInputs);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void CudaStreamPlugin::destroy() noexcept
{
    delete this;
}


CudaStreamPluginCreator::CudaStreamPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("side_stream_id", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("num_inputs", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CudaStreamPluginCreator::getPluginName() const noexcept
{
    return CUDA_STREAM_PLUGIN_NAME;
}

char const* CudaStreamPluginCreator::getPluginVersion() const noexcept
{
    return CUDA_STREAM_PLUGIN_VERSION;
}

PluginFieldCollection const* CudaStreamPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* CudaStreamPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int sideStreamId;
    int nbInputs;
    int type;

    struct MapPair
    {
        char const* key;
        int& field;
        bool optional = false;
        bool set = false;
    };

    std::array input_map{
        MapPair{"side_stream_id", std::ref(sideStreamId)},
        MapPair{"num_inputs", std::ref(nbInputs)},
        MapPair{"type_id", std::ref(type)},
    };
    bool typeSet = false;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        for (auto& item : input_map)
        {
            if (!strcmp(item.key, attrName))
            {
                CHECK(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                CHECK_WITH_INFO(!item.set, "Parameter %s was set twice", item.key);
                item.field = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
                item.set = true;
            }
        }
    }

    for (auto& item : input_map)
    {
        CHECK_WITH_INFO(item.set || item.optional, "Parameter %s is required but not set", item.key);
    }

    try
    {
        auto* obj = new CudaStreamPlugin(sideStreamId, nbInputs, static_cast<nvinfer1::DataType>(type));
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* CudaStreamPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new CudaStreamPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
