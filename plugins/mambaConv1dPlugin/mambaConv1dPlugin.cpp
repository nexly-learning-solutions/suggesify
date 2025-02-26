
#include "mambaConv1dPlugin.h"
#include "../common/assert.h"
#include <algorithm>

using namespace nvinfer1;
using namespace suggestify::kernels;
using namespace suggestify::common;
using suggestify::plugins::MambaConv1dPluginCreator;
using suggestify::plugins::MambaConv1dPlugin;

static char const* MAMBA_CONV1D_PLUGIN_VERSION{"1"};
static char const* MAMBA_CONV1D_PLUGIN_NAME{"MambaConv1d"};

PluginFieldCollection MambaConv1dPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> MambaConv1dPluginCreator::mPluginAttributes;

MambaConv1dPlugin::MambaConv1dPlugin(int dim, int dconv, int preStride, int postStride, nvinfer1::DataType type,
    bool removePadding, bool pagedState, bool applySilu)
    : mDim(dim)
    , mDConv(dconv)
    , mPreStride(preStride)
    , mPostStride(postStride)
    , mType(type)
    , mRemovePadding(removePadding)
    , mPagedState(pagedState)
    , mApplySilu(applySilu)
{
    CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
    CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

MambaConv1dPlugin::MambaConv1dPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDim);
    read(d, mDConv);
    read(d, mPreStride);
    read(d, mPostStride);
    read(d, mType);
    read(d, mRemovePadding);
    read(d, mPagedState);
    read(d, mApplySilu);
    CHECK(d == a + length);
    CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
    CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

nvinfer1::IPluginV2DynamicExt* MambaConv1dPlugin::clone() const noexcept
{
    auto* plugin
        = new MambaConv1dPlugin(mDim, mDConv, mPreStride, mPostStride, mType, mRemovePadding, mPagedState, mApplySilu);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs MambaConv1dPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        auto ret = inputs[getInputTensorIdx()];
        ret.d[mRemovePadding ? 1 : 2] = exprBuilder.constant(mDim);
        return ret;
    }
    return inputs[getConvStateIdx()];
}

bool MambaConv1dPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getHostRequestTypesIdx() || pos == getLastTokenIdsIdx()
        || (mRemovePadding && pos == getHostContextLengthIdx()) || (mPagedState && pos == getSlotMappingIdx()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (mPagedState && pos == getConvStateIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void MambaConv1dPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t MambaConv1dPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void MambaConv1dPlugin::setMambaConv1dParams(suggestify::kernels::MambaConv1dParamsBase& params, const size_t batch,
    const size_t dim, const size_t maxSeqLen, const size_t dconv, const size_t preStride, const size_t postStride,
    void const* inPtr, void const* stateInPtr, void* stateOutPtr, void const* convWeight, void const* convBias,
    void* outPtr, int const* lastTokenIds, int const* stateSlotMapping, bool removePadding, bool applySilu)
{
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.max_seqlen = maxSeqLen;
    params.dconv = dconv;
    params.pre_stride = preStride;
    params.post_stride = postStride;

    params.remove_padding = removePadding;
    params.apply_silu = applySilu;

    params.in_ptr = const_cast<void*>(inPtr);
    params.state_in_ptr = const_cast<void*>(stateInPtr);
    params.state_out_ptr = stateOutPtr;
    params.weight_ptr = const_cast<void*>(convWeight);
    params.bias_ptr = const_cast<void*>(convBias);
    params.out_ptr = outPtr;
    params.last_token_ids_ptr = lastTokenIds;
    params.state_slot_mapping_ptr = stateSlotMapping;
}

template <typename T>
int MambaConv1dPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    auto const batchSize = inputDesc[getHostRequestTypesIdx()].dims.d[0];
    int maxSeqLen;
    if (mRemovePadding)
    {
        int const* host_context_length = static_cast<int const*>(inputs[getHostContextLengthIdx()]);
        maxSeqLen = *std::max_element(host_context_length, host_context_length + batchSize);
    }
    else
    {
        maxSeqLen = inputDesc[getInputTensorIdx()].dims.d[1];
    }

    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    MambaConv1dParamsBase mambaConv1dParams;

    int const* slotMapping = mPagedState ? static_cast<int const*>(inputs[getSlotMappingIdx()]) : nullptr;
    void* stateInPtr = mPagedState ? *reinterpret_cast<void**>(const_cast<void*>(inputs[getConvStateIdx()]))
                                   : const_cast<void*>(inputs[getConvStateIdx()]);
    void* stateOutPtr
        = mPagedState ? *reinterpret_cast<void**>(const_cast<void*>(inputs[getConvStateIdx()])) : outputs[1];

    setMambaConv1dParams(mambaConv1dParams, batchSize, mDim, maxSeqLen, mDConv, mPreStride, mPostStride,
        inputs[getInputTensorIdx()], stateInPtr, stateOutPtr, inputs[getWeightIdx()], inputs[getBiasIdx()], outputs[0],
        static_cast<int const*>(inputs[getLastTokenIdsIdx()]), slotMapping, mRemovePadding, mApplySilu);

    if (reqTypes[0] == RequestType::kCONTEXT)
    {
        invokeMambaConv1dContext<T>(mambaConv1dParams, stream);
    }
    else if (reqTypes[0] == RequestType::kGENERATION)
    {
        invokeMambaConv1dGeneration<T>(mambaConv1dParams, stream);
    }
    sync_check_cuda_error();
    return 0;
}

int MambaConv1dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    if (mType == DataType::kHALF)
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

nvinfer1::DataType MambaConv1dPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[getInputTensorIdx()];
}


char const* MambaConv1dPlugin::getPluginType() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_NAME;
}

char const* MambaConv1dPlugin::getPluginVersion() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_VERSION;
}

int MambaConv1dPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int MambaConv1dPlugin::initialize() noexcept
{
    return 0;
}

void MambaConv1dPlugin::terminate() noexcept {}

size_t MambaConv1dPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDim) + sizeof(mDConv) + sizeof(mPreStride) + sizeof(mPostStride) + sizeof(mType)
        + sizeof(mRemovePadding) + sizeof(mPagedState) + sizeof(mApplySilu);
}

void MambaConv1dPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDim);
    write(d, mDConv);
    write(d, mPreStride);
    write(d, mPostStride);
    write(d, mType);
    write(d, mRemovePadding);
    write(d, mPagedState);
    write(d, mApplySilu);
    assert(d == a + getSerializationSize());
}

void MambaConv1dPlugin::destroy() noexcept
{
    delete this;
}


MambaConv1dPluginCreator::MambaConv1dPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dim", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("dconv", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("pre_stride", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("post_stride", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("paged_state", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("apply_silu", nullptr, PluginFieldType::kINT8, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* MambaConv1dPluginCreator::getPluginName() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_NAME;
}

char const* MambaConv1dPluginCreator::getPluginVersion() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_VERSION;
}

PluginFieldCollection const* MambaConv1dPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* MambaConv1dPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int dim, dconv, pre_stride, post_stride;
    bool removePadding;
    bool pagedState;
    bool applySilu;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "dim"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            dim = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dconv"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            dconv = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "pre_stride"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            pre_stride = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "post_stride"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            post_stride = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_input_padding"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT8);
            removePadding = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "paged_state"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT8);
            pagedState = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "apply_silu"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT8);
            applySilu = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj
            = new MambaConv1dPlugin(dim, dconv, pre_stride, post_stride, type, removePadding, pagedState, applySilu);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* MambaConv1dPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new MambaConv1dPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
