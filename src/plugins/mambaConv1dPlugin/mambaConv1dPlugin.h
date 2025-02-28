
#ifndef TRT_MAMBA_CONV1D_PLUGIN_H
#define TRT_MAMBA_CONV1D_PLUGIN_H
#include "../src/mambaConv1dKernels.h"
#include "../plugins/common/plugin.h"
#include <cassert>

namespace suggestify::plugins
{


class MambaConv1dPlugin : public BasePlugin
{
public:
    MambaConv1dPlugin(int dim, int dconv, int preStride, int postStride, nvinfer1::DataType type, bool removePadding,
        bool pagedState, bool applySilu);

    MambaConv1dPlugin(void const* data, size_t length);

    ~MambaConv1dPlugin() override = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    template <typename T>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };

private:
    using IndexType = std::int32_t;

    IndexType getInputTensorIdx() const
    {
        return 0;
    };

    IndexType getConvStateIdx() const
    {
        return 1;
    };

    IndexType getWeightIdx() const
    {
        return 2;
    };

    IndexType getBiasIdx() const
    {
        return 3;
    };

    IndexType getHostRequestTypesIdx() const
    {
        return 4;
    };

    IndexType getLastTokenIdsIdx() const
    {
        return 5;
    };

    IndexType getHostContextLengthIdx() const
    {
        return 6;
    };

    IndexType getSlotMappingIdx() const
    {
        return mRemovePadding ? 7 : 6;
    };

    void setMambaConv1dParams(suggestify::kernels::MambaConv1dParamsBase& params,
        const size_t batch, const size_t dim, const size_t maxSeqLen, const size_t dconv, const size_t preStride,
        const size_t postStride,
        void const* inPtr, void const* stateInPtr, void* stateOutPtr, void const* convWeight, void const* convBias,
        void* outPtr, int const* lastTokenIds, int const* stateSlotMapping, bool removePadding, bool applySilu);

private:
    int mDim;
    int mDConv;
    int mPreStride;
    int mPostStride;
    nvinfer1::DataType mType;
    bool mRemovePadding = false;
    bool mPagedState = false;
    bool mApplySilu = true;
};

class MambaConv1dPluginCreator : public BaseCreator
{
public:
    MambaConv1dPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}

#endif
