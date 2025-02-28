
#ifndef TRT_LRU_PLUGIN_H
#define TRT_LRU_PLUGIN_H
#include "../src/lruKernel.h"
#include "../plugins/common/plugin.h"
#include <cassert>

namespace suggestify::plugins
{


class lruPlugin : public BasePlugin
{
public:
    lruPlugin(int dim, int block_size, nvinfer1::DataType type, bool removePadding, bool pagedState, bool yEnabled,
        bool yBiasEnabled, bool fuseGateEnabled, bool gateBiasEnabled);

    lruPlugin(void const* data, size_t length);

    ~lruPlugin() override = default;

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

    IndexType getXIdx() const
    {
        return 0;
    };

    IndexType getAIdx() const
    {
        return 1;
    };

    IndexType getStateIdx() const
    {
        return 2;
    };

    IndexType getHostRequestTypesIdx() const
    {
        return 3;
    };

    IndexType getLastTokenIdsIdx() const
    {
        return 4;
    };

    IndexType getSlotMappingIdx() const
    {
        if (mPagedState)
            return 5;
        else
            return 4;
    };

    IndexType getYIdx() const
    {
        if (mYEnabled)
            return getSlotMappingIdx() + 1;
        else
            return getSlotMappingIdx();
    };

    IndexType getYBiasIdx() const
    {
        if (mYBiasEnabled)
            return getYIdx() + 1;
        else
            return getYIdx();
    };

    IndexType getGateIdx() const
    {
        if (mFuseGateEnabled)
            return getYBiasIdx() + 1;
        else
            return getYBiasIdx();
    };

    IndexType getGateBiasIdx() const
    {
        if (mFuseGateEnabled && mGateBiasEnabled)
            return getGateIdx() + 1;
        else
            return getGateIdx();
    };

    IndexType getGateXIdx() const
    {
        if (mFuseGateEnabled)
            return getGateBiasIdx();
        else
            return getGateBiasIdx() + 1;
    };

    IndexType getGateAIdx() const
    {
        if (mFuseGateEnabled)
            return getGateXIdx();
        else
            return getGateXIdx() + 1;
    };

    IndexType getGateXBiasIdx() const
    {
        if (!mFuseGateEnabled && mGateBiasEnabled)
            return getGateAIdx() + 1;
        else
            return getGateAIdx();
    };

    IndexType getGateABiasIdx() const
    {
        if (!mFuseGateEnabled && mGateBiasEnabled)
            return getGateXBiasIdx() + 1;
        else
            return getGateXBiasIdx();
    };

    static void setLruParams(suggestify::kernels::lruParams& params,
        const size_t batch, const size_t dim, const size_t block_size, const size_t maxSeqLen,
        void* statePtr, void const* x, void const* gate, void const* gate_bias, void const* gate_x,
        void const* gate_x_bias, void const* gate_a, void const* gate_a_bias, void const* y, void const* y_bias,
        void const* A, int const* lastTokenIds, int const* slotMapping, void* out, bool removePadding);

private:
    int mDim;
    int mBlockSize;
    nvinfer1::DataType mType;
    bool mRemovePadding = false;
    bool mPagedState = false;
    bool mYEnabled = false;
    bool mYBiasEnabled = false;
    bool mFuseGateEnabled = false;
    bool mGateBiasEnabled = false;
};

class lruPluginCreator : public BaseCreator
{
public:
    lruPluginCreator();

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
