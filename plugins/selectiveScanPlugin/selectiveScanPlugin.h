
#ifndef TRT_SELECTIVE_SCAN_PLUGIN_H
#define TRT_SELECTIVE_SCAN_PLUGIN_H
#include "../src/selectiveScan.h"
#include "../plugins/common/plugin.h"
#include <cassert>

namespace suggestify::plugins
{


class SelectiveScanPlugin : public BasePlugin
{
public:
    SelectiveScanPlugin(int dim, int dstate, int dtRank, int nHeads, int nGroups, int chunkSize, bool deltaSoftplus,
        nvinfer1::DataType type, bool removePadding, bool pagedState, bool zEnabled, bool isMamba2);

    SelectiveScanPlugin(void const* data, size_t length);

    ~SelectiveScanPlugin() override = default;

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

    IndexType getStateIdx() const
    {
        return 1;
    };

    IndexType getDeltaIdx() const
    {
        return 2;
    };

    IndexType getDeltaBiasIdx() const
    {
        return 3;
    };

    IndexType getAIdx() const
    {
        return 4;
    };

    IndexType getBCIdx() const
    {
        return 5;
    };

    IndexType getDIdx() const
    {
        return 6;
    };

    IndexType getHostRequestTypesIdx() const
    {
        return 7;
    };

    IndexType getLastTokenIdsIdx() const
    {
        return 8;
    };

    IndexType getHostContextLengthIdx() const
    {
        if (mRemovePadding)
            return 9;
        else
            return 8;
    };

    IndexType getSlotMappingIdx() const
    {
        if (mPagedState)
            return getHostContextLengthIdx() + 1;
        else
            return getHostContextLengthIdx();
    };

    IndexType getZIdx() const
    {
        if (mZEnabled)
            return getSlotMappingIdx() + 1;
        else
            return getSlotMappingIdx();
    };

    void setSSMParams(suggestify::kernels::SSMParamsBase& params,
        const size_t batch, const size_t dim, const size_t maxSeqLen, const size_t numTokens, const size_t dstate,
        const size_t dtRank, const size_t nHeads, const size_t nGroups, const size_t chunkSize,
        void* statePtr, void const* x, void const* delta, void const* deltaBias, void const* A, void const* BC,
        void const* D, void const* z, void* osPtr, void* stPtr, void* dcPtr, void* dAPtr, void* cbPtr, void* descs,
        int const* lastTokenIds, int const* slotMapping, void* out, bool deltaSoftplus, bool removePadding);

private:
    int mDim;
    int mDState;
    int mDtRank;
    int mNHeads;
    int mNGroups;
    int mChunkSize;
    bool mDeltaSoftplus;
    nvinfer1::DataType mType;
    bool mRemovePadding = false;
    bool mPagedState = false;
    bool mZEnabled = true;
    bool mIsMamba2 = false;
    std::shared_ptr<suggestify::common::CUDADriverWrapper> mDriver;
};

class SelectiveScanPluginCreator : public BaseCreator
{
public:
    SelectiveScanPluginCreator();

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
