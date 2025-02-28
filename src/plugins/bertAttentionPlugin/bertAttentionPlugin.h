#pragma once

#include "../common/cublasMMWrapper.h"
#include "../common/quantization.h"
#include "../src/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "../src/gptKernels.h"
#include "../plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace suggestify::plugins
{

class BertAttentionPlugin : public BasePlugin
{
public:
    BertAttentionPlugin() = delete;

    BertAttentionPlugin(int num_heads, int head_size, float q_scaling,
        suggestify::kernels::ContextFMHAType context_fmha_type, nvinfer1::DataType type,
        bool do_relative_attention = false, int max_distance = 0, bool remove_padding = false);

    BertAttentionPlugin(void const* data, size_t length);

    ~BertAttentionPlugin() override = default;

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

private:
    const std::string mLayerName;

    int mNumHeads;
    int mHeadSize;
    float mQScaling;
    nvinfer1::DataType mType;
    bool mRelativeAttention = false;
    int mMaxDistance = 0;
    bool mRemovePadding = false;

    bool mQKHalfAccum = false;

    bool mEnableContextFMHA = false;
    bool mFMHAForceFP32Acc = false;
    int mSM = suggestify::common::getSMVersion();

    UniqPtrWNullCopy<suggestify::kernels::FusedMHARunnerV2> mFMHARunner;
    UniqPtrWNullCopy<suggestify::common::CublasMMWrapper> mCublasWrapper;
};

class BertAttentionPluginCreator : public BaseCreator
{
public:
    BertAttentionPluginCreator();

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
