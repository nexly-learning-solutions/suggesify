#pragma once

#include "../plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace suggestify::plugins
{

class EagleDecodeDraftTokensPlugin : public BasePlugin
{
public:
    EagleDecodeDraftTokensPlugin(nvinfer1::DataType type, int32_t layerIdx, int32_t numEagleLayers, bool topKSampling);

    EagleDecodeDraftTokensPlugin(void const* data, size_t length);

    ~EagleDecodeDraftTokensPlugin() override = default;

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
    enum class InputIdxEntry : int32_t
    {
        LOGITS = 0,
        RAND_SAMPLE,
        PATHS,
        NUM_VALID_LOGITS,
        USE_DYNAMIC_TREE,
        DYNAMIC_TREE_MAX_TOPK,

        INPUT_DRAFT_TOKEN_IDS,
        INPUT_DRAFT_LENS,

        INPUT_PREV_SCORES,

        INPUT_CURRENT_EXPAND_INDICES,

        INPUT_ALL_LAYERS_SCORES,
        INPUT_ALL_LAYERS_DRAFT_TOKEN_IDS,
        INPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR
    };

    enum class OutputIdxEntry : int32_t
    {
        OUTPUT_DRAFT_TOKEN_IDS = 0,
        OUTPUT_DRAFT_LENS,

        OUTPUT_PATHS,

        OUTPUT_CURRENT_SCORES,

        OUTPUT_NEXT_EXPAND_INDICES,

        OUTPUT_ALL_LAYERS_SCORES,
        OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS,
        OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR
    };

    int32_t getIdx(InputIdxEntry idx) const
    {
        return static_cast<int32_t>(idx);
    }

    int32_t getIdx(OutputIdxEntry idx) const
    {
        return static_cast<int32_t>(idx);
    }

private:
    template <typename T>
    size_t getWorkspaceSizeType(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept;

    template <typename T>
    void enqueueType(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    template <typename T>
    void doTopKSampling(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

private:
    nvinfer1::DataType mDtype;
    int32_t mLayerIdx{-1};
    int32_t mNumEagleLayers{-1};
    bool mTopKSampling;
};

class EagleDecodeDraftTokensPluginCreator : public BaseCreator
{
public:
    EagleDecodeDraftTokensPluginCreator();

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
