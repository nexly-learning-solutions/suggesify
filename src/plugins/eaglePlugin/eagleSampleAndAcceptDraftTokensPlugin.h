#pragma once

#include "../plugins/common/plugin.h"

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace suggestify::plugins
{

class EagleSampleAndAcceptDraftTokensPlugin : public BasePlugin
{
public:
    EagleSampleAndAcceptDraftTokensPlugin(nvinfer1::DataType type);

    EagleSampleAndAcceptDraftTokensPlugin(void const* data, size_t length);

    ~EagleSampleAndAcceptDraftTokensPlugin() override = default;

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
        DRAFT_TOKEN_IDS,
        DRAFT_LENS,
        TEMPERATURE,
        RAND_VALIDATION,
        POSTERIOR_ALPHA,
        POSTERIOR_THRESHOLD,
        PATHS,
        GREEDY_SAMPLING
    };

    enum class OutputIdxEntry : int32_t
    {
        ACCEPTED_TOKENS = 0,
        ACCEPTED_LENS,
        BEST_ACCEPTED_PATHS,
        NEXT_DRAFT_TOKEN_IDS,
        NEXT_DRAFT_LENS,
        NEXT_DRAFT_PATHS,
        HIDDEN_SIZE_BATCH_LEVEL_STARTS,
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
    void samplePrimeHeadTokens(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept;

    template <typename T>
    void doTypicalAcceptance(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    template <typename T>
    void acceptDraftTokens(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    template <typename T>
    void enqueueType(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

private:
    nvinfer1::DataType mDtype;
    int32_t mSmCnt{0};
};

class EagleSampleAndAcceptDraftTokensPluginCreator : public BaseCreator
{
public:
    EagleSampleAndAcceptDraftTokensPluginCreator();

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
