#pragma once

#include "../src/cutlass_kernels/fused_gated_gemm/fused_gated_gemm.h"
#include "../plugins/common/gemmPluginProfiler.h"
#include "../plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace suggestify::plugins
{

using GemmSwigluRunnerPtr
    = std::shared_ptr<suggestify::kernels::cutlass_kernels::CutlassFusedGatedGemmRunnerInterface>;

class GemmSwigluPluginProfiler : public GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig,
                                     GemmSwigluRunnerPtr, GemmIdCore, GemmIdCoreHash>

{
public:
    using Config = suggestify::cutlass_extensions::CutlassGemmConfig;

    void setQuantMode(suggestify::common::QuantMode const& quantMode);

    virtual int getMaxProfileM() const override;

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;


    std::vector<Config> getTactics(int m, int n, int k) const override;

    void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream) override;

private:
    size_t getBytePerElement(nvinfer1::DataType type);

    suggestify::common::QuantMode mQuantMode;
};

class GemmSwigluPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<GemmSwigluPluginProfiler>;

    GemmSwigluPlugin() = delete;

    GemmSwigluPlugin(suggestify::common::QuantMode quantMode, nvinfer1::DataType type, bool hasBias, float scale_d0,
        float scale_d1, float scale_output, PluginProfilerPtr const& pluginProfiler);

    GemmSwigluPlugin(void const* data, size_t length, PluginProfilerPtr const& profiler);

    ~GemmSwigluPlugin() override = default;

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
    void init(nvinfer1::DataType type);

    void configGemm();

private:
    const std::string mLayerName;

    GemmSwigluRunnerPtr mGemmRunner;
    suggestify::common::QuantMode mQuantMode;
    size_t mWorkspaceMaxSize;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;

    nvinfer1::DataType mType;
    bool mHasBias;
    float mScaleD0;
    float mScaleD1;
    float mScaleOutput;
};

class GemmSwigluPluginCreator : public BaseCreator
{
public:
    GemmSwigluPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<GemmSwigluPluginProfiler> mGemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
