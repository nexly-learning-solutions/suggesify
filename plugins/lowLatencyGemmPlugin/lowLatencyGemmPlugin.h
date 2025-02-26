
#pragma once

#include "../src/internal_cutlass_kernels/include/low_latency_gemm.h"

#include "../plugins/common/gemmPluginProfiler.h"
#include "../plugins/common/plugin.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace suggestify::plugins
{

using LowLatencyGemmRunnerPtr
    = std::shared_ptr<suggestify::kernels::internal_cutlass_kernels::CutlassLowLatencyFp8GemmRunnerInterface>;

class LowLatencyGemmPluginProfiler
    : public GemmPluginProfiler<
          suggestify::kernels::internal_cutlass_kernels::CutlassLowLatencyFp8GemmRunnerInterface::ConfigType,
          LowLatencyGemmRunnerPtr, GemmIdCore, GemmIdCoreHash>
{

public:
    using Config = suggestify::kernels::internal_cutlass_kernels::CutlassLowLatencyFp8GemmRunnerInterface::ConfigType;

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;
};

class LowLatencyGemmPlugin : public BasePlugin
{

public:
    using PluginProfilerPtr = std::shared_ptr<LowLatencyGemmPluginProfiler>;

    LowLatencyGemmPlugin() = delete;

    LowLatencyGemmPlugin(nvinfer1::DataType type, float alpha, PluginProfilerPtr const& pluginProfiler);

    LowLatencyGemmPlugin(void const* data, size_t length, PluginProfilerPtr const& pluginProfiler);
    ~LowLatencyGemmPlugin() override = default;

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
    std::string const mLayerName;

    LowLatencyGemmRunnerPtr m_lowLatencyGemmRunner;
    size_t m_workspaceMaxSize;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;

    nvinfer1::DataType mType;
    float mAplha{1.0F};
};

class LowLatencyGemmPluginCreator : public BaseCreator
{
public:
    LowLatencyGemmPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<LowLatencyGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
