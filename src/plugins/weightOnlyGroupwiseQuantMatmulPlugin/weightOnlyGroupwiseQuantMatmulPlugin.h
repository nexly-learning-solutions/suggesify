#pragma once

#include "../common/quantization.h"
#include "../src/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "../src/preQuantScaleKernel.h"
#include "../src/weightOnlyBatchedGemv//kernelLauncher.h"
#include "../plugins/common/gemmPluginProfiler.h"
#include "../plugins/common/plugin.h"
#include "../plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

#include <cutlass/numeric_types.h>

#include <cassert>
#include <cuda_runtime.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cutlass/integer_subbyte.h"

namespace suggestify::plugins
{

using WeightOnlyGemmRunner = suggestify::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyGroupwiseQuantGemmPluginProfiler
    : public GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
          GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = suggestify::cutlass_extensions::CutlassGemmConfig;

    void setQuantAlgo(int quantAlgo)
    {
        mQuantAlgo = quantAlgo;
    }

    void setGroupSize(int groupSize)
    {
        mGroupSize = groupSize;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    int mQuantAlgo;
    int mGroupSize;
};

class WeightOnlyGroupwiseQuantMatmulPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;

    WeightOnlyGroupwiseQuantMatmulPlugin() = delete;

    WeightOnlyGroupwiseQuantMatmulPlugin(
        nvinfer1::DataType type, int quant_algo, int group_size, PluginProfilerPtr const& profiler);

    WeightOnlyGroupwiseQuantMatmulPlugin(void const* data, size_t length, PluginProfilerPtr const& profiler);

    ~WeightOnlyGroupwiseQuantMatmulPlugin() override = default;

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
    void init(nvinfer1::DataType type, int quant_algo, int group_size);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGroupwiseGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    bool mCudaKernelEnabled;
    suggestify::kernels::weight_only::KernelType mCudaKernelType;
    int mArch;

    static constexpr int SMALL_M_FAST_PATH = 5;

    int mQuantAlgo;

    int mGroupSize;

    int mPreQuantScaleInputIdx;
    int mWeightInputIdx;
    int mScalesInputIdx;
    int mZerosInputIdx;
    int mBiasesInputIdx;
    int mAlphaInputIdx;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

class WeightOnlyGroupwiseQuantMatmulPluginCreator : public BaseCreator
{
public:
    WeightOnlyGroupwiseQuantMatmulPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
