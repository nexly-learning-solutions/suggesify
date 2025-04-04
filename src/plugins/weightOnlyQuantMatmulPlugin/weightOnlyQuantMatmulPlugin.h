#pragma once

#include "../common/quantization.h"
#include "../src/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "../src/weightOnlyBatchedGemv/kernelLauncher.h"
#include "../plugins/common/gemmPluginProfiler.h"
#include "../plugins/common/plugin.h"

#include <cassert>
#include <cutlass/numeric_types.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cutlass/integer_subbyte.h"

namespace suggestify::plugins
{
enum class WeightTypeId
{
    INT8 = 1,
    INT4 = 2,
};

constexpr int32_t FP16_BITS = 16;
constexpr int32_t INT8_BITS = 8;
constexpr int32_t INT4_BITS = 4;
constexpr int32_t INT8_INT4_RATIO = INT8_BITS / INT4_BITS;
constexpr int32_t FP16_INT4_RATIO = FP16_BITS / INT4_BITS;
constexpr int32_t FP16_INT8_RATIO = FP16_BITS / INT8_BITS;

inline int32_t getWeightTypeMultiplier(WeightTypeId weightTypeId)
{
    return weightTypeId == WeightTypeId::INT8 ? 1 : INT8_INT4_RATIO;
}

using WeightOnlyGemmRunner = suggestify::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyQuantGemmPluginProfiler : public GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig,
                                              WeightOnlyGemmRunnerPtr, GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = suggestify::cutlass_extensions::CutlassGemmConfig;

    void setWeightTypeId(WeightTypeId weightId)
    {
        mWeightTypeId = weightId;
    }

protected:
    void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) override;

    void computeTmpSize(size_t maxM, size_t n, size_t k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    WeightTypeId mWeightTypeId;
};

class WeightOnlyQuantMatmulPlugin : public BasePlugin
{
public:
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
    WeightOnlyQuantMatmulPlugin() = delete;

    WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId, PluginProfilerPtr const& profiler);

    WeightOnlyQuantMatmulPlugin(void const* data, size_t length, PluginProfilerPtr const& profiler);

    ~WeightOnlyQuantMatmulPlugin() override = default;

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
    void init(nvinfer1::DataType type, WeightTypeId weightTypeId);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    WeightTypeId mWeightTypeId;
    bool mCudaKernelEnabled;
    suggestify::kernels::weight_only::KernelType mCudaKernelType;
    int mArch;

    static constexpr int SMALL_M_FAST_PATH = 5;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

class WeightOnlyQuantMatmulPluginCreator : public BaseCreator
{
public:
    WeightOnlyQuantMatmulPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    GemmPluginProfilerManager<WeightOnlyQuantGemmPluginProfiler> gemmPluginProfileManager;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
