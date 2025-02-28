
#pragma once
#include "low_latency_gemm.h"


namespace tkc = suggestify::cutlass_extensions;

namespace suggestify
{
namespace kernels
{
namespace internal_cutlass_kernels
{

class CutlassLowLatencyFp8GemmSwigluRunnerInterface
{
public:
    using ConfigType = LowLatencyCutlassGemmConfig;

    CutlassLowLatencyFp8GemmSwigluRunnerInterface() {}

    virtual ~CutlassLowLatencyFp8GemmSwigluRunnerInterface() {}

    virtual void gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta, float scale_d0, float scale_d1,
        void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetch_ratio,
        ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
        = 0;

    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<ConfigType> getConfigs() const = 0;
};

template <typename T>
class CutlassLowLatencyFp8GemmSwigluRunner : public virtual CutlassLowLatencyFp8GemmSwigluRunnerInterface
{
public:
    CutlassLowLatencyFp8GemmSwigluRunner();
    ~CutlassLowLatencyFp8GemmSwigluRunner() = default;
    void gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta, float scale_d0, float scale_d1,
        void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetech_ratio,
        ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream) override;
    size_t getWorkspaceSize(int const m, int const n, int const k) override;
    std::vector<ConfigType> getConfigs() const override;

private:
    size_t dispatchToArch(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha, float beta, float scale_d0,
        float scale_d1, void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetech_ratio,
        ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream);

    size_t getWorkspaceSizeImpl(int const m, int const n, int const k);
    int mSm;
};

};
};
};
