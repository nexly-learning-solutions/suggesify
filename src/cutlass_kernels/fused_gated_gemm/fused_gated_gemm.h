
#pragma once

#include "cutlass_extensions/gemm_configs.h"
#include "../common/quantization.h"

#include <cuda_runtime_api.h>
#include <vector>

namespace tk = suggestify::common;
namespace tkc = suggestify::cutlass_extensions;

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{


class CutlassFusedGatedGemmRunnerInterface
{
public:
    CutlassFusedGatedGemmRunnerInterface() {}

    virtual ~CutlassFusedGatedGemmRunnerInterface() {}

    virtual void gemm(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption, int m,
        int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
        = 0;

    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;
};

template <typename T>
class CutlassFusedGatedGemmRunner : public virtual CutlassFusedGatedGemmRunnerInterface
{
public:
    CutlassFusedGatedGemmRunner();
    ~CutlassFusedGatedGemmRunner();

    void gemm(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption, int m, int n, int k,
        float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig, char* workspace,
        size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr) override;

    size_t getWorkspaceSize(int const m, int const n, int const k) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

private:
    size_t dispatchToArch(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption, int m,
        int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

    size_t getWorkspaceSizeImpl(int const m, int const n, int const k);

    int mSm;
};

}
}
}
