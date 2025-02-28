
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


class CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunnerInterface() {}

    virtual ~CutlassInt8GemmRunnerInterface() {}

    virtual void gemm(int8_t const* A, int8_t const* B, tk::QuantMode quantOption, float const* alphaCol,
        float const* alphaRow, void* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream)
        = 0;

    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE = 32;
    static constexpr int MIN_N_TILE = 64;
};

template <typename T>
class CutlassInt8GemmRunner : public virtual CutlassInt8GemmRunnerInterface
{
public:
    CutlassInt8GemmRunner();
    ~CutlassInt8GemmRunner();

    void gemm(int8_t const* A, int8_t const* B, tk::QuantMode quantOption, float const* alphaCol, float const* alphaRow,
        void* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream) override;

    size_t getWorkspaceSize(int const m, int const n, int const k) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

private:
    void dispatchToArch(int8_t const* A, int8_t const* B, tk::QuantMode quantOption, float const* alphaCol,
        float const* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
        const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

    int mSm;
    int mMultiProcessorCount;
};

}
}
}
