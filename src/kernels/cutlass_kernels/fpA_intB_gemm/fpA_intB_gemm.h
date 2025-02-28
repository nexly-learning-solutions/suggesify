
#pragma once

#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/weight_only_quant_op.h"
#include <cuda_runtime_api.h>
#include <vector>

namespace tkc = suggestify::cutlass_extensions;

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

enum class ActivationType
{
    Gelu,
    Relu,
    Silu,
    Identity,
    InvalidType
};


class CutlassFpAIntBGemmRunnerInterface
{
public:
    CutlassFpAIntBGemmRunnerInterface() {}

    virtual ~CutlassFpAIntBGemmRunnerInterface() {}

    virtual void gemm(void const* A, void const* B, void const* weight_scales, void* C, int m, int n, int k,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    virtual void gemm(void const* A, void const* B, void const* weight_scales, float const alpha, void* C, int m, int n,
        int k, tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream)
        = 0;

    virtual void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, void* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    virtual void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, float const alpha, void* C, int m, int n, int k, int const group_size,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
        = 0;

    virtual size_t getWorkspaceSize(int const m, int const n, int const k) = 0;

    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

protected:
    static constexpr int SPLIT_K_LIMIT = 7;
    static constexpr int MIN_M_TILE = 16;
    static constexpr int MIN_N_TILE = 64;

    static constexpr int MAX_M_TILE_SM90 = 128;
    static constexpr int MAX_N_TILE_SM90 = 256;
};

template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp,
    typename ScaleZeroType = ActivationType, typename BiasType = ActivationType, typename OutputType = ActivationType>
class CutlassFpAIntBGemmRunner : public virtual CutlassFpAIntBGemmRunnerInterface
{
public:
    CutlassFpAIntBGemmRunner();
    ~CutlassFpAIntBGemmRunner();

    void gemm(void const* A, void const* B, void const* weight_scales, void* C, int m, int n, int k,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;

    void gemm(void const* A, void const* B, void const* weight_scales, float const alpha, void* C, int m, int n, int k,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;

    void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, void* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig,
        char* workspace_ptr, const size_t workspace_bytes, cudaStream_t stream) override;

    void gemm(void const* A, void const* B, void const* weight_scales, void const* weight_zero_points,
        void const* biases, float const alpha, void* C, int m, int n, int k, int const group_size,
        tkc::CutlassGemmConfig gemmConfig, char* workspace_ptr, const size_t workspace_bytes,
        cudaStream_t stream) override;



    size_t getWorkspaceSize(int const m, int const n, int const k) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

private:
    template <typename EpilogueTag>
    void dispatch_to_arch(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
        ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
        int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace_ptr,
        const size_t workspace_bytes, cudaStream_t stream, int* occupancy = nullptr);

private:
    int sm_;
    int multi_processor_count_;
};

}
}
}
