
#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/gemm_configs.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "fp8_rowwise_gemm.h"
#include "fp8_rowwise_gemm_kernel_template_sm89.h"
#include "fp8_rowwise_gemm_kernel_template_sm90.h"
#include "../common/cudaUtils.h"
#include "../common/quantization.h"
#include "../src/cutlass_kernels/cutlass_heuristic.h"
#include "../src/cutlass_kernels/cutlass_type_conversion.h"

#include <algorithm>
#include <vector>

namespace tk = suggestify::common;
namespace tkc = suggestify::cutlass_extensions;

using namespace cute;

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename Gemm>
size_t typedFp8RowwiseGemmKernelLauncher(Gemm gemm, typename Gemm::Arguments args, void* D, void const* A,
    void const* B, void const* C_bias, char* workspace, size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementT = typename Gemm::ElementA;

    int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
    static int mMaxSmemSize = tk::getMaxSharedMemoryPerBlockOptin();
    if (smem_size > mMaxSmemSize)
    {
        std::string errMsg = "SMEM size exceeds maximum allowed. Required " + std::to_string(smem_size) + ", got "
            + std::to_string(mMaxSmemSize);
        throw std::runtime_error("[nexly Error][fp8RowwiseGemm Runner] " + errMsg);
    }

    if (!A && !B && !D)
    {
        return gemm.get_workspace_size(args);
    }

    if (gemm.get_workspace_size(args) > workspaceBytes)
    {
        std::string errMsg("Requested workspace size insufficient. Required "
            + std::to_string(gemm.get_workspace_size(args)) + ", got " + std::to_string(workspaceBytes));
        throw std::runtime_error("[nexly Error][fp8RowwiseGemm Runner] " + errMsg);
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string errMsg = "fp8RowwiseGemm cutlass kernel not implemented given the params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[nexly Error][fp8RowwiseGemm Runner] " + errMsg);
    }

    auto initStatus = gemm.initialize(args, workspace, stream);
    if (initStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to initialize. Error: " + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[nexly Error][fp8RowwiseGemm Runner] " + errMsg);
    }

    auto runStatus = gemm.run(stream);
    if (runStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to run gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[nexly Error][fp8RowwiseGemm Runner] " + errMsg);
    }
    return gemm.get_workspace_size(args);
}

template <typename Gemm>
typename Gemm::Arguments prepareGemmArgsSm89(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float const* scale_d0, float const* scale_d1,
    tkc::CutlassGemmConfig gemmConfig)
{
    using ElementT = typename Gemm::ElementA;
    using ElementOutput = typename Gemm::ElementD;
    using ElementComputeEpilogue = float;

    int const lda = k;
    int const ldb = k;
    int const ldc = n;

    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        {},
        reinterpret_cast<ElementT const*>(A),
        reinterpret_cast<ElementT const*>(B),
        nullptr,
        nullptr,
        m * k,
        n * k,
        m * n,
        m * n,
        lda,
        ldb,
        ldc,
        ldc);

    args.epilogue = {
        {
            {
                {},
                {reinterpret_cast<ElementComputeEpilogue const*>(scale_d1), ElementComputeEpilogue(0),
                    {_0{}, _1{}, _0{}}},
                {}
            },
            {reinterpret_cast<ElementComputeEpilogue const*>(scale_d0), ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
            {}
        },
        {reinterpret_cast<ElementOutput*>(D), {n, _1{}, _0{}}}};
    return args;
}

template <typename T, typename CtaShape, typename WarpShape, int Stages>
size_t genericFp8RowwiseGemmKernelLauncherSm89(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float const* scale_d0, float const* scale_d1,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    using AccumElementType = float;

    using Gemm = typename DeviceGemmFp8RowwiseSm89<ElementInput, ElementOutput, AccumElementType, CtaShape, WarpShape,
        Stages>::Gemm;
    auto args = prepareGemmArgsSm89<Gemm>(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig);
    return typedFp8RowwiseGemmKernelLauncher(
        Gemm{}, args, D, A, B, C_bias, workspace, workspaceBytes, stream, occupancy);
}

template <typename T, typename CtaShape, typename WarpShape>
size_t dispatchGemmConfigSm89(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float const* scale_d0, float const* scale_d1, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemmConfig.stages)
    {
    case 2:
        return genericFp8RowwiseGemmKernelLauncherSm89<T, CtaShape, WarpShape, 2>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case 3:
        return genericFp8RowwiseGemmKernelLauncherSm89<T, CtaShape, WarpShape, 3>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case 4:
        return genericFp8RowwiseGemmKernelLauncherSm89<T, CtaShape, WarpShape, 4>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmConfigSm89] Config is invalid for "
            "Fp8 Rowwise GEMM.");
        break;
    }
}

template <typename T>
size_t dispatchGemmToCutlassSm89(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float const* scale_d0, float const* scale_d1, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemmConfig.tile_config)
    {

    case tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<32, 128, 64>, cutlass::gemm::GemmShape<32, 32, 64>>(D,
            A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<32, 64, 64>>(D,
            A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>>(D,
            A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(D,
            A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(D,
            A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(
            D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(
            D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<128, 32, 64>>(
            D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(
            D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(
            D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape128x64x128_WarpShape64x32x128:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<128, 64, 128>, cutlass::gemm::GemmShape<64, 32, 128>>(
            D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128:
        return dispatchGemmConfigSm89<T, cutlass::gemm::GemmShape<16, 256, 128>, cutlass::gemm::GemmShape<16, 64, 128>>(
            D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;

    case tkc::CutlassTileConfig::Undefined:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmToCutlassSm89] gemm config undefined.");
        break;
    case tkc::CutlassTileConfig::ChooseWithHeuristic:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmToCutlassSm89] gemm config should have "
            "already been set by heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmToCutlassSm89] Config is invalid for "
            "Fp8 Rowwise GEMM.");
        break;
    }
}

template <typename Gemm>
typename Gemm::Arguments prepareGemmArgsSm90(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float const* scale_d0, float const* scale_d1,
    tkc::CutlassGemmConfig gemmConfig)
{
    using ElementT = typename Gemm::ElementA;
    using ElementOutput = typename Gemm::ElementD;
    using ElementComputeEpilogue = float;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    ElementT const* ptr_A = reinterpret_cast<ElementT const*>(A);
    ElementT const* ptr_B = reinterpret_cast<ElementT const*>(B);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    StrideC stride_C;
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
    typename Gemm::Arguments args
        = {cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1}, {ptr_A, stride_A, ptr_B, stride_B},
            {{},
                nullptr, stride_C, reinterpret_cast<ElementOutput*>(D), stride_D}};
    args.epilogue.thread = {
        {reinterpret_cast<ElementComputeEpilogue*>(const_cast<float*>(scale_d0))},
        {
            {reinterpret_cast<ElementComputeEpilogue*>(const_cast<float*>(scale_d1))}, {},
            {}
        },
        {},
    };
    return args;
}

template <typename T, typename CTAShape, typename ClusterShape>
size_t genericFp8RowwiseGemmKernelLauncherSm90(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float const* scale_d0, float const* scale_d1,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

#ifdef COMPILE_HOPPER_TMA_GEMMS
    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    using AccumElementType = float;
    using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedulerType = void;
    using Gemm = typename DeviceGemmFp8RowwiseSm90<ElementInput, ElementOutput, AccumElementType, CTAShape,
        ClusterShape, MainloopScheduleType, EpilogueScheduleType, TileSchedulerType>::Gemm;
    auto args = prepareGemmArgsSm90<Gemm>(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig);
    return typedFp8RowwiseGemmKernelLauncher(
        Gemm{}, args, D, A, B, C_bias, workspace, workspaceBytes, stream, occupancy);
#else
    throw std::runtime_error(
        "[nexly Error][Fp8RowwiseGemmKernelLauncherSm90] Please recompile with support for hopper by passing "
        "90-real as an arch to build_wheel.py.");
#endif
}

template <typename T, typename CTAShape>
size_t dispatchGemmConfigSm90(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float const* scale_d0, float const* scale_d1, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemmConfig.cluster_shape)
    {
    case tkc::ClusterShape::ClusterShape_1x1x1:
        return genericFp8RowwiseGemmKernelLauncherSm90<T, CTAShape, Shape<_1, _1, _1>>(D, A, B, C_bias, quantOption, m,
            n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x1x1:
        return genericFp8RowwiseGemmKernelLauncherSm90<T, CTAShape, Shape<_2, _1, _1>>(D, A, B, C_bias, quantOption, m,
            n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
        return genericFp8RowwiseGemmKernelLauncherSm90<T, CTAShape, Shape<_1, _2, _1>>(D, A, B, C_bias, quantOption, m,
            n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
        return genericFp8RowwiseGemmKernelLauncherSm90<T, CTAShape, Shape<_2, _2, _1>>(D, A, B, C_bias, quantOption, m,
            n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_1x8x1:
        return genericFp8RowwiseGemmKernelLauncherSm90<T, CTAShape, Shape<_1, _8, _1>>(D, A, B, C_bias, quantOption, m,
            n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_8x1x1:
        return genericFp8RowwiseGemmKernelLauncherSm90<T, CTAShape, Shape<_8, _1, _1>>(D, A, B, C_bias, quantOption, m,
            n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmConfigSm90] Config is invalid for "
            "Fp8 Rowwise GEMM.");
        break;
    }
}

template <typename T>
size_t dispatchGemmToCutlassSm90(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float const* scale_d0, float const* scale_d1, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    constexpr int Ktile = 128 / sizeof(T);
    using _Ktile = Int<Ktile>;
    switch (gemmConfig.tile_config_sm90)
    {
    case tkc::CutlassTileConfigSM90::CtaShape64x16x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _16, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x32x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _32, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x64x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _64, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x128x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _128, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x16x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _16, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x32x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _32, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x64x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _64, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x128x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _128, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::Undefined:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmToCutlassSm90] gemm config undefined.");
        break;
    case tkc::CutlassTileConfigSM90::ChooseWithHeuristic:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmToCutlassSm90] gemm config should have "
            "already been set by heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][dispatchGemmToCutlassSm90] Config is invalid for "
            "Fp8 Rowwise GEMM.");
        break;
    }
}

template <typename T>
CutlassFp8RowwiseGemmRunner<T>::CutlassFp8RowwiseGemmRunner()
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    mSm = tk::getSMVersion();
}

template <typename T>
CutlassFp8RowwiseGemmRunner<T>::~CutlassFp8RowwiseGemmRunner()
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T>
size_t CutlassFp8RowwiseGemmRunner<T>::dispatchToArch(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float const* scale_d0, float const* scale_d1,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    if (mSm == 90)
    {
        return dispatchGemmToCutlassSm90<T>(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig,
            workspace, workspaceBytes, stream, occupancy);
    }
    else if (mSm == 89)
    {
        return dispatchGemmToCutlassSm89<T>(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig,
            workspace, workspaceBytes, stream, occupancy);
    }
    else
    {
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS "
            "Fp8 Rowwise GEMM");
    }
    return 0;
}

template <typename T>
void CutlassFp8RowwiseGemmRunner<T>::gemm(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float const* scale_d0, float const* scale_d1,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatchToArch(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, gemmConfig, workspace, workspaceBytes,
        stream, occupancy);
}

template <typename T>
std::vector<tkc::CutlassGemmConfig> CutlassFp8RowwiseGemmRunner<T>::getConfigs() const
{
    using tkc::CutlassTileConfig;
    using tkc::CutlassGemmConfig;
    using tkc::SplitKStyle;

    std::vector<CutlassGemmConfig> candidateConfigs;
    if (mSm == 90)
    {
        tkc::CutlassGemmConfig::CandidateConfigTypeParam config_type_param
            = tkc::CutlassGemmConfig::CandidateConfigTypeParam::HOPPER;
        std::vector<CutlassGemmConfig> commonConfigs = get_candidate_configs(mSm, 2, config_type_param);
        candidateConfigs.insert(candidateConfigs.end(), commonConfigs.begin(), commonConfigs.end());
        candidateConfigs.erase(std::remove_if(candidateConfigs.begin(), candidateConfigs.end(),
                                   [](auto const& config)
                                   {
                                       return config.tile_config_sm90 == tkc::CutlassTileConfigSM90::CtaShape64x256x128B
                                           || config.tile_config_sm90
                                           == tkc::CutlassTileConfigSM90::CtaShape128x256x128B;
                                   }),
            candidateConfigs.end());
        std::vector<tkc::CutlassTileConfigSM90> tilesSm90
            = {tkc::CutlassTileConfigSM90::CtaShape64x16x128B, tkc::CutlassTileConfigSM90::CtaShape64x32x128B,
                tkc::CutlassTileConfigSM90::CtaShape64x64x128B, tkc::CutlassTileConfigSM90::CtaShape64x128x128B,
                tkc::CutlassTileConfigSM90::CtaShape128x16x128B, tkc::CutlassTileConfigSM90::CtaShape128x32x128B,
                tkc::CutlassTileConfigSM90::CtaShape128x64x128B, tkc::CutlassTileConfigSM90::CtaShape128x128x128B};
        for (auto const& tile_config : tilesSm90)
        {
            {
                CutlassGemmConfig config(tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                    tkc::ClusterShape::ClusterShape_1x8x1);
                candidateConfigs.push_back(config);
            }
            {
                CutlassGemmConfig config(tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                    tkc::ClusterShape::ClusterShape_8x1x1);
                candidateConfigs.push_back(config);
            }
        }
    }
    else if (mSm == 89)
    {
        tkc::CutlassGemmConfig::CandidateConfigTypeParam config_type_param
            = tkc::CutlassGemmConfig::CandidateConfigTypeParam::FP8_ONLY;
        std::vector<CutlassGemmConfig> commonConfigs = get_candidate_configs(mSm, 1, config_type_param);
        candidateConfigs.insert(candidateConfigs.end(), commonConfigs.begin(), commonConfigs.end());
    }
    else
    {
        throw std::runtime_error(
            "[nexly Error][CutlassFp8RowwiseGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS "
            "Fp8 Rowwise GEMM");
    }
    return candidateConfigs;
}

template <typename T>
size_t CutlassFp8RowwiseGemmRunner<T>::getWorkspaceSizeImpl(int const m, int const n, int const k)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t workspace_size = 0;
    auto gemmConfigs = CutlassFp8RowwiseGemmRunner<T>{}.getConfigs();
    for (auto const& gemmConfig : gemmConfigs)
    {
        try
        {
            size_t curr_workspace_size = CutlassFp8RowwiseGemmRunner<T>::dispatchToArch(nullptr, nullptr, nullptr,
                nullptr, tk::QuantMode{}, m, n, k, nullptr, nullptr, gemmConfig, nullptr, 0, 0);
            workspace_size = std::max(workspace_size, curr_workspace_size);
        }
        catch (std::runtime_error& e)
        {
            continue;
        }
    }

    return workspace_size;
}

template <typename T>
size_t CutlassFp8RowwiseGemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

    using MNK = std::tuple<int, int, int>;

    struct MNKHash
    {
        size_t operator()(const MNK& mnk) const
        {
            auto h1 = std::hash<int>{}(std::get<0>(mnk));
            auto h2 = std::hash<int>{}(std::get<1>(mnk));
            auto h3 = std::hash<int>{}(std::get<2>(mnk));
            return h1 ^ h2 ^ h3;
        }
    };

    static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

    size_t workspace_size = 0;
    if (workspace_hashmap.find(std::make_tuple(m, n, k)) == workspace_hashmap.end())
    {
        workspace_size = CutlassFp8RowwiseGemmRunner<T>::getWorkspaceSizeImpl(m, n, k);
        workspace_hashmap[std::make_tuple(m, n, k)] = workspace_size;
    }
    else
    {
        workspace_size = workspace_hashmap[std::make_tuple(m, n, k)];
    }
    return workspace_size;
}

}
}
}
