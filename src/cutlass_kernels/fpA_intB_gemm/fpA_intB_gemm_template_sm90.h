

#pragma once

#include "cute/numeric/integral_constant.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"
#include "../src/cutlass_kernels/cutlass_heuristic.h"

#include "../src/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.h"

namespace tk = suggestify::common;
namespace tkc = suggestify::cutlass_extensions;

using namespace cute;

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType>
void sm90_dispatch_epilogue_schedules(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy = nullptr)
{

    LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemm_config.epilogue_schedule)
    {
    case tkc::EpilogueScheduleType::AUTO:
        using EpilogueScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
            cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;
        sm90_generic_mixed_gemm_kernelLauncher<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, CTAShape, ClusterShape, MainloopScheduleType, EpilogueScheduleType>(A, B, weight_scales,
            weight_zero_points, biases, alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream,
            occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLM Error][fpA_intB][sm90_dispatch_epilogue_schedules] epilogue schedule config is invalid for "
            "mixed "
            "type GEMM.");
        break;
    }
}

template <typename CTAShape, typename ClusterShape>
constexpr bool are_tile_shapes_supported()
{
    [[maybe_unused]] constexpr int cta_m = get<0>(CTAShape{});
    [[maybe_unused]] constexpr int cta_n = get<1>(CTAShape{});
    constexpr int cga_m = get<0>(ClusterShape{});
    constexpr int cga_n = get<1>(ClusterShape{});

    if constexpr (cga_m == _1{} && cga_n == _1{})
    {
        return true;
    }
    else if constexpr (cga_m == _2{} && cga_n == _1{} && cta_m >= _128{})
    {
        return true;
    }
    else if constexpr (cga_m == _1{} && cga_n == _2{} && cta_n >= _128{})
    {
        return true;
    }
    else if constexpr (cga_m == _2{} && cga_n == _2{} && cta_m >= _128{} && cta_n >= _128{})
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape>
void sm90_dispatch_mainloop_schedules(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

    constexpr bool tile_shapes_supported = are_tile_shapes_supported<CTAShape, ClusterShape>();

    if constexpr (tile_shapes_supported)
    {
        switch (gemm_config.mainloop_schedule)
        {
        case tkc::MainloopScheduleType::AUTO:
            using KernelScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
                cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput,
                cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput>;
            sm90_dispatch_epilogue_schedules<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
                EpilogueTag, CTAShape, ClusterShape, KernelScheduleType>(A, B, weight_scales, weight_zero_points,
                biases, alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
            break;
        default:
            throw std::runtime_error(
                "[TensorRT-LLM Error][fpA_intB][sm90_dispatch_mainloop_schedules] mainloop schedule config is invalid "
                "for "
                "mixed type GEMM.");
            break;
        }
    }
    else
    {
        throw std::runtime_error(
            "[TensorRT-LLM Error][fpA_intB][sm90_dispatch_mainloop_schedules] Unsupported CTA and Cluster shapes for "
            "mixed type GEMM.");
    }
}

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape>
void sm90_dispatch_gemm_config(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemm_config.cluster_shape)
    {
    case tkc::ClusterShape::ClusterShape_1x1x1:
        sm90_dispatch_mainloop_schedules<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, CTAShape, Shape<_1, _1, _1>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n,
            k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x1x1:
        sm90_dispatch_mainloop_schedules<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, CTAShape, Shape<_2, _1, _1>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n,
            k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
        sm90_dispatch_mainloop_schedules<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, CTAShape, Shape<_1, _2, _1>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n,
            k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
        sm90_dispatch_mainloop_schedules<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, CTAShape, Shape<_2, _2, _1>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n,
            k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLM Error][fpA_intB][dispatch_CGA_config] Config is invalid for mixed type GEMM.");
        break;
    }
}

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag>
void sm90_dispatch_gemm_to_cutlass(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, char* workspace, size_t workspace_bytes, tkc::CutlassGemmConfig gemm_config,
    cudaStream_t stream, int* occupancy = nullptr)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

    constexpr int Ktile = 128 / sizeof(ActivationType);
    using _Ktile = Int<Ktile>;
    switch (gemm_config.tile_config_sm90)
    {
    case tkc::CutlassTileConfigSM90::CtaShape64x16x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_64, _16, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x32x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_64, _32, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x64x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_64, _64, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x128x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_64, _128, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x256x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_64, _256, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x16x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_128, _16, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x32x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_128, _32, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x64x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_128, _64, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x128x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_128, _128, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x256x128B:
        sm90_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp, EpilogueTag,
            Shape<_128, _256, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
            gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::Undefined:
        throw std::runtime_error(
            "[TensorRT-LLm Error][fpA_intB][sm90_dispatch_gemm_to_cutlass] gemm config undefined.");
        break;
    case tkc::CutlassTileConfigSM90::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT-LLm Error][fpA_intB][sm90_dispatch_gemm_to_cutlass] gemm config should have already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLm Error][fpA_intB][sm90_dispatch_gemm_to_cutlass] Config is invalid for mixed type GEMM.");
        break;
    }
}

}
}
}
