
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm_configs.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"
#include "../src/cutlass_kernels/cutlass_heuristic.h"
#include "../src/cutlass_kernels/cutlass_type_conversion.h"
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
    typename MainloopScheduleType, typename EpilogueScheduleType>
void sm90_generic_mixed_gemm_kernelLauncher(ActivationType const* A, WeightType const* B,
    ScaleZeroType const* weight_scales, ScaleZeroType const* weight_zero_points, BiasType const* biases,
    float const alpha, OutputType* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemm_config,
    char* workspace, size_t workspace_bytes, cudaStream_t stream, int* occupancy)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

#ifdef COMPILE_HOPPER_TMA_GEMMS
    using CutlassActivationType = typename TllmToCutlassTypeAdapter<ActivationType>::type;

    if constexpr (!should_filter_sm90_gemm_problem_shape_v<CTAShape, ClusterShape, ActivationType>)
    {
        using CutlassWeightType__ = typename TllmToCutlassTypeAdapter<WeightType>::type;
        using CutlassWeightType_ = std::conditional_t<std::is_same_v<CutlassWeightType__, cutlass::uint4b_t>,
            cutlass::int4b_t, CutlassWeightType__>;

        using CutlassWeightType
            = std::conditional_t<std::is_same_v<CutlassWeightType_, uint8_t>, int8_t, CutlassWeightType_>;

        using CutlassScaleZeroType = typename TllmToCutlassTypeAdapter<ScaleZeroType>::type;
        using CutlassBiasType = typename TllmToCutlassTypeAdapter<BiasType>::type;
        using CutlassOutputType = typename TllmToCutlassTypeAdapter<OutputType>::type;

        static_assert(std::is_same_v<CutlassActivationType, cutlass::half_t>
                || std::is_same_v<CutlassActivationType, cutlass::bfloat16_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e5m2_t>,
            "Activation type must be bfloat16, half, FP8");

        static_assert(std::is_same_v<CutlassWeightType, int8_t> || std::is_same_v<CutlassWeightType, cutlass::int4b_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e5m2_t>,
            "Weight type must be fp8, int8_t or int4_t");

        static_assert(!std::is_same_v<CutlassActivationType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassScaleZeroType, cutlass::half_t>,
            "Scale/Zero type must be half for fp8 activation");

        using LayoutA = cutlass::layout::RowMajor;
        constexpr int AlignmentA = 128 / cutlass::sizeof_bits<CutlassActivationType>::value;

        using LayoutB = cutlass::layout::ColumnMajor;
        constexpr int AlignmentB = 128 / cutlass::sizeof_bits<CutlassWeightType>::value;

        using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
        using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

        using ElementZero = CutlassScaleZeroType;
        using ElementScale = CutlassScaleZeroType;

        using LayoutBias = cutlass::layout::RowMajor;
        constexpr int AlignmentBias = 128 / cutlass::sizeof_bits<CutlassBiasType>::value;

        using LayoutOutput = cutlass::layout::RowMajor;
        constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;

        using ElementAccumulator = float;
        using ElementCompute = float;
        using ArchTag = cutlass::arch::Sm90;
        using OperatorClass = cutlass::arch::OpClassTensorOp;
        using TileShape = CTAShape;
        using KernelSchedule = MainloopScheduleType;
        using EpilogueSchedule = EpilogueScheduleType;

        constexpr int epi_tile_M = cute::min(shape<0>(TileShape{}), 128);
        constexpr int epi_tile_N = cute::min(shape<1>(TileShape{}), 32);
        using EpilogueTileType = cute::Shape<cute::Int<epi_tile_M>, cute::Int<epi_tile_N>>;

        static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
        static_assert(std::is_same_v<EpilogueTag, suggestify::cutlass_extensions::EpilogueOpBias>, "");
        using EVT_bias_addition = cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, CutlassOutputType, ElementCompute,
                RoundStyle>,
            cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAccumulator>,
            cutlass::epilogue::fusion::Sm90AccFetch,
            cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, CutlassBiasType, Stride<_1, _0, _0>,
                AlignmentBias>
            >;

        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
            TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator,
            void, typename cutlass::layout::LayoutTranspose<LayoutBias>::type, AlignmentBias, CutlassOutputType,
            typename cutlass::layout::LayoutTranspose<LayoutOutput>::type, AlignmentOutput, EpilogueSchedule,
            EVT_bias_addition>::CollectiveOp;

        using PackedScaleZero = cute::tuple<CutlassWeightType, ElementScale, ElementZero>;
        using PackedScale = cute::tuple<CutlassWeightType, ElementScale>;
        using ElementBCollectiveInfo = std::conditional_t<cutlass::hasZero(QuantOp), PackedScaleZero, PackedScale>;

        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass,
            ElementBCollectiveInfo, LayoutB_Transpose, AlignmentB, CutlassActivationType, LayoutA_Transpose, AlignmentA,
            ElementAccumulator, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelSchedule>::CollectiveOp;

        using TileScheduler = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{}, cutlass::gemm::PersistentScheduler,
            cutlass::gemm::StreamKScheduler>;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
            CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

        if (occupancy != nullptr)
        {
            *occupancy = suggestify::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel, true>();
            return;
        }

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

        using StrideA = typename GemmKernel::StrideA;
        using StrideB = typename GemmKernel::StrideB;
        using StrideC = typename GemmKernel::StrideC;
        using StrideD = typename GemmKernel::StrideD;
        using StrideS = typename CollectiveMainloop::StrideScale;

        if (weight_scales == nullptr)
        {
            throw std::runtime_error("Weight scales must always be set to a non-null value.");
        }

        if constexpr (cutlass::isFinegrained(QuantOp))
        {
            int cta_shape_k = cute::size<2>(TileShape{});
            if (group_size % cta_shape_k != 0)
            {
                std::string err_msg = "The group size must a multiple of " + std::to_string(cta_shape_k);
                throw std::runtime_error("[nexly Error][fpA_intB Runner]" + err_msg);
            }

            if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY)
            {
                if (weight_zero_points != nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be a nullptr for scale only fine grained");
                }
            }
            else if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
            {
                if (weight_zero_points == nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be valid for scale and bias fine grained");
                }
            }
        }
        else
        {
            if (group_size != k)
            {
                throw std::runtime_error("Invalid group size for per column scaling kernels.");
            }

            if (weight_zero_points != nullptr)
            {
                throw std::runtime_error("Weight zero-points must be null when running per column scaling");
            }
        }

        auto cutlass_scale_k = (k + group_size - 1) / group_size;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
        StrideS stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, cutlass_scale_k, 1));

        auto output_as_bias_type = reinterpret_cast<CutlassBiasType const*>(C);

        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1},
            {reinterpret_cast<CutlassWeightType const*>(B), stride_B, reinterpret_cast<CutlassActivationType const*>(A),
                stride_A, reinterpret_cast<ElementScale const*>(weight_scales), stride_S, group_size,
                reinterpret_cast<ElementZero const*>(weight_zero_points)},
            {{}, output_as_bias_type, stride_D, reinterpret_cast<CutlassOutputType*>(C), stride_D}};

        args.epilogue.thread = {
            {alpha},
            {},
            {reinterpret_cast<CutlassBiasType const*>(biases), CutlassBiasType(0.f)},
            {}
        };

        Gemm gemm;
        if (gemm.get_workspace_size(args) > workspace_bytes)
        {
            LOG_ERROR("[nexly Error][fpA_intB Runner] given workspace size insufficient.");
        }

        auto can_implement = gemm.can_implement(args);
        if (can_implement != cutlass::Status::kSuccess)
        {
            std::string err_msg = "fpA_intB cutlass kernel will fail for params. Error: "
                + std::string(cutlassGetStatusString(can_implement));
            std::cout << err_msg << std::endl;
            throw std::runtime_error("[nexly Error][fpA_intB Runner] " + err_msg);
        }

        auto init_status = gemm.initialize(args, workspace, stream);
        if (init_status != cutlass::Status::kSuccess)
        {
            std::string err_msg = "Failed to initialize cutlass fpA_intB gemm. Error: "
                + std::string(cutlassGetStatusString(init_status));
            throw std::runtime_error("[nexly Error][fpA_intB Runner] " + err_msg);
        }

        auto run_status = gemm.run(stream);
        if (run_status != cutlass::Status::kSuccess)
        {
            std::string err_msg
                = "Failed to run cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
            throw std::runtime_error("[nexly Error][fpA_intB Runner] " + err_msg);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "[nexly Error][fpA_intB Runner] Config (" << (int64_t) cute::size<0>(CTAShape{}) << ","
           << (int64_t) cute::size<1>(CTAShape{}) << "," << (int64_t) cute::size<2>(CTAShape{}) << ") ("
           << (int64_t) cute::size<0>(ClusterShape{}) << "," << (int64_t) cute::size<1>(ClusterShape{}) << ","
           << (int64_t) cute::size<2>(ClusterShape{}) << ") not compiled with FAST_BUILD.";

        throw std::runtime_error(ss.str());
    }

#else
    throw std::runtime_error(
        "[nexly Error][fpA_intB Runner] Please recompile with support for hopper by passing 90-real as an arch "
        "to build_wheel.py.");
#endif
}

}
}
}
