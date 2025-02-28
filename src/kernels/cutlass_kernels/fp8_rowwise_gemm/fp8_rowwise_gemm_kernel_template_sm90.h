
#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

using namespace cute;

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CTAShape,
    typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
    typename TileSchedulerType = void>
struct DeviceGemmFp8RowwiseSm90
{
    static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

    using ElementA = ElementType;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = ElementType;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementC = void;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<OutElementType>::value;

    using ElementOutput = OutElementType;
    using LayoutOutput = cutlass::layout::RowMajor;
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using ElementBias = float;

    using ElementAccumulator = AccumElementType;
    using ElementCompute = float;
    using ElementComputeEpilogue = float;
    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using TileShape = CTAShape;
    using TileScheduler = TileSchedulerType;

    static constexpr bool PONG = false;
    static constexpr bool FAST_ACCUM = true;
    static constexpr bool USE_BIAS = false;

    using StageCountType = cutlass::gemm::collective::StageCountAuto;
    using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
    using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementComputeEpilogue,
        cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

    using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementComputeEpilogue,
        cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

    using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementBias,
        cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

    using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
        ElementComputeEpilogue,
        ElementComputeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

    using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementOutput,
        ElementComputeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

    using ComputeBias = cutlass::epilogue::fusion::Sm90Compute<cutlass::plus,
        ElementOutput,
        ElementBias,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTComputeBias = cutlass::epilogue::fusion::Sm90EVT<ComputeBias, Bias, EVTCompute1>;

    using EpilogueEVT = EVTCompute1;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementComputeEpilogue, ElementC, LayoutC, AlignmentC, ElementOutput, LayoutOutput,
        AlignmentOutput, cutlass::epilogue::TmaWarpSpecialized, EpilogueEVT>::CollectiveOp;

    using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
    using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using FastDefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using FastPongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;

    using SlowAccum = DefaultSchedule;
    using FastAccum = FastDefaultSchedule;
    using MainLoopSchedule = cute::conditional_t<FAST_ACCUM, FastAccum, SlowAccum>;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA,
        LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainLoopSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}
}
}
