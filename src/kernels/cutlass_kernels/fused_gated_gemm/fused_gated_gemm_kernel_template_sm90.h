
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
#include "cutlass_extensions/gemm/collective/collective_builder_gated.hpp"
#include "cutlass_extensions/gemm/kernel/gemm_universal_gated.hpp"

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

template <typename ElementType, typename AccumElementType, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType, typename EpilogueScheduleType, typename TileSchedulerType = void,
    template <class> class Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = false>
struct DeviceGemmGatedSm90
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

    using ElementC = ElementType;
    using LayoutC = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    static constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementOutput = ElementType;
    using LayoutOutput = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using ElementAccumulator = AccumElementType;
    using ElementCompute = float;
    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using TileShape = CTAShape;
    using KernelSchedule = MainloopScheduleType;
    using EpilogueSchedule = EpilogueScheduleType;
    using TileScheduler = TileSchedulerType;

    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using FusionOperation = cutlass::epilogue::fusion::ScaledAcc<ElementOutput, ElementCompute>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator, ElementC, LayoutC,
        AlignmentC, ElementOutput, LayoutOutput, AlignmentOutput, EpilogueSchedule, FusionOperation>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderGated<ArchTag, OperatorClass,
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule, Activation, SwapAB>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversalGated<Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}
}
}
