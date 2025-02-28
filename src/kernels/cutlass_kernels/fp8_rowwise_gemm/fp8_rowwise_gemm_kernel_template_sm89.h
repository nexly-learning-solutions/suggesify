
#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

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

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CtaShape,
    typename WarpShape, int Stages>
struct DeviceGemmFp8RowwiseSm89
{
    static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

    using ElementA = ElementType;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = ElementType;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementC = OutElementType;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementOutput = OutElementType;
    using LayoutOutput = cutlass::layout::RowMajor;
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    using ElementAccumulator = AccumElementType;
    using ElementComputeEpilogue = float;
    using ArchTag = cutlass::arch::Sm89;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTEpilogueStages = 1;

    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<CtaShape, WarpShape, ElementC,
        AlignmentC, EVTEpilogueStages>;

    using accSrc = cutlass::epilogue::threadblock::VisitorAccFetch;

    using ComputeBScale = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    using bScaleSrc = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementComputeEpilogue,
        Stride<_0, _1, _0>>;
    using EpilogueBScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeBScale, accSrc, bScaleSrc>;

    using ComputeAScale = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementC,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    using aScaleSrc = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, ElementComputeEpilogue,
        Stride<_1, _0, _0>>;
    using EpilogueAScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeAScale, EpilogueBScale, aScaleSrc>;

    using dTar = cutlass::epilogue::threadblock::VisitorAuxStore<OutputTileThreadMap, ElementC,
        cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, _0>>;
    using EpilogueStore = cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScale>;

    using EpilogueOp = EpilogueStore;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, AlignmentA, ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
        ElementC, LayoutC, AlignmentC, ElementAccumulator, ElementComputeEpilogue, OperatorClass, ArchTag, CtaShape,
        WarpShape, InstructionShape, EpilogueOp, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages,
        cutlass::arch::OpMultiplyAdd, EVTEpilogueStages>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}
}
}
