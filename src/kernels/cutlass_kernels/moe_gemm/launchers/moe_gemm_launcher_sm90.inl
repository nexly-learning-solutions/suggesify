#pragma once
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue/collective/epilogue_moe_finalize.hpp"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../src/cutlass_kernels/cutlass_heuristic.h"
#include "../src/cutlass_kernels/cutlass_type_conversion.h"
#include "../src/cutlass_kernels/moe_gemm/moe_sm90_traits.h"

#include "../src/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.h"
#include "../src/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{
using EpilogueFusion = HopperGroupedGemmInput::EpilogueFusion;

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag, typename TileShape,
    typename ClusterShape, bool BIAS, EpilogueFusion FUSION>
struct HopperGroupedGemmInfo
{
    using Arch = cutlass::arch::Sm90;

    static_assert(cutlass::platform::is_same<T, WeightType>::value,
        "CUTLASS does not currently have specialised SM90 support for quantized operations");

#ifdef ENABLE_FP8
    constexpr static bool IsFP8
        = cutlass::platform::is_same<T, __nv_fp8_e4m3>::value || cutlass::platform::is_same<T, __nv_fp8_e5m2>::value;
#else
    constexpr static bool IsFP8 = false;
#endif

#ifdef ENABLE_BF16
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value
            || cutlass::platform::is_same<T, float>::value || IsFP8,
        "Specialized for bfloat16, half, float, fp8");
#else
    static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value || IsFP8,
        "Specialized for half, float, fp8");
#endif

    static_assert(cutlass::platform::is_same<T, WeightType>::value
            || cutlass::platform::is_same<WeightType, uint8_t>::value
            || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value
            || cutlass::platform::is_same<WeightType, cutlass::float_e4m3_t>::value
            || cutlass::platform::is_same<WeightType, cutlass::float_e5m2_t>::value,
        "Unexpected quantization type");

    using ElementType = typename TllmToCutlassTypeAdapter<T>::type;

    using CutlassWeightTypeMaybeUint4 = typename TllmToCutlassTypeAdapter<WeightType>::type;
    using CutlassWeightTypeMaybeUint8
        = std::conditional_t<std::is_same_v<CutlassWeightTypeMaybeUint4, cutlass::uint4b_t>, cutlass::int4b_t,
            CutlassWeightTypeMaybeUint4>;
    using CutlassWeightType
        = std::conditional_t<std::is_same_v<CutlassWeightTypeMaybeUint8, uint8_t>, int8_t, CutlassWeightTypeMaybeUint8>;

    using ElementA = ElementType;
    using ElementB = CutlassWeightType;

    using ElementD = typename TllmToCutlassTypeAdapter<HopperGroupedGemmInput::OutputTypeAdaptor_t<OutputType>>::type;
    using ElementFinalOutput = typename TllmToCutlassTypeAdapter<OutputType>::type;

    using ElementC = void;
    using ElementCNoVoid = ElementD;

    using ElementAccumulator = float;

    using ElementBias = ElementFinalOutput;
    using ElementRouterScales = float;

    using LayoutA = HopperGroupedGemmInput::LayoutA;
    constexpr static int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value;

    using LayoutB = HopperGroupedGemmInput::LayoutB;
    constexpr static int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value;

    using LayoutC = HopperGroupedGemmInput::LayoutC;
    using StrideC = HopperGroupedGemmInput::StrideC;
    constexpr static int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementType>::value;

    using LayoutD = HopperGroupedGemmInput::DefaultEpilogue::LayoutD;
    using StrideD = HopperGroupedGemmInput::DefaultEpilogue::StrideD;
    constexpr static int AlignmentD
        = 128 / cutlass::sizeof_bits<ElementD>::value;

    static_assert(cutlass::platform::is_same<EpilogueTag, suggestify::cutlass_extensions::EpilogueOpDefault>::value,
        "Hopper Grouped GEMM specialisation doesn't support fused activation");

    using EpilogueOp
        = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator, ElementC, ElementAccumulator>;

    using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;

    using CollectiveEpilogueDefault = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    using CollectiveEpilogueFinalize = typename cutlass::epilogue::collective::EpilogueMoeFusedFinalizeBuilder<
        TileShape,
        ElementCNoVoid, StrideC*,
        ElementFinalOutput, HopperGroupedGemmInput::FusedFinalizeEpilogue::StrideFinalOutput,
        ElementAccumulator,
        ElementAccumulator,
        ElementBias, HopperGroupedGemmInput::FusedFinalizeEpilogue::StrideBias,
        ElementRouterScales, HopperGroupedGemmInput::FusedFinalizeEpilogue::StrideRouterScales
        >::CollectiveOp;

    using CollectiveEpilogue
        = std::conditional_t<FUSION == EpilogueFusion::FINALIZE, CollectiveEpilogueFinalize, CollectiveEpilogueDefault>;

    using StageCountAutoCarveout = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;

    using KernelSchedule
        = std::conditional_t<IsFP8, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp,
        CutlassWeightType, LayoutB*, AlignmentB,
        ElementType, LayoutA*, AlignmentA,
        ElementAccumulator,
        TileShape, ClusterShape,
        StageCountAutoCarveout, KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<HopperGroupedGemmInput::ProblemShape, CollectiveMainloop,
        CollectiveEpilogue>;

    using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag, EpilogueFusion FUSION,
    typename TileShape, typename ClusterShape, bool BIAS>
void sm90_generic_moe_gemm_kernelLauncher(HopperGroupedGemmInput hopper_input, int num_experts,
    int const multi_processor_count, cudaStream_t stream, int* kernel_occupancy, size_t* workspace_size)
{
#ifdef COMPILE_HOPPER_TMA_GEMMS
    using namespace cute;
    if constexpr (!should_filter_sm90_gemm_problem_shape_v<TileShape, ClusterShape, T>)
    {
        using GemmInfo
            = HopperGroupedGemmInfo<T, WeightType, OutputType, EpilogueTag, TileShape, ClusterShape, BIAS, FUSION>;

        using ElementAccumulator = typename GemmInfo::ElementAccumulator;
        using ElementA = typename GemmInfo::ElementA;
        using ElementB = typename GemmInfo::ElementB;
        using ElementC = typename GemmInfo::ElementC;
        using ElementCNoVoid = typename GemmInfo::ElementCNoVoid;
        using ElementD = typename GemmInfo::ElementD;

        using CollectiveMainloop = typename GemmInfo::CollectiveMainloop;
        using CollectiveEpilogue = typename GemmInfo::CollectiveEpilogue;
        using GemmKernel = typename GemmInfo::GemmKernel;
        using GemmGrouped = typename GemmInfo::GemmGrouped;

        if (kernel_occupancy != nullptr)
        {
            *kernel_occupancy = suggestify::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel, true>();
            return;
        }

        cutlass::KernelHardwareInfo hw_info;
        hw_info.device_id = 0;
        hw_info.sm_count = multi_processor_count;

        GemmGrouped gemm;

        if (workspace_size != nullptr)
        {
            typename HopperGroupedGemmInput::ProblemShape shape_info{num_experts, nullptr, nullptr};

            typename GemmGrouped::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGrouped, shape_info, {}, {}, hw_info};
            *workspace_size = gemm.get_workspace_size(args);
            return;
        }

        using MainloopArguments = typename CollectiveMainloop::Arguments;
        CHECK(hopper_input.stride_a);
        CHECK(hopper_input.stride_b);
        CHECK(hopper_input.ptr_a);
        CHECK(hopper_input.ptr_b);

        MainloopArguments const mainloop_params = {reinterpret_cast<ElementB const**>(hopper_input.ptr_b),
            hopper_input.stride_b, reinterpret_cast<ElementA const**>(hopper_input.ptr_a), hopper_input.stride_a};

        typename GemmGrouped::EpilogueOutputOp::Params epilogue_scalars{
            ElementAccumulator(1.f), hopper_input.ptr_c ? ElementAccumulator(1.f) : ElementAccumulator(0.f)};
        epilogue_scalars.alpha_ptr_array = hopper_input.alpha_scale_ptr_array;
        using EpilogueArguments = typename CollectiveEpilogue::Arguments;
        auto make_epi_args = [&]()
        {
            if constexpr (FUSION == EpilogueFusion::NONE)
            {
                auto epi_params = hopper_input.default_epilogue;
                return EpilogueArguments{epilogue_scalars, reinterpret_cast<ElementCNoVoid const**>(hopper_input.ptr_c),
                    hopper_input.stride_c, reinterpret_cast<ElementD**>(epi_params.ptr_d), epi_params.stride_d};
            }
            else if constexpr (FUSION == EpilogueFusion::FINALIZE)
            {
                auto epi_params = hopper_input.fused_finalize_epilogue;
                return EpilogueArguments{
                    epilogue_scalars,
                    reinterpret_cast<ElementCNoVoid const**>(hopper_input.ptr_c), hopper_input.stride_c,
                    reinterpret_cast<typename GemmInfo::ElementFinalOutput*>(epi_params.ptr_final_output),
                    epi_params.stride_final_output,
                    reinterpret_cast<typename GemmInfo::ElementBias const*>(epi_params.ptr_bias),
                    epi_params.stride_bias,
                    epi_params.ptr_router_scales, epi_params.stride_router_scales,
                    epi_params.ptr_expert_first_token_offset,
                    epi_params.ptr_source_token_index,
                    epi_params.num_rows_in_final_output
                };
            }
            else
            {
                static_assert(
                    sizeof(EpilogueArguments) == 0, "Unimplemented fusion provided to SM90+ MoE gemm launcher");
            }
        };
        EpilogueArguments const epilogue_params = make_epi_args();

        typename GemmKernel::TileScheduler::Arguments scheduler_args{
            1, GemmKernel::TileScheduler::RasterOrderOptions::AlongN};

        typename GemmGrouped::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped, hopper_input.shape_info,
            mainloop_params, epilogue_params, hw_info, scheduler_args};

        size_t calculated_ws_size = gemm.get_workspace_size(args);
        CHECK_WITH_INFO(calculated_ws_size <= hopper_input.gemm_workspace_size,
            "Workspace is size %zu but only %zu were allocated", calculated_ws_size, hopper_input.gemm_workspace_size);

        auto can_implement = gemm.can_implement(args);
        CHECK_WITH_INFO(can_implement == cutlass::Status::kSuccess,
            "Grouped GEMM kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement)));

        auto init_status = gemm.initialize(args, hopper_input.gemm_workspace);
        CHECK_WITH_INFO(init_status == cutlass::Status::kSuccess,
            "Failed to initialize cutlass SM90 grouped gemm. Error: "
                + std::string(cutlassGetStatusString(init_status)));

        auto run_status = gemm.run(stream);
        CHECK_WITH_INFO(run_status == cutlass::Status::kSuccess,
            "Failed to run cutlass SM90 grouped gemm. Error: " + std::string(cutlassGetStatusString(run_status)));
        sync_check_cuda_error();
    }
    else
    {
        THROW("Configuration was disabled by FAST_BUILD");
    }

#else
    THROW("Please recompile with support for hopper by passing 90-real as an arch to build_wheel.py.");
#endif
}

}
}
}
