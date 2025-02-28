
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"

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
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"

#include "../src/cutlass_kernels/cutlass_heuristic.h"
#include "../src/cutlass_kernels/cutlass_type_conversion.h"

#include "moe_gemm_kernels_template_sm90.h"
#include "../src/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.h"
#include "../src/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "../src/cutlass_kernels/moe_gemm/moe_sm90_traits.h"
#include <../src/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace suggestify
{
namespace kernels::cutlass_kernels
{

template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
    typename ThreadblockShape, typename WarpShape, int Stages>
void genericMoeGemmKernelLauncher(T const* A, WeightType const* B, GemmOutputType const* weight_scales,
    GemmOutputType const* biases, bool bias_is_broadcast, GemmOutputType* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int const multi_processor_count, bool use_fused_moe,
    float const** alpha_scale_ptr_array, cudaStream_t stream, int* kernel_occupancy = nullptr)
{
#if defined(ENABLE_FP8)
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value
            || cutlass::platform::is_same<T, __nv_fp8_e4m3>::value
            || cutlass::platform::is_same<T, __nv_fp8_e5m2>::value || cutlass::platform::is_same<T, float>::value,
        "Specialized for fp8, bfloat16, half, float");
#elif defined(ENABLE_BF16)
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value
            || cutlass::platform::is_same<T, float>::value,
        "Specialized for bfloat16, half, float");
#else
    static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
        "Specialized for half, float");
#endif

    static_assert(cutlass::platform::is_same<T, WeightType>::value
            || cutlass::platform::is_same<WeightType, uint8_t>::value
            || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
        "");

    static_assert(!cutlass::platform::is_same<arch, cutlass::arch::Sm90>::value,
        "Sm90 architecture should use specialised kernels");

    using ElementType = typename TllmToCutlassTypeAdapter<T>::type;
    using CutlassGemmOutputType = typename TllmToCutlassTypeAdapter<GemmOutputType>::type;
    using CutlassWeightType = typename TllmToCutlassTypeAdapter<WeightType>::type;
    if (!use_fused_moe)
    {
        using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
        using ElementAccumulator = typename MixedGemmArchTraits::AccType;

        using EpilogueOp = typename suggestify::cutlass_extensions::Epilogue<CutlassGemmOutputType,
            MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

        typename EpilogueOp::Params epilogue_op(
            ElementAccumulator(1.f), biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));

#if defined(ENABLE_FP8)
        if constexpr ((std::is_same_v<T, __nv_fp8_e4m3>
                          || std::is_same_v<T, __nv_fp8_e5m2>) &&std::is_same_v<EpilogueTag,
                          cutlass_extensions::EpilogueOpDefault>)
        {
            CHECK_WITH_INFO(weight_scales == nullptr && biases == nullptr && alpha_scale_ptr_array,
                "weight_scales and biases should be nullptr and alpha_scale_ptr_array shouldn't be nullptr for FP8 "
                "Ada");
            epilogue_op.alpha_ptr_array = alpha_scale_ptr_array;
        }
#endif

        using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor,
            cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType,
            typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
            MixedGemmArchTraits::ElementsPerAccessB, CutlassGemmOutputType, cutlass::layout::RowMajor,
            ElementAccumulator, typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
            typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
            cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
            cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;

        using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
            typename GemmKernel_::ThreadblockSwizzle,
            arch,
            GemmKernel_::kGroupScheduleMode>;

        using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

        if (kernel_occupancy != nullptr)
        {
            *kernel_occupancy = suggestify::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
            return;
        }
        int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
        CHECK_WITH_INFO(occupancy > 0, "GPU lacks the shared memory resources to run GroupedGEMM kernel");
        int const threadblock_count = multi_processor_count * occupancy;

        int const group_size = gemm_k;
        typename GemmGrouped::Arguments args(num_experts, threadblock_count, group_size, epilogue_op,
            reinterpret_cast<ElementType const*>(A), reinterpret_cast<CutlassWeightType const*>(B),
            reinterpret_cast<CutlassGemmOutputType const*>(weight_scales),
            reinterpret_cast<CutlassGemmOutputType const*>(biases), bias_is_broadcast,
            reinterpret_cast<CutlassGemmOutputType*>(C), total_tokens_including_expert, gemm_n, gemm_k);

        GemmGrouped gemm;

        auto can_implement = gemm.can_implement(args);
        CHECK_WITH_INFO(can_implement == cutlass::Status::kSuccess,
            "MoE FC kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement)));

        auto init_status = gemm.initialize(args);
        CHECK_WITH_INFO(init_status == cutlass::Status::kSuccess,
            "Failed to initialize cutlass grouped gemm. Error: " + std::string(cutlassGetStatusString(init_status)));

        auto run_status = gemm.run(stream);
        CHECK_WITH_INFO(run_status == cutlass::Status::kSuccess,
            "Failed to run cutlass grouped gemm. Error: " + std::string(cutlassGetStatusString(run_status)));
    }
    else if constexpr (sizeof(ElementType) == 2 && sizeof(CutlassWeightType) == 2
        && (std::is_same_v<EpilogueTag, cutlass_extensions::EpilogueOpDefaultSilu>
            || std::is_same_v<EpilogueTag, cutlass_extensions::EpilogueOpDefaultFtGelu>) )
    {
        sm80_generic_fused_moe_gemm_kernelLauncher<ElementType, CutlassWeightType, ThreadblockShape::kM,
            ThreadblockShape::kN, ThreadblockShape::kK, Stages, EpilogueTag>(reinterpret_cast<ElementType const*>(A),
            reinterpret_cast<CutlassWeightType const*>(B), reinterpret_cast<ElementType const*>(biases),
            bias_is_broadcast, reinterpret_cast<ElementType*>(C), total_tokens_including_expert, num_rows, gemm_n,
            gemm_k, num_experts, multi_processor_count, stream, kernel_occupancy);
    }
}

}

template <typename T, typename WeightType, typename GemmOutputType, typename Arch, typename EpilogueTag,
    typename ThreadblockShape, typename WarpShape, int Stages>
static void dispatch(T const* A, WeightType const* B, GemmOutputType const* weight_scales, GemmOutputType const* biases,
    bool bias_is_broadcast, GemmOutputType* C, int64_t const* total_tokens_including_expert, int64_t num_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
    int multi_processor_count, bool use_fused_moe, float const** alpha_scale_ptr_array, cudaStream_t stream,
    int* occupancy = nullptr)
{

    static_assert(!std::is_same_v<Arch, cutlass::arch::Sm90>, "Use TMA specialised functions for arch SM90");
#if defined(ENABLE_FP8)
    constexpr bool isFp8 = std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;
#else
    constexpr bool isFp8 = false;
#endif

    if constexpr ((Stages == 2 || Arch::kMinComputeCapability >= 80)
        && (!isFp8 || std::is_same_v<Arch, cutlass::arch::Sm89>) )
    {
        kernels::cutlass_kernels::genericMoeGemmKernelLauncher<T, WeightType, GemmOutputType, Arch, EpilogueTag,
            ThreadblockShape, WarpShape, Stages>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
    }
    else
    {
        THROW(
            "Cutlass gemm. Not instantiated for arch %d with stages set to %d", Arch::kMinComputeCapability, Stages);
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
    typename ThreadblockShape, typename WarpShape>
void dispatchGemmConfig(T const* A, WeightType const* B, GemmOutputType const* weight_scales,
    GemmOutputType const* biases, bool bias_is_broadcast, GemmOutputType* C,
    int64_t const* total_tokens_including_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
    float const** alpha_scale_ptr_array, cudaStream_t stream, int* occupancy = nullptr)
{
    switch (gemm_config.stages)
    {
    case 2:
        dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(A, B, weight_scales,
            biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts,
            gemm_config, multi_processor_count, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case 3:
        dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>(A, B, weight_scales,
            biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts,
            gemm_config, multi_processor_count, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case 4:
        dispatch<T, WeightType, GemmOutputType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>(A, B, weight_scales,
            biases, bias_is_broadcast, C, total_tokens_including_expert, num_rows, gemm_n, gemm_k, num_experts,
            gemm_config, multi_processor_count, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    default: THROW("dispatchGemmConfig does not support stages %d", gemm_config.stages); break;
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value
#if defined(ENABLE_FP8)
        && !std::is_same<T, __nv_fp8_e4m3>::value && !std::is_same<T, __nv_fp8_e5m2>::value
#endif
        && std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(T const* A, WeightType const* B, GemmOutputType const* weight_scales,
    GemmOutputType const* biases, bool bias_is_broadcast, GemmOutputType* C,
    int64_t const* total_tokens_including_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
    float const** alpha_scale_ptr_array, cudaStream_t stream, int* occupancy = nullptr)
{
    switch (gemm_config.tile_config)
    {
    case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
        CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
        if constexpr (arch::kMinComputeCapability >= 75)
        {
            dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
                cutlass::gemm::GemmShape<16, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
                total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                multi_processor_count, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        }
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
        CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
        if constexpr (arch::kMinComputeCapability >= 75)
        {
            dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
                cutlass::gemm::GemmShape<16, 64, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
                total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                multi_processor_count, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        }
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<32, 64, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined: THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        THROW("GEMM config should have already been set by heuristic.");
        break;
    default: THROW("Config is invalid for same type tensorop GEMM."); break;
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(T const* A, WeightType const* B, GemmOutputType const* weight_scales,
    GemmOutputType const* biases, bool bias_is_broadcast, GemmOutputType* C,
    int64_t const* total_tokens_including_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
    float const** alpha_scale_ptr_array, cudaStream_t stream, int* occupancy = nullptr)
{
    switch (gemm_config.tile_config)
    {
    case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
        CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
        if constexpr (arch::kMinComputeCapability >= 75)
        {
            dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
                cutlass::gemm::GemmShape<16, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
                total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                multi_processor_count, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        }
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
        CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
        if constexpr (arch::kMinComputeCapability >= 75)
        {
            dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
                cutlass::gemm::GemmShape<16, 64, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
                total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config,
                multi_processor_count, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        }
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<128, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined: THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        THROW("GEMM config should have already been set by heuristic.");
        break;
    default: THROW("Config is invalid for mixed type tensorop GEMM."); break;
    }
}

#if defined(ENABLE_FP8)
template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
    typename std::enable_if<(std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value)
        && std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(T const* A, WeightType const* B, GemmOutputType const* weight_scales,
    GemmOutputType const* biases, bool bias_is_broadcast, GemmOutputType* C,
    int64_t const* total_tokens_including_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
    float const** alpha_scale_ptr_array, cudaStream_t stream, int* occupancy = nullptr)
{
    switch (gemm_config.tile_config)
    {
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 128>,
            cutlass::gemm::GemmShape<16, 64, 128>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 64, 128>,
            cutlass::gemm::GemmShape<32, 64, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 64, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 256, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<256, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined: THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        THROW("GEMM config should have already been set by heuristic.");
        break;
    default: THROW("Config is invalid for same type tensorop GEMM."); break;
    }
}
#endif

template <typename T, typename WeightType, typename GemmOutputType, typename arch, typename EpilogueTag,
    typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(T const* A, WeightType const* B, GemmOutputType const* weight_scales,
    GemmOutputType const* biases, bool bias_is_broadcast, GemmOutputType* C,
    int64_t const* total_tokens_including_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
    float const** alpha_scale_ptr_array, cudaStream_t stream, int* occupancy = nullptr)
{
    switch (gemm_config.tile_config)
    {
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
        dispatchGemmConfig<T, WeightType, GemmOutputType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<64, 64, 8>>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
            use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined: THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        THROW("GEMM config should have already been set by heuristic.");
        break;
    default: THROW("Unsupported config for float MoE gemm."); break;
    }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getConfigs() const
{
    return getConfigs(sm_);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig> MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getConfigs(
    int sm)
{
    std::vector<cutlass_extensions::CutlassGemmConfig> candidate_configs = getHopperConfigs(sm);
    std::vector<cutlass_extensions::CutlassGemmConfig> ampere_configs = getAmpereConfigs(sm);
    std::copy(ampere_configs.begin(), ampere_configs.end(), std::back_inserter(candidate_configs));

    return candidate_configs;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getAmpereConfigs(int sm)
{
    using suggestify::cutlass_extensions::CutlassGemmConfig;
    static constexpr auto weight_only_flag
        = std::is_same<T, WeightType>::value ? CutlassGemmConfig::NONE : CutlassGemmConfig::WEIGHT_ONLY;
    static constexpr auto simt_only_flag
        = std::is_same<T, float>::value ? CutlassGemmConfig::SIMT_ONLY : CutlassGemmConfig::NONE;
    static constexpr auto fp8_only_flag = use_fp8 ? CutlassGemmConfig::FP8_ONLY : CutlassGemmConfig::NONE;
    int const max_split_k = 1;
    int const grouped_gemm_flag = CutlassGemmConfig::GROUPED_GEMM;
    int const enable_hopper = CutlassGemmConfig::NONE;

    auto config_type_param = static_cast<CutlassGemmConfig::CandidateConfigTypeParam>(
        weight_only_flag | simt_only_flag | grouped_gemm_flag | enable_hopper | fp8_only_flag);

    if (!kernels::cutlass_kernels::isValidAmpereMOESpecialisation<T, WeightType>())
    {
        return {};
    }

    std::vector<cutlass_extensions::CutlassGemmConfig> ampere_configs
        = kernels::cutlass_kernels::get_candidate_configs(sm, max_split_k, config_type_param);
    return ampere_configs;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
std::vector<cutlass_extensions::CutlassGemmConfig>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getHopperConfigs(int sm)
{
    using suggestify::cutlass_extensions::CutlassGemmConfig;
    static constexpr auto weight_only_flag
        = std::is_same<T, WeightType>::value ? CutlassGemmConfig::NONE : CutlassGemmConfig::WEIGHT_ONLY;
    static constexpr auto simt_only_flag
        = std::is_same<T, float>::value ? CutlassGemmConfig::SIMT_ONLY : CutlassGemmConfig::NONE;
    int const max_split_k = 1;
    int const grouped_gemm_flag = CutlassGemmConfig::GROUPED_GEMM;
    int const enable_hopper = CutlassGemmConfig::HOPPER;
    static constexpr auto fp8_only_flag = use_fp8 ? CutlassGemmConfig::FP8_ONLY : CutlassGemmConfig::NONE;
    auto config_type_param = static_cast<CutlassGemmConfig::CandidateConfigTypeParam>(
        weight_only_flag | simt_only_flag | grouped_gemm_flag | enable_hopper | fp8_only_flag);

    if (!kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType>())
    {
        return {};
    }

    std::vector<cutlass_extensions::CutlassGemmConfig> hopper_configs
        = kernels::cutlass_kernels::get_candidate_configs(sm, max_split_k, config_type_param);
    return hopper_configs;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::isHopperSpecialised(
    cutlass_extensions::CutlassGemmConfig gemm_config) const
{
    bool config_is_sm90 = gemm_config.is_sm90;
    return supportsHopperSpecialisation() && config_is_sm90;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::supportsHopperSpecialisation() const
{
    return sm_ == 90 && kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType>();
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
int MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getSM() const
{
    return this->sm_;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::supportsFusedGatedActivation(
    bool is_gated_activation, int gemm_n, int gemm_k) const
{
    constexpr bool ENABLE_FUSED_GATED_ACTIVATION = true;
    return is_gated_activation && std::is_same_v<T, WeightType> && !std::is_same_v<T, float> && !use_fp8
        && (this->getSM() >= 80) && (gemm_k % 64 == 0) && (gemm_n % 64 == 0) && ENABLE_FUSED_GATED_ACTIVATION;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
bool MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::isFusedGatedActivation(
    cutlass_extensions::CutlassGemmConfig gemm_config, bool is_gated_activation, int gemm_n, int gemm_k) const
{
    return supportsFusedGatedActivation(is_gated_activation, gemm_n, gemm_k) && !gemm_config.is_sm90;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::MoeGemmRunner()
{
    int device{-1};
    suggestify::common::check_cuda_error(cudaGetDevice(&device));
    sm_ = suggestify::common::getSMVersion();
    suggestify::common::check_cuda_error(
        cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::dispatchToArch<EpilogueTag>(T const* A,
    WeightType const* B, ScaleBiasType const* weight_scales, ScaleBiasType const* biases, bool bias_is_broadcast,
    void* C_void, int64_t const* total_tokens_including_expert, HopperGroupedGemmInput hopper_input, int64_t total_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
    bool use_fused_moe, float const** alpha_scale_ptr_array, cudaStream_t stream, int* occupancy)
{
    static_assert(std::is_same_v<ScaleBiasType, OutputType>,
        "Separate Scale/Bias type is not supported. This is assumed to be the gemm output type");

    auto* C = reinterpret_cast<OutputType*>(C_void);

    CHECK_WITH_INFO(
        sm_ >= 89 || !hopper_input.isValid(), "Hopper input information is set for non specialised implementation");
    CHECK_WITH_INFO(
        sm_ == 90 || !gemm_config.is_sm90, "Hopper configuration provided for non-Hopper architecture");

    if (sm_ >= 75 && sm_ < 80)
    {
        dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm75, EpilogueTag>(A, B, weight_scales,
            biases, bias_is_broadcast, C, total_tokens_including_expert, total_rows, gemm_n, gemm_k, num_experts,
            gemm_config, multi_processor_count_, use_fused_moe, alpha_scale_ptr_array, stream, occupancy);
    }
    else if (sm_ >= 80 && sm_ < 90)
    {
        if constexpr (use_fp8)
        {
#if defined(ENABLE_FP8)
            static_assert(!std::is_same_v<OutputType, __nv_fp8_e4m3> && !std::is_same_v<OutputType, __nv_fp8_e5m2>,
                "FP8 GEMM Output not supported");
#endif

            CHECK_WITH_INFO(sm_ == 89, "For sm >= 80 and < 90, fp8 is only supported with sm == 89");
            dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm89, EpilogueTag>(A, B,
                weight_scales, biases, bias_is_broadcast, C, total_tokens_including_expert, total_rows, gemm_n, gemm_k,
                num_experts, gemm_config, multi_processor_count_, use_fused_moe, alpha_scale_ptr_array, stream,
                occupancy);
        }
        else
        {
            dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm80, EpilogueTag>(A, B,
                weight_scales, biases, bias_is_broadcast, C, total_tokens_including_expert, total_rows, gemm_n, gemm_k,
                num_experts, gemm_config, multi_processor_count_, use_fused_moe, alpha_scale_ptr_array, stream,
                occupancy);
        }
    }
    else if (sm_ >= 90)
    {
        if constexpr (kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType, EpilogueTag>())
        {

            if (gemm_config.is_sm90)
            {
                CHECK_WITH_INFO(biases != nullptr || hopper_input.ptr_c == nullptr,
                    "Input biases and hopper input disagree if bias is enabled");
                CHECK_WITH_INFO(hopper_input.isValid(), "Calling SM90 configuration with invalid hopper config");

                auto select_function = [&]()
                {
                    switch (hopper_input.fusion)
                    {
                    case HopperGroupedGemmInput::EpilogueFusion::FINALIZE:
                        return &dispatchMoeGemmSelectTileShapeSM90<T, WeightType, OutputType, EpilogueTag,
                            HopperGroupedGemmInput::EpilogueFusion::FINALIZE>;
                    case HopperGroupedGemmInput::EpilogueFusion::NONE:
                        return &dispatchMoeGemmSelectTileShapeSM90<T, WeightType, OutputType, EpilogueTag,
                            HopperGroupedGemmInput::EpilogueFusion::NONE>;
                    case HopperGroupedGemmInput::EpilogueFusion::ACTIVATION:
                    case HopperGroupedGemmInput::EpilogueFusion::GATED_ACTIVATION:
                    default: THROW("Unimplemented fusion %d requested", (int) hopper_input.fusion);
                    };
                };
                auto selected_func = select_function();
                selected_func(
                    hopper_input, num_experts, gemm_config, multi_processor_count_, stream, occupancy, nullptr);
                return;
            }

        }

        if constexpr (kernels::cutlass_kernels::isValidAmpereMOESpecialisation<T, WeightType, EpilogueTag>())
        {
            CHECK_WITH_INFO(!hopper_input.isValid(),
                "Non-specialised Hopper implementation is being rerouted to fallback implementation so input "
                "information is not required");
            CHECK_WITH_INFO(!gemm_config.is_sm90,
                "GEMM config is for SM90 configuration, but this configuration is not valid for Hppper");
            dispatchMoeGemmToCutlass<T, WeightType, ScaleBiasType, cutlass::arch::Sm80, EpilogueTag>(A, B,
                weight_scales, biases, bias_is_broadcast, C, total_tokens_including_expert, total_rows, gemm_n, gemm_k,
                num_experts, gemm_config, multi_processor_count_, use_fused_moe, alpha_scale_ptr_array, stream,
                occupancy);
        }
        else
        {
            THROW("Configuration expects SM80 but configuration is not supported by SM80 kernels");
        }
    }
    else
    {
        THROW("Arch unsupported for MoE GEMM");
    }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
size_t MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::getMaxWorkspaceSize(int num_experts) const
{
    if (num_experts != num_experts_)
    {
        LOG_TRACE("Calling getMaxWorkspaceSize() with a new expert count %d vs %d", num_experts, num_experts_);
        num_experts_ = num_experts;
        gemm_workspace_size_ = calcMaxWorkspaceSize(num_experts);
    }
    return gemm_workspace_size_;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
size_t MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::calcMaxWorkspaceSize(int num_experts) const
{
    if (!supportsHopperSpecialisation())
    {
        return 0;
    }
    if constexpr (kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType>())
    {
        auto configs = getHopperConfigs(sm_);
        size_t max_size = 0;
        bool has_config = false;
        for (auto conf : configs)
        {
#define CALC_SIZE_FUSION(FUSION)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        try                                                                                                            \
        {                                                                                                              \
            size_t size = calcMaxWorkspaceSizeSM90<T, WeightType, OutputType, FUSION>(                                 \
                num_experts, conf, multi_processor_count_);                                                            \
            max_size = std::max(max_size, size);                                                                       \
            has_config = true;                                                                                         \
        }                                                                                                              \
        catch (suggestify::common::TllmException const& e)                                                           \
        {                                                                                                              \
            LOG_TRACE("Unsupported config skipped when calculating MOE workspace size");                          \
        }                                                                                                              \
    } while (0)

            CALC_SIZE_FUSION(HopperGroupedGemmInput::EpilogueFusion::NONE);
            CALC_SIZE_FUSION(HopperGroupedGemmInput::EpilogueFusion::FINALIZE);
        }
        CHECK_WITH_INFO(has_config, "Could not find valid config when calculating workspace size");
        return max_size;
    }
    else
    {
        THROW("Attempting to calculate Hopper GEMM workspace size with unsupported weight combination");
        return 0;
    }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::runGemm(T const* A, WeightType const* B,
    ScaleBiasType const* weight_scales, ScaleBiasType const* biases, bool bias_is_broadcast, void* C,
    int64_t const* total_tokens_including_expert, HopperGroupedGemmInput hopper_input, int64_t total_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, bool use_fused_moe, float const** alpha_scale_ptr_array,
    cudaStream_t stream, cutlass_extensions::CutlassGemmConfig chosen_conf)
{
    dispatchToArch<EpilogueTag>(A, B, weight_scales, biases, bias_is_broadcast, C, total_tokens_including_expert,
        hopper_input, total_rows, gemm_n, gemm_k, num_experts, chosen_conf, use_fused_moe, alpha_scale_ptr_array,
        stream, nullptr);
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::moeGemmBiasAct(T const* A, WeightType const* B,
    ScaleBiasType const* weight_scales, ScaleBiasType const* biases, bool bias_is_broadcast, void* C,
    int64_t const* total_tokens_including_expert, HopperGroupedGemmInput hopper_input, int64_t total_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, ActivationType activation_type, bool use_fused_moe,
    float const** alpha_scale_ptr_array, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig chosen_conf)
{
    switch (activation_type)
    {
    case ActivationType::Relu:
        runGemm<cutlass_extensions::EpilogueOpDefaultReLU>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe,
            alpha_scale_ptr_array, stream, chosen_conf);
        break;
    case ActivationType::Gelu:
        runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe,
            alpha_scale_ptr_array, stream, chosen_conf);
        break;
    case ActivationType::Silu:
        runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe,
            alpha_scale_ptr_array, stream, chosen_conf);
        break;
    case ActivationType::Identity:
        runGemm<cutlass_extensions::EpilogueOpDefault>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe,
            alpha_scale_ptr_array, stream, chosen_conf);
        break;
    case ActivationType::Swiglu:
        runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe,
            alpha_scale_ptr_array, stream, chosen_conf);
        break;
    case ActivationType::Geglu:
        runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(A, B, weight_scales, biases, bias_is_broadcast, C,
            total_tokens_including_expert, hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe,
            alpha_scale_ptr_array, stream, chosen_conf);
        break;
    case ActivationType::InvalidType: THROW("Activation type for fpA_intB must be valid."); break;
    default: THROW("Invalid activation type."); break;
    }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleBiasType>
void MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>::moeGemm(T const* A, WeightType const* B,
    ScaleBiasType const* weight_scales, void* C, int64_t const* total_tokens_including_expert,
    HopperGroupedGemmInput hopper_input, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    bool use_fused_moe, float const** alpha_scale_ptr_array, cudaStream_t stream,
    cutlass_extensions::CutlassGemmConfig chosen_conf)
{
    runGemm<cutlass_extensions::EpilogueOpDefault>(A, B, weight_scales, nullptr, true, C, total_tokens_including_expert,
        hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, alpha_scale_ptr_array, stream,
        chosen_conf);
}

}
