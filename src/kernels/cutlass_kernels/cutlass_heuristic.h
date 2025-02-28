
#pragma once

#include "cute/tensor.hpp"
#include "cutlass_extensions/gemm_configs.h"
#include "../common/cudaUtils.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

template <class TileShape, class ClusterShape, class ActivationType>
struct should_filter_sm90_gemm_problem_shape
{
#ifdef FAST_BUILD
    constexpr static int TILE_K = 128 * 8 / cutlass::sizeof_bits<ActivationType>::value;
    using SupportedCtaShape = cute::Shape<cute::_128, cute::_128, cute::Int<TILE_K>>;
    using SupportedCgaShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    constexpr static bool value
        = !cute::is_same_v<SupportedCtaShape, TileShape> || !cute::is_same_v<SupportedCgaShape, ClusterShape>;
#else
    constexpr static bool value = false;
#endif
};
template <class TileShape, class ClusterShape, class ActivationType>
constexpr static bool should_filter_sm90_gemm_problem_shape_v
    = should_filter_sm90_gemm_problem_shape<TileShape, ClusterShape, ActivationType>::value;

std::vector<suggestify::cutlass_extensions::CutlassGemmConfig> get_candidate_configs(
    int sm, int const max_split_k, suggestify::cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam const);

suggestify::cutlass_extensions::CutlassGemmConfig estimate_best_config_from_occupancies(
    std::vector<suggestify::cutlass_extensions::CutlassGemmConfig> const& candidate_configs,
    std::vector<int> const& occupancies, int64_t const m, int64_t const n, int64_t const k, int64_t const num_experts,
    int const split_k_limit, size_t const workspace_bytes, int const multi_processor_count, int const is_weight_only);

}
}
}
