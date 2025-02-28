
#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

#include "../common/cudaUtils.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

enum class QuantType
{
    W8_A16,
    W4_A16,
    W4_AFP8
};

constexpr int get_weight_quant_bits(QuantType quant_type)
{
    switch (quant_type)
    {
    case QuantType::W8_A16: return 8;
    case QuantType::W4_A16: return 4;
    case QuantType::W4_AFP8: return 4;
    default: CHECK_WITH_INFO(false, "Invalid quant_type"); return -1;
    }
}

void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type, const int64_t arch_version);

void subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type);

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type);

void preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t const* row_major_quantized_weight,
    std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave = false);

template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, ComputeType* scale_ptr, WeightType const* input_weight_ptr,
    std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave);

template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, int8_t* unprocessed_quantized_weight,
    ComputeType* scale_ptr, WeightType const* input_weight_ptr, std::vector<size_t> const& shape, QuantType quant_type,
    bool force_interleave);

}
}
}
