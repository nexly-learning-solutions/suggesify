
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"

namespace sugesstify
{
namespace kernels
{

template <typename T>
void invokeBuildRelativeAttentionBias(T* relative_attention_bias, T const* relative_attention_bias_table,
    int const head_num, int const seq_len, int const num_bucket, bool const is_bidirectional, int const max_distance,
    cudaStream_t stream);

}
}
