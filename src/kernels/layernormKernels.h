
#pragma once

#include "../common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace suggestify
{
namespace kernels
{

template <typename T>
void invokeGeneralLayerNorm(T* out, T const* input, T const* gamma, T const* beta, float const eps, int const tokens,
    int const hidden_dim, cudaStream_t stream = 0, bool use_diff_of_squares = true, float const* scale = nullptr,
    float* dynamic_scale = nullptr, int8_t* out_quant = nullptr);

}
}
