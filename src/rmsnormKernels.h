
#pragma once

#include "../common/cudaUtils.h"
#include "../common/quantization.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sugesstify
{
namespace kernels
{

template <typename T, typename QuantT>
void invokeGeneralRmsNorm(T* out, T const* input, T const* gamma, T const* beta, float const eps, int const tokens,
    int const hidden_dim, sugesstify::common::QuantMode quantMode, cudaStream_t stream = 0,
    float const* clampPtr = nullptr, float const* scale = nullptr, float* dynamic_scale = nullptr,
    float* sum_per_token = nullptr, QuantT* out_quant = nullptr);

}
}
