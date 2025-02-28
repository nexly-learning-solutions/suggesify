#pragma once

#include "../common/quantization.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace suggestify
{
namespace kernels
{

template <typename T>
void invokeQuantization(
    int8_t* dst, T const* src, int64_t const size, float const* scalePtr, cudaStream_t stream = 0, int maxGirdSize = 0);

template <typename T, typename QuantT>
void invokePerTokenQuantization(QuantT* dst, T const* src, int64_t const numRows, int64_t const numCols,
    float const* clampPtr, float* scalePtr, float* sumPtr, suggestify::common::QuantMode quantMode,
    cudaStream_t stream = 0);

}
}
