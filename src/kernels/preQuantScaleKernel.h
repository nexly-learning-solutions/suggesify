#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#if defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif

#include <type_traits>
#include <vector>

namespace suggestify
{
namespace kernels
{

template <typename T_in, typename T_out = T_in>
void apply_per_channel_scale_kernel_launcher(
    T_out* smoothed_act, T_in const* act, T_in const* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

}
}
