
#pragma once

#include "../runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace suggestify
{
namespace kernels
{

template <typename T>
void invokeLogitsBitmask(
    T** logits, uint32_t const** bitmask, int32_t batchSize, int32_t vocabSizePadded, cudaStream_t stream);

}
}
