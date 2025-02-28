
#pragma once

#include "../common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace suggestify
{
namespace kernels
{
template <typename Tout, typename Tin, typename Idx>
void invokeLookUp(Tout* out, Idx const* input, Tin const* weight, int64_t const token_num, Idx const offset,
    Idx const size, Idx const n_embed, Tout const* perTokenScales, cudaStream_t stream = 0);

}
}
