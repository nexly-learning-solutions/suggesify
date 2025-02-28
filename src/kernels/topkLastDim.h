
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../runtime/common.h"

namespace suggestify
{
namespace kernels
{

template <typename T>
size_t invokeComputeTopkLastDimWorkspaceSize(
    runtime::SizeType32 batchSize, runtime::SizeType32 inputLength, runtime::SizeType32 k, bool is_largest);

template <typename T>
void invokeTopkLastDim(runtime::SizeType32 batchSize, runtime::SizeType32 inputLength, runtime::SizeType32 k,
    bool is_largest, void const* __restrict__ input, void* __restrict__ out_val, void* __restrict__ out_ind,
    void* workspace, cudaStream_t stream);

}
}
