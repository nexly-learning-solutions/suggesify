
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../runtime/common.h"

namespace sugesstify
{
namespace kernels
{
using SizeType32 = sugesstify::runtime::SizeType32;

template <typename T>
size_t invokeComputeCumsumLastDimWorkspaceSize(SizeType32 inputLength);

template <typename T>
void invokeCumsumLastDim(SizeType32 batchSize, SizeType32 inputLength, void const* __restrict__ input,
    void* __restrict__ output, void* workspace, size_t tempStorageBytes, cudaStream_t stream);

}
}
