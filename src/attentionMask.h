#pragma once

#include "cudaUtils.h"
#include "suggestify/kernels/gptKernels.h"
#include "suggestify/runtime/iTensor.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

namespace tc = suggestify::common;

namespace suggestify
{
namespace kernels
{


template <typename MaskDataType>
struct AttentionMaskParams
{
    MaskDataType* mask = nullptr;
    int* cuQSeqLens = nullptr;
    int const* actualQSeqLens = nullptr;
    int const* actualKvSeqLens = nullptr;
    AttentionMaskType attentionMaskType = AttentionMaskType::PADDING;
    BlockSparseParams blockSparseParams;
    int batchSize;
    int maxQSeqLen;
    int maxKvSeqLen;
    int slidingWindowSize;
};


template <typename MaskDataType>
void invokeBuildAttentionMask(AttentionMaskParams<MaskDataType> const& params, cudaStream_t stream);

}
}
