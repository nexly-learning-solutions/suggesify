#pragma once

#include "../common/cudaUtils.h"
#include "../src/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "../runtime/iTensor.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

namespace tc = suggestify::common;

namespace suggestify
{
namespace kernels
{


static inline std::pair<int, int> roundUpPackedMaskMNDims(int m, int n)
{
    return std::make_pair(tc::roundUp(m, FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT),
        tc::roundUp(n, FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT) / NUM_POSITIONS_IN_UINT32);
}


template <typename MaskInputDataType>
struct PackedMaskParams
{
    MaskInputDataType const* maskInput = nullptr;
    int* cuQSeqLens = nullptr;
    uint32_t* packedMask = nullptr;
    int* cuMaskRows = nullptr;
    int const* actualQSeqLens = nullptr;
    int const* actualKvSeqLens = nullptr;
    ContextAttentionMaskType attentionMaskType = ContextAttentionMaskType::PADDING;
    int batchSize;
    int maxQSeqLen;
    int maxKvSeqLen;
    int slidingWindowSize;
    MaskInputDataType validPosVal = MaskInputDataType(1.0f);
};


template <typename MaskInputDataType>
void invokeBuildPackedMask(PackedMaskParams<MaskInputDataType> const& params, cudaStream_t stream);

}
}
