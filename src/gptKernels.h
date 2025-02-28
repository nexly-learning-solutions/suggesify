#pragma once

#include "../src/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "../runtime/iTensor.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

namespace suggestify
{
namespace kernels
{

enum class AttentionMaskType
{
    PADDING = 0,
    CAUSAL = 1,
    SLIDING_WINDOW_CAUSAL = 2,
    BIDIRECTIONAL = 3,
    BIDIRECTIONALGLM = 4,
    BLOCKSPARSE = 5,
    CUSTOM_MASK = 6,
};

enum class PositionEmbeddingType : int8_t
{
    kLEARNED_ABSOLUTE = 0,
    kROPE_GPTJ = 1,
    kROPE_GPT_NEOX = 2,
    kLONG_ROPE = 3,
    kALIBI = 4,
    kALIBI_WITH_SCALE = 5,
    kRELATIVE = 6,
    kCHATGLM = 7,
    kYARN = 8,
    kROPE_M = 9,
};

enum class RotaryScalingType : int8_t
{
    kNONE = 0,
    kLINEAR = 1,
    kDYNAMIC = 2,
    kLONG = 3,
    kLLAMA3 = 4
};

struct BlockSparseParams
{
    int block_size;
    int homo_head_pattern;
    int num_local_blocks;
    int vertical_stride;

    __device__ bool computeMask(
        int row_idx, int col_idx, int q_seq_length, int kv_seq_length, int num_heads, int head_idx) const
    {
        bool causal_mask = row_idx < q_seq_length && col_idx < kv_seq_length && col_idx <= row_idx;

        int block_row_idx = row_idx / block_size;
        int block_col_idx = col_idx / block_size;

        bool block_local_mask = (block_row_idx - block_col_idx) < num_local_blocks;

        int head_sliding_step = homo_head_pattern ? 0 : std::max(1, int(vertical_stride / num_heads));
        bool block_vertical_stride_mask = ((block_col_idx + head_idx * head_sliding_step + 1) % vertical_stride) == 0;

        bool is_valid = causal_mask && (block_local_mask || block_vertical_stride_mask);
        return is_valid;
    }

    __device__ bool computeMask(int row_idx, int col_idx, int seq_length, int num_heads, int head_idx) const
    {
        return computeMask(row_idx, col_idx, seq_length, seq_length, num_heads, head_idx);
    }
};

template <typename AttentionMaskDataType>
struct BuildDecoderInfoParams
{
    int* seqQOffsets;
    int* seqKVOffsets;
    int* paddingOffsets;
    int* encoderPaddingOffsets;
    int* packedMaskRowOffsets;
    int* seqCpPartialOffsets;

    AttentionMaskDataType* attentionMask;

    int const* seqQLengths;
    int const* seqKVLengths;
    int cpSize;

    uint32_t* fmhaTileCounter;

    float const* dequantScaleQkv;
    float const* quantScaleO;
    float fmhaHostBmm1Scale;
    float* fmhaBmm1Scale;
    float* fmhaBmm2Scale;

    int batchSize;
    int maxQSeqLength;
    int maxEncoderQSeqLength;
    int attentionWindowSize;
    int sinkTokenLength;
    int numTokens;
    AttentionMaskType attentionMaskType;
    BlockSparseParams blockSparseParams;

    float rotaryEmbeddingScale;
    float rotaryEmbeddingBase;
    int rotaryEmbeddingDim;
    RotaryScalingType rotaryScalingType;
    float* rotaryEmbeddingInvFreq;
    float const* rotaryEmbeddingInvFreqCache;
    float2* rotaryEmbeddingCoeffCache;
    int rotaryEmbeddingMaxPositions;

    std::string toString() const
    {
        std::stringstream ss;
        ss << "BuildDecoderInfoParams ====================" << std::endl;
        ss << "seqQOffsets: "
           << *(runtime::ITensor::wrap(
                  (void*) seqQOffsets, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batchSize})))
           << std::endl;
        ss << "seqKVOffsets: "
           << *(runtime::ITensor::wrap(
                  (void*) seqKVOffsets, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batchSize})))
           << std::endl;
        ss << "paddingOffsets: "
           << *(runtime::ITensor::wrap(
                  (void*) paddingOffsets, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batchSize})))
           << std::endl;
        if (encoderPaddingOffsets != nullptr)
        {
            ss << "encoderPaddingOffsets: "
               << *(runtime::ITensor::wrap((void*) encoderPaddingOffsets, nvinfer1::DataType::kINT32,
                      runtime::ITensor::makeShape({batchSize})))
               << std::endl;
        }
        ss << "attentionMask: " << static_cast<void*>(attentionMask) << std::endl;
        ss << "seqQLengths: " << seqQLengths << std::endl;
        ss << "seqKVLengths: " << seqKVLengths << std::endl;
        ss << "fmhaTileCounter: " << fmhaTileCounter << std::endl;
        ss << "batchSize: " << batchSize << std::endl;
        ss << "maxQSeqLength: " << maxQSeqLength << std::endl;
        ss << "maxEncoderQSeqLength: " << maxEncoderQSeqLength << std::endl;
        ss << "attentionWindowSize: " << attentionWindowSize << std::endl;
        ss << "sinkTokenLength: " << sinkTokenLength << std::endl;
        ss << "numTokens: " << numTokens << std::endl;
        ss << "attentionMaskType: " << static_cast<int>(attentionMaskType) << std::endl;
        ss << "rotaryEmbeddingScale: " << rotaryEmbeddingScale << std::endl;
        ss << "rotaryEmbeddingBase: " << rotaryEmbeddingBase << std::endl;
        ss << "rotaryEmbeddingDim: " << rotaryEmbeddingDim << std::endl;
        ss << "rotaryScalingType: " << static_cast<int>(rotaryScalingType) << std::endl;
        ss << "rotaryEmbeddingInvFreq: " << rotaryEmbeddingInvFreq << std::endl;
        ss << "rotaryEmbeddingInvFreqCache: " << rotaryEmbeddingInvFreqCache << std::endl;
        ss << "rotaryEmbeddingCoeffCache: " << rotaryEmbeddingCoeffCache << std::endl;
        ss << "rotaryEmbeddingMaxPositions: " << rotaryEmbeddingMaxPositions << std::endl;

        return ss.str();
    }
};

template <typename T>
void invokeBuildDecoderInfo(BuildDecoderInfoParams<T> const& params, cudaStream_t stream);

}
}
