
#pragma once

#include "../kernels/kvCacheUtils.h"
#include "tmaDescriptor.h"
#include <limits.h>
#include <stdint.h>

#include "../kernels/multiHeadAttentionCommon.h"

namespace suggestify
{
namespace kernels
{


static constexpr int NUM_COMPUTE_GROUPS = 2;

static constexpr int FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT = 128;
static constexpr int FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT = 256;
static constexpr int FLASH_ATTEN_PACKED_MASK_MMA_M = 64;
static constexpr int FLASH_ATTEN_PACKED_MASK_MMA_N = 64;
static constexpr int FLASH_ATTEN_WARPS_M = 4;
static constexpr int FLASH_ATTEN_WARPS_N = 1;
static constexpr int NUM_POSITIONS_IN_UINT32 = 32;
static constexpr int NUM_THREADS_PER_WARP_GROUP = FLASH_ATTEN_WARPS_M * FLASH_ATTEN_WARPS_N * 32;
static constexpr int NUM_CORE_MMAS_N = FLASH_ATTEN_PACKED_MASK_MMA_N / 8;


enum class ContextFMHAType
{
    DISABLED,
    ENABLED,
    ENABLED_WITH_FP32_ACC
};

enum class ContextAttentionMaskType
{
    PADDING = 0,
    CAUSAL,
    SLIDING_WINDOW_CAUSAL,
    CUSTOM_MASK
};

enum class AttentionInputLayout
{
    PACKED_QKV = 0,
    Q_CONTIGUOUS_KV,
    Q_PAGED_KV
};


struct MHARunnerFixedParams
{
    Data_type dataType;
    bool forceFp32Acc;
    ContextAttentionMaskType attentionMaskType;
    AttentionInputLayout attentionInputLayout;
    bool isSPadded;
    int numQHeads;
    int numKvHeads;
    int headSize;
    int headSizeV = 0;
    float qScaling;
    float attnLogitSoftcappingScale;
    bool hasAlibi;
    bool scaleAlibi;
    int tpSize = 1;
    int tpRank = 0;

    std::string convertToStrOutput()
    {
        std::string output = "data_type = ";
        switch (dataType)
        {
        case DATA_TYPE_FP16: output += forceFp32Acc ? "fp16_fp32" : "fp16"; break;
        case DATA_TYPE_BF16: output += "bf16"; break;
        case DATA_TYPE_E4M3: output += "e4m3"; break;
        default: TLLM_CHECK_WITH_INFO(false, "not supported.");
        }
        output += ", head_size = " + std::to_string(headSize);
        output += ", head_size_V = " + std::to_string(headSizeV);
        output += ", attention_mask_type = ";
        switch (attentionMaskType)
        {
        case ContextAttentionMaskType::PADDING: output += "padding"; break;
        case ContextAttentionMaskType::CAUSAL: output += "causal"; break;
        case ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL: output += "sliding_window_causal"; break;
        case ContextAttentionMaskType::CUSTOM_MASK: output += "custom_mask"; break;
        default: TLLM_CHECK_WITH_INFO(false, "not supported.");
        }
        output += ", attention_input_layout = ";
        switch (attentionInputLayout)
        {
        case AttentionInputLayout::PACKED_QKV: output += "packed_qkv"; break;
        case AttentionInputLayout::Q_CONTIGUOUS_KV: output += "q_contiguous_kv"; break;
        case AttentionInputLayout::Q_PAGED_KV: output += "q_paged_kv"; break;
        default: TLLM_CHECK_WITH_INFO(false, "not supported.");
        }
        output += ", alibi = ";
        output += (hasAlibi ? "true" : "false");
        output += ", attn_logit_softcapping_scale = ";
        output += (attnLogitSoftcappingScale != 0.f ? "true" : "false");

        return output;
    }
};


struct MHARunnerParams
{
    int b;
    int qSeqLen;
    int kvSeqLen;
    int slidingWindowSize;
    int totalQSeqLen;
    int totalKvSeqLen;

    void const* qkvPtr;
    void const* qPtr;
    void const* kvPtr;
    KVBlockArray pagedKvCache;
    void* outputPtr;
    void const* packedMaskPtr;
    void const* cuQSeqLenPtr;
    void const* cuKvSeqLenPtr;
    void const* cuMaskRowsPtr;
    void* tileCounterPtr;
    float const* scaleBmm1Ptr;
    float const* scaleBmm2Ptr;
    cudaStream_t stream;
    bool forceFp32Acc = false;
};

struct AlibiParams
{
    constexpr static int round_down_to_power_two(int x)
    {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    AlibiParams() = default;

    AlibiParams(int h, float scale_after_alibi)
        : scale_after_alibi(scale_after_alibi)
    {
        h_pow_2 = round_down_to_power_two(h);
        alibi_neg4_div_h = -4.0f / h_pow_2;
    }

    AlibiParams(int h, int s, int tp_size, int rank, float scale_after_alibi)
        : AlibiParams(h * tp_size, scale_after_alibi)
    {
        head_idx_offset = h * rank;
        sequence_pos_offset = s * rank;
    }

    int h_pow_2{};
    float alibi_neg4_div_h{};
    float scale_after_alibi{};
    int head_idx_offset = 0;
    int sequence_pos_offset = 0;
};


struct Fused_multihead_attention_params_v2
{
    void const* qkv_ptr;
    void const* q_ptr;
    void const* kv_ptr;
    KVBlockArrayForContextFMHA paged_kv_cache;
    void const* packed_mask_ptr;
    void* o_ptr;

    int64_t qkv_stride_in_bytes;
    int64_t q_stride_in_bytes;
    int64_t kv_stride_in_bytes;
    int64_t packed_mask_stride_in_bytes;
    int64_t o_stride_in_bytes;

    cudaTmaDesc tma_desc_q;
    cudaTmaDesc tma_desc_kv;
    cudaTmaDesc tma_desc_o;

    int blocks_per_tma_load;
    int blocks_per_tma_load_log2;

    int b, h, h_kv, h_q_per_kv, s, d;
    int sliding_window_size = INT_MAX;
    uint32_t scale_bmm1, softcapping_scale_bmm1, scale_softmax, scale_bmm2;

    uint32_t const* scale_bmm1_d;
    uint32_t const* scale_bmm2_d;

    int const* cu_q_seqlens;
    int const* cu_kv_seqlens;
    int const* cu_mask_rows;

    bool has_alibi = false;
    AlibiParams alibi_params{};

    uint32_t* tile_id_counter_ptr;
    uint32_t num_tiles;
    uint32_t num_tiles_per_head;
    bool use_balanced_scheduling;

    bool is_s_padded = false;

    int dv = 0;
    int64_t v_stride_in_bytes = 0;
};


struct Launch_params
{
    int kernel_s = 0;
    int total_q_seqlen = 0;
    int total_kv_seqlen = 0;
    int padded_d = 0;
    bool ignore_b1opt = false;
    bool force_unroll = false;
    bool force_fp32_acc = false;
    bool interleaved = false;
    bool use_tma = false;
    bool flash_attention = false;
    bool warp_specialization = false;
    bool granular_tiling = false;
    bool dynamic_scheduler = false;
    ContextAttentionMaskType attention_mask_type = ContextAttentionMaskType::PADDING;
    AttentionInputLayout attention_input_layout = AttentionInputLayout::PACKED_QKV;
    bool useKernelWithoutAlibi = false;
    bool useBase2ExpTrick = false;
    bool enableAttnLogitSoftcapping = false;
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;
    size_t total_device_memory = 0;
};

}
}
