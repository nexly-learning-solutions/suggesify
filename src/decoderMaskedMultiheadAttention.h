
#pragma once

#include "suggestify/common/cudaBf16Wrapper.h"
#include "suggestify/common/cudaFp8Utils.h"
#include "suggestify/kernels/gptKernels.h"
#include "suggestify/kernels/kvCacheUtils.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace suggestify
{
namespace kernels
{


#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)


inline int pow2roundup(int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}



template <typename T>
struct Multihead_attention_params_base
{

    void* out = nullptr;

    T const *q = nullptr, *q_bias = nullptr;
    T const *k = nullptr, *k_bias = nullptr;
    T const *v = nullptr, *v_bias = nullptr;

    int const* cache_indir = nullptr;

    float const* query_weight_output_scale = nullptr;
    float const* attention_qk_scale = nullptr;
    float const* attention_output_weight_input_scale_inv = nullptr;

    int stride = 0;

    int batch_size = 0;
    int beam_width = 0;
    int max_attention_window_size = 0;
    int cyclic_attention_window_size = 0;
    int sink_token_length = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int hidden_size_per_head = 0;
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::kLEARNED_ABSOLUTE;
    int rotary_embedding_dim = 0;
    float rotary_embedding_base = 0.0f;
    RotaryScalingType rotary_embedding_scale_type = RotaryScalingType::kNONE;
    float rotary_embedding_scale = 1.0f;
    float const* rotary_embedding_inv_freq_cache = nullptr;
    float rotary_embedding_short_m_scale = 1.0f;
    float rotary_embedding_long_m_scale = 1.0f;
    int rotary_embedding_max_positions = 0;
    int rotary_embedding_original_max_positions = 0;
    int rotary_cogvlm_vision_start = -1;
    int rotary_cogvlm_vision_length = -1;
    bool position_shift_enabled = false;
    int timestep = 0;

    float inv_sqrt_dh = 0.0f;

    float attn_logit_softcapping_scale = 0.0f;
    float attn_logit_softcapping_inverse_scale = 0.0f;

    bool const* attention_mask = nullptr;
    int attention_mask_stride = 0;

    T const* relative_attention_bias = nullptr;
    int relative_attention_bias_stride = 0;
    int max_distance = 0;

    float const* logn_scaling_ptr = nullptr;

    bool block_sparse_attention = false;
    BlockSparseParams block_sparse_params{64, false, 16, 8};

    T const* linear_bias_slopes = nullptr;

    T const* ia3_key_weights = nullptr;
    T const* ia3_value_weights = nullptr;
    int const* ia3_tasks = nullptr;

    float const* qkv_scale_quant_orig = nullptr;
    float const* attention_out_scale_orig_quant = nullptr;

    float const* kv_scale_orig_quant = nullptr;
    float const* kv_scale_quant_orig = nullptr;

    bool int8_kv_cache = false;
    bool fp8_kv_cache = false;

    mutable bool multi_block_mode = true;

    int multi_processor_count = 1;

    mutable int timesteps_per_block = 1;
    mutable int seq_len_tile = 1;

    mutable int min_seq_len_tile = 1;
    mutable int max_seq_len_tile = 1;
    T* partial_out = nullptr;
    float* partial_sum = nullptr;
    float* partial_max = nullptr;
    int* block_counter = nullptr;

    int const* memory_length_per_sample = nullptr;
    int32_t const* mrope_position_deltas = nullptr;
};

template <typename T, bool USE_CROSS_ATTENTION = false>
struct Multihead_attention_params;

template <typename T>
struct Multihead_attention_params<T, false> : public Multihead_attention_params_base<T>
{
    static constexpr bool DO_CROSS_ATTENTION = false;

    int max_decoder_seq_len = 0;

    bool* finished = nullptr;

    int const* length_per_sample = nullptr;

    int const* input_lengths = nullptr;
};
template <class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;

template <typename T>
struct Multihead_attention_params<T, true> : public Multihead_attention_params_base<T>
{
    static constexpr bool DO_CROSS_ATTENTION = true;

    int max_decoder_seq_len = 0;

    bool* finished = nullptr;

    int const* length_per_sample = nullptr;

    int const* input_lengths = nullptr;
};
template <class T>
using Cross_multihead_attention_params = Multihead_attention_params<T, true>;


bool mmha_supported(int head_size);

#define DECLARE_MMHA_NORMAL_AND_PAGED(T)                                                                               \
    void masked_multihead_attention(const Masked_multihead_attention_params<T>& params,                                \
        const KVBlockArray& block_array, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);             \
    void masked_multihead_attention(const Masked_multihead_attention_params<T>& params,                                \
        const KVLinearBuffer& kv_cache_buffer, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);       \
    void masked_multihead_attention(const Cross_multihead_attention_params<T>& params,                                 \
        const KVBlockArray& block_array, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);             \
    void masked_multihead_attention(const Cross_multihead_attention_params<T>& params,                                 \
        const KVLinearBuffer& kv_cache_buffer, const KVLinearBuffer& shift_k_cache, const cudaStream_t& stream);
DECLARE_MMHA_NORMAL_AND_PAGED(float);
DECLARE_MMHA_NORMAL_AND_PAGED(uint16_t);
#ifdef ENABLE_BF16
DECLARE_MMHA_NORMAL_AND_PAGED(__nv_bfloat16);
#endif
#undef DECLARE_MMHA_NORMAL_AND_PAGED


template <typename T>
inline int estimate_min_multi_block_count(int max_timesteps, int max_dynamic_shmem_per_block)
{
    auto const qk_elts = static_cast<int>((max_timesteps + 1 + 4 - 1) / 4);
    int size_per_elts = 16;
    auto const qk_sz = qk_elts * 16;
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4)
    {
        size_per_elts += 4 * sizeof(T);
    }
#endif
    int elts_per_block = max_dynamic_shmem_per_block / size_per_elts;
    int min_block_count = (qk_elts + elts_per_block - 1) / elts_per_block;
    return std::max(1, min_block_count);
}

}
}
