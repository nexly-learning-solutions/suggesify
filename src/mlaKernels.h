
#pragma once

#include "../common/cudaUtils.h"
#include "../src/unfusedAttentionKernels.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace suggestify
{
namespace kernels
{

struct mlaMetaParams
{
    int32_t q_lora_rank;
    int32_t kv_lora_rank;
    int32_t qk_nope_head_dim;
    int32_t qk_rope_head_dim;
    int32_t v_head_dim;
};

template <typename T>
struct mlaParams
{
    T const* fused_a_input;
    T* attention_input_buf;
    T* context_buf;
    T const* fused_q_proj;
    T const* q_b_proj;
    T const* kv_b_proj;
    float2 const* cos_sin_cache;
    int32_t batch_size;
    int32_t acc_q_len;
    int32_t head_num;
    void* workspace;
    int32_t const* cache_seq_lens;
    int* seqQOffset;
    uint32_t* fmha_tile_counter;
    int32_t max_input_seq_len;
    int* cu_q_seqlens;
    int* cu_kv_seqlens;
    mlaMetaParams meta;
};

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContext(mlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGeneration(mlaParams<T>& params, KVCacheBuffer kv_cache_buffer, cudaStream_t stream);

}
}
