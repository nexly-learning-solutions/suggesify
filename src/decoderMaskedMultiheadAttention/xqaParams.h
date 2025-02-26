
#pragma once
#include "quantization.h"
#include "suggestify/kernels/gptKernels.h"
#include "suggestify/kernels/multiHeadAttentionCommon.h"

namespace suggestify
{
namespace kernels
{

using XQADataType = Data_type;

struct XQAParams
{
    XQADataType data_type = DATA_TYPE_FP16;
    XQADataType kv_cache_data_type = DATA_TYPE_FP16;
    void* output = nullptr;
    void const* qkv = nullptr;
    int32_t const* cache_indir = nullptr;
    float const* kv_scale_orig_quant = nullptr;
    float const* kv_scale_quant_orig = nullptr;
    int32_t const* host_past_key_value_lengths = nullptr;
    int32_t const* host_context_lengths = nullptr;
    int32_t* semaphores = nullptr;
    void* workspaces = nullptr;
    uint32_t batch_size = 0;
    int32_t beam_width = 0;
    int32_t max_attention_window_size = 0;
    int32_t cyclic_attention_window_size = 0;
    int32_t sink_token_length = 0;
    int timestep = 0;
    void const* qkv_bias;
    int32_t const* sequence_lengths;                  //
    int32_t const* context_lengths;                   // maybe not used now
    void const* alibi_slopes;                         // maybe not used now
    float const* rotary_embedding_inv_freq_cache;     // precomputed rotary inv freq
    int32_t const* spec_decoding_packed_mask;
    int const* spec_decoding_position_offsets;        // for position embedding.
    int const* spec_decoding_generation_lengths;      // variable input lengths.
    bool spec_decoding_is_generation_length_variable; // whether the generation lengths actually vary
    int32_t spec_decoding_max_generation_length;      // max possible input length
    int32_t const* mrope_position_deltas = nullptr;

    // almost copy from GPTAttentionPluginCommon.
    // maybe use one struct for parameters in GPTAttentionPluginCommon and share the same here.
    int32_t generation_input_length;
    int32_t layer_idx = 0;
    int32_t num_q_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_size = 0;
    int unidirectional;
    float q_scaling = 0;
    int32_t rotary_embedding_dim = 0;
    float rotary_embedding_base = 0.0f;
    suggestify::kernels::RotaryScalingType rotary_embedding_scale_type;
    float rotary_embedding_scale;
    int rotary_embedding_max_positions;
    int rotary_vision_start;
    int rotary_vision_length;
    suggestify::kernels::PositionEmbeddingType position_embedding_type;
    bool position_shift_enabled = false;
    bool remove_padding = false;
    suggestify::kernels::AttentionMaskType mask_type;
    // Paged KV cache parameters.
    bool paged_kv_cache;
    int tokens_per_block;
    int max_blocks_per_sequence;
    suggestify::common::QuantMode kv_cache_quant_mode;
    int tp_size = 1;
    int tp_rank = 0;
    bool qkv_bias_enabled;
    bool cross_attention;
    int max_distance = 0;
    bool multi_block_mode;
    bool multi_query_tokens = false;

    float const* logn_scaling_ptr = nullptr; // for logn scaling in XQA

    int32_t total_num_input_tokens;          // total number of input tokens. may differ from batch_size due to medusa.
    float const* fp8_out_scale = nullptr; // fp8 output scale in case we need post-processing to convert output to fp8.
                                          // nullptr means no conversion.
};

} // namespace kernels
} // namespace suggestify
