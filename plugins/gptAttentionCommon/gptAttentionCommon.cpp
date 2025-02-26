#include "gptAttentionCommon.h"
#include "../common/assert.h"
#include "../common/dataType.h"
#include "../common/envUtils.h"
#include "../common/memoryUtils.h"
#include "../common/mpiUtils.h"
#include "../src/decoderMaskedMultiheadAttention.h"
#include "../src/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "../src/gptKernels.h"
#include "../src/kvCacheUtils.h"
#include "../src/rmsnormKernels.h"
#include "../src/unfusedAttentionKernels.h"
#include "../plugins/common/checkMacrosPlugin.h"
#include "../runtime/iBuffer.h"
#include "../runtime/utils/debugUtils.h"
#include <NvInferRuntimePlugin.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

using namespace nvinfer1;
using namespace suggestify::kernels;
namespace tc = suggestify::common;
using suggestify::plugins::GPTAttentionPluginCreatorCommon;
using suggestify::plugins::GPTAttentionPluginCommon;

template <typename T>
struct SATypeConverter
{
    using Type = T;
};

template <>
struct SATypeConverter<half>
{
    using Type = uint16_t;
};

template <typename T, typename KVCacheBuffer>
struct FusedQKVMaskedAttentionDispatchParams
{
    T const* qkv_buf;
    T const* qkv_bias;
    T const* relative_attention_bias;
    bool const* attention_mask;
    float const* logn_scaling_ptr;
    int const* cache_indir;
    void* context_buf;
    bool const* finished;
    int const* sequence_lengths;
    int max_batch_size;
    int inference_batch_size;
    int beam_width;
    int head_num;
    int kv_head_num;
    int size_per_head;
    int rotary_embedding_dim;
    float rotary_embedding_base;
    RotaryScalingType rotary_embedding_scale_type;
    float rotary_embedding_scale;
    float const* rotary_embedding_inv_freq_cache;
    float rotary_embedding_short_m_scale;
    float rotary_embedding_long_m_scale;
    int rotary_embedding_max_positions;
    int rotary_embedding_original_max_positions;
    int rotary_cogvlm_vision_start;
    int rotary_cogvlm_vision_length;
    PositionEmbeddingType position_embedding_type;
    bool position_shift_enabled;

    int attention_mask_stride;
    int max_attention_window;
    int cyclic_attention_window_size;
    int sink_token_length;
    int const* input_lengths;
    int timestep;
    float q_scaling;
    float attn_logit_softcapping_scale;
    int relative_attention_bias_stride;
    T const* linear_bias_slopes;
    int const* ia3_tasks;
    T const* ia3_key_weights;
    T const* ia3_value_weights;
    float const* qkv_scale_out;
    bool fp8_context_fmha;
    float const* attention_out_scale;
    bool mUnfuseQkvGemm;
    tc::QuantMode quant_option;
    bool multi_block_mode;
    int max_seq_len_tile;
    int min_seq_len_tile;
    T* partial_out;
    float* partial_sum;
    float* partial_max;
    int* block_counter;
    float const* kv_scale_orig_quant;
    float const* kv_scale_quant_orig;
    tc::QuantMode kv_cache_quant_mode;
    int multi_processor_count;
    KVCacheBuffer kv_block_array;
    KVLinearBuffer shift_k_cache_buffer;
    bool cross_attention = false;
    int const* memory_length_per_sample = nullptr;
    int max_distance = 0;
    bool block_sparse_attention = false;
    BlockSparseParams block_sparse_params;
    int32_t const* mrope_position_deltas;
};

template <typename T, typename KVCacheBuffer>
struct ConvertMMHAToXQAParamsHelper
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = false;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__half, KVLinearBuffer>
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = true;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__half, KVBlockArray>
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = true;
};

#ifdef ENABLE_BF16
template <>
struct ConvertMMHAToXQAParamsHelper<__nv_bfloat16, KVLinearBuffer>
{
    static constexpr Data_type data_type = DATA_TYPE_BF16;
    static constexpr bool supported = true;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__nv_bfloat16, KVBlockArray>
{
    static constexpr Data_type data_type = DATA_TYPE_BF16;
    static constexpr bool supported = true;
};
#endif

template <typename T, typename KVCacheBuffer>
bool GPTAttentionPluginCommon::convertMMHAParamsToXQAParams(suggestify::kernels::XQAParams& xqaParams,
    EnqueueGenerationParams<T> const& generationsParams, bool forConfigurePlugin)
{
    bool retval = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::supported;
    if (!retval)
    {
        return false;
    }
    memset(&xqaParams, 0, sizeof(XQAParams));
    xqaParams.data_type = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::data_type;

    xqaParams.layer_idx = mLayerIdxInCachePool;
    xqaParams.num_q_heads = mNumHeads / mCpSize;
    xqaParams.num_kv_heads = mNumKVHeads / mCpSize;
    xqaParams.head_size = mHeadSize;
    xqaParams.unidirectional = mUnidirectional;
    xqaParams.q_scaling = mQScaling;
    xqaParams.rotary_embedding_dim = mRotaryEmbeddingDim;
    xqaParams.rotary_embedding_base = mRotaryEmbeddingBase;
    xqaParams.rotary_embedding_scale_type = mRotaryEmbeddingScaleType;
    xqaParams.rotary_embedding_scale = mRotaryEmbeddingScale;
    xqaParams.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
    xqaParams.rotary_vision_start = mVisionStart;
    xqaParams.rotary_vision_length = mVisionLength;
    xqaParams.position_embedding_type = mPositionEmbeddingType;
    xqaParams.position_shift_enabled = mPosShiftEnabled;
    xqaParams.remove_padding = mRemovePadding;
    xqaParams.mask_type = mMaskType;
    xqaParams.paged_kv_cache = mPagedKVCache;
    xqaParams.tokens_per_block = mTokensPerBlock;
    xqaParams.kv_cache_quant_mode = mKVCacheQuantMode;
    xqaParams.tp_size = mTpSize;
    xqaParams.tp_rank = mTpRank;
    xqaParams.qkv_bias_enabled = mQKVBiasEnabled;
    xqaParams.cross_attention = mCrossAttention;
    xqaParams.max_distance = mMaxDistance;
    xqaParams.multi_block_mode = mMultiBlockMode;
    xqaParams.multi_query_tokens = mIsSpecDecodingEnabled && mUseSpecDecoding;

    if (mKVCacheQuantMode.hasInt8KvCache())
    {
        xqaParams.kv_cache_data_type = DATA_TYPE_INT8;
    }
    else if (mKVCacheQuantMode.hasFp8KvCache())
    {
        xqaParams.kv_cache_data_type = DATA_TYPE_E4M3;
    }
    else
    {
        xqaParams.kv_cache_data_type = xqaParams.data_type;
    }
    if (xqaParams.kv_cache_data_type == DATA_TYPE_INT8
        || (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 && mSM != kSM_90))
    {
        xqaParams.multi_block_mode = false;
    }

    xqaParams.output = generationsParams.context_buf;
    xqaParams.qkv = generationsParams.attention_input;
    xqaParams.cache_indir = generationsParams.cache_indir;
    xqaParams.kv_scale_orig_quant = generationsParams.kv_scale_orig_quant;
    xqaParams.kv_scale_quant_orig = generationsParams.kv_scale_quant_orig;
    xqaParams.host_past_key_value_lengths = generationsParams.host_past_key_value_lengths;
    xqaParams.host_context_lengths = generationsParams.host_context_lengths;
    xqaParams.semaphores = generationsParams.semaphores;
    xqaParams.workspaces = generationsParams.workspace;
    xqaParams.batch_size = generationsParams.num_requests;
    xqaParams.beam_width = generationsParams.beam_width;
    xqaParams.generation_input_length = generationsParams.input_seq_length;
    xqaParams.max_attention_window_size = generationsParams.max_attention_window;
    xqaParams.cyclic_attention_window_size = generationsParams.cyclic_attention_window_size;
    xqaParams.max_blocks_per_sequence = generationsParams.max_blocks_per_sequence;
    xqaParams.sink_token_length = generationsParams.sink_token_length;
    xqaParams.timestep = generationsParams.max_past_kv_length;
    xqaParams.qkv_bias = generationsParams.qkv_bias;
    xqaParams.sequence_lengths = generationsParams.sequence_lengths;
    xqaParams.context_lengths = generationsParams.context_lengths;
    xqaParams.alibi_slopes = generationsParams.alibi_slopes;
    xqaParams.rotary_embedding_inv_freq_cache = generationsParams.rotary_inv_freq;
    if (!forConfigurePlugin)
    {
        TLLM_CHECK_WITH_INFO(
            !(mIsSpecDecodingEnabled && mUseSpecDecoding) || generationsParams.spec_decoding_packed_mask != nullptr,
            "Speculative decoding mode needs a valid packed_mask input tensor.");
    }
    xqaParams.spec_decoding_packed_mask = generationsParams.spec_decoding_packed_mask;
    xqaParams.spec_decoding_position_offsets = generationsParams.spec_decoding_position_offsets;
    xqaParams.spec_decoding_generation_lengths = generationsParams.spec_decoding_generation_lengths;
    xqaParams.spec_decoding_is_generation_length_variable
        = generationsParams.spec_decoding_is_generation_length_variable;
    xqaParams.spec_decoding_max_generation_length = generationsParams.spec_decoding_max_generation_length;
    xqaParams.mrope_position_deltas = generationsParams.mrope_position_deltas;

    xqaParams.logn_scaling_ptr = generationsParams.logn_scaling_ptr;
    xqaParams.total_num_input_tokens = generationsParams.total_num_input_tokens;
    xqaParams.fp8_out_scale = (mFP8ContextFMHA ? generationsParams.attention_output_orig_quant : nullptr);
    return true;
}

template <typename T_MMHA, typename T, typename KVCacheBuffer, bool CROSS_ATTENTION>
void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, CROSS_ATTENTION>& params,
    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> const& input_params, cudaStream_t stream)
{
    using DataType = typename SATypeConverter<T>::Type;

    memset(&params, 0, sizeof(params));

    int hidden_units = input_params.head_num * input_params.size_per_head;
    int hidden_units_kv = input_params.kv_head_num * input_params.size_per_head;
    if (input_params.qkv_bias != nullptr)
    {
        params.q_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias);
        params.k_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias) + hidden_units + hidden_units_kv;
    }
    else
    {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }

    params.out = input_params.context_buf;

    params.q = reinterpret_cast<DataType const*>(input_params.qkv_buf);
    params.k = reinterpret_cast<DataType const*>(input_params.qkv_buf) + hidden_units;
    params.v = reinterpret_cast<DataType const*>(input_params.qkv_buf) + hidden_units + hidden_units_kv;

    params.int8_kv_cache = input_params.kv_cache_quant_mode.hasInt8KvCache();
    params.fp8_kv_cache = input_params.kv_cache_quant_mode.hasFp8KvCache();
    if (input_params.kv_cache_quant_mode.hasKvCacheQuant())
    {
        params.kv_scale_orig_quant = input_params.kv_scale_orig_quant;
        params.kv_scale_quant_orig = input_params.kv_scale_quant_orig;
    }

    params.stride = hidden_units + 2 * hidden_units_kv;
    params.finished = const_cast<bool*>(input_params.finished);

    params.cache_indir = input_params.cache_indir;
    params.batch_size = input_params.inference_batch_size;
    params.beam_width = input_params.beam_width;
    params.max_attention_window_size = input_params.max_attention_window;
    params.cyclic_attention_window_size = input_params.cyclic_attention_window_size;
    params.sink_token_length = input_params.sink_token_length;
    params.length_per_sample = input_params.sequence_lengths;
    params.timestep = input_params.timestep;
    params.num_heads = input_params.head_num;
    params.num_kv_heads = input_params.kv_head_num;
    params.hidden_size_per_head = input_params.size_per_head;
    params.rotary_embedding_dim = input_params.rotary_embedding_dim;
    params.rotary_embedding_base = input_params.rotary_embedding_base;
    params.rotary_embedding_scale_type = input_params.rotary_embedding_scale_type;
    params.rotary_embedding_scale = input_params.rotary_embedding_scale;
    params.rotary_embedding_inv_freq_cache = input_params.rotary_embedding_inv_freq_cache;
    params.rotary_embedding_short_m_scale = input_params.rotary_embedding_short_m_scale;
    params.rotary_embedding_long_m_scale = input_params.rotary_embedding_long_m_scale;
    params.rotary_embedding_max_positions = input_params.rotary_embedding_max_positions;
    params.rotary_embedding_original_max_positions = input_params.rotary_embedding_original_max_positions;
    params.rotary_cogvlm_vision_start = input_params.rotary_cogvlm_vision_start;
    params.rotary_cogvlm_vision_length = input_params.rotary_cogvlm_vision_length;
    params.position_embedding_type = input_params.position_embedding_type;
    params.position_shift_enabled = input_params.position_shift_enabled;
    params.inv_sqrt_dh = 1.F / (sqrtf((float) params.hidden_size_per_head) * input_params.q_scaling);
    params.attn_logit_softcapping_scale = input_params.attn_logit_softcapping_scale;
    params.attn_logit_softcapping_inverse_scale = 1.0f / input_params.attn_logit_softcapping_scale;

    params.logn_scaling_ptr = input_params.logn_scaling_ptr;
    params.relative_attention_bias = reinterpret_cast<DataType const*>(input_params.relative_attention_bias);
    params.relative_attention_bias_stride = input_params.relative_attention_bias_stride;
    params.max_distance = input_params.max_distance;
    params.block_sparse_attention = input_params.block_sparse_attention;
    params.block_sparse_params = input_params.block_sparse_params;

    params.attention_mask = input_params.attention_mask;
    params.attention_mask_stride = input_params.attention_mask_stride;

    if (input_params.linear_bias_slopes != nullptr)
    {
        params.linear_bias_slopes = reinterpret_cast<DataType const*>(input_params.linear_bias_slopes);
    }
    params.input_lengths = input_params.input_lengths;

    params.ia3_tasks = input_params.ia3_tasks;
    params.ia3_key_weights = reinterpret_cast<DataType const*>(input_params.ia3_key_weights);
    params.ia3_value_weights = reinterpret_cast<DataType const*>(input_params.ia3_value_weights);

    if (input_params.quant_option.hasStaticActivationScaling() || input_params.fp8_context_fmha)
    {
        params.qkv_scale_quant_orig = input_params.qkv_scale_out;
        TLLM_CHECK_WITH_INFO(!input_params.fp8_context_fmha || input_params.attention_out_scale != nullptr,
            "attention output scale should be provided.");
        params.attention_out_scale_orig_quant = input_params.attention_out_scale;
    }

    params.multi_block_mode = input_params.multi_block_mode;
    if (input_params.multi_block_mode)
    {
        params.min_seq_len_tile = input_params.min_seq_len_tile;
        params.max_seq_len_tile = input_params.max_seq_len_tile;

        params.partial_out = reinterpret_cast<DataType*>(input_params.partial_out);
        params.partial_sum = input_params.partial_sum;
        params.partial_max = input_params.partial_max;

        params.block_counter = input_params.block_counter;
    }

    params.multi_processor_count = input_params.multi_processor_count;

    params.memory_length_per_sample = input_params.memory_length_per_sample;

    params.mrope_position_deltas = input_params.mrope_position_deltas;
    sync_check_cuda_error();

    masked_multihead_attention(params, input_params.kv_block_array, input_params.shift_k_cache_buffer, stream);
}

#define INSTANTIATE_MMHA_DISPATCH(T_MMHA, T)                                                                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer> const&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer> const&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray> const&, cudaStream_t stream);                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray> const&, cudaStream_t stream);
INSTANTIATE_MMHA_DISPATCH(float, float)
INSTANTIATE_MMHA_DISPATCH(uint16_t, half)
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_DISPATCH(__nv_bfloat16, __nv_bfloat16)
#endif
#undef INSTANTIATE_MMHA_DISPATCH

GPTAttentionPluginCommon::GPTAttentionPluginCommon(int layer_idx, int num_heads, int vision_start, int vision_length,
    int num_kv_heads, int layer_idx_in_cache_pool, int head_size, int unidirectional, float q_scaling,
    float attn_logit_softcapping_scale, suggestify::kernels::PositionEmbeddingType position_embedding_type,
    int rotary_embedding_dim,
    float rotary_embedding_base, suggestify::kernels::RotaryScalingType rotary_embedding_scale_type,
    float rotary_embedding_scale, float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale,
    int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, int tp_size,
    int tp_rank,
    bool unfuse_qkv_gemm,
    bool use_logn_scaling,
    suggestify::kernels::ContextFMHAType context_fmha_type, int kv_cache_quant_mode, bool remove_input_padding,
    suggestify::kernels::AttentionMaskType mask_type, suggestify::kernels::BlockSparseParams block_sparse_params,
    bool paged_kv_cache, int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length,
    bool qkv_bias_enabled, bool cross_attention, int max_distance, bool pos_shift_enabled, bool dense_context_fmha,
    bool use_paged_context_fmha, bool use_fp8_context_fmha, bool has_full_attention_mask, bool use_cache,
    bool is_spec_decoding_enabled, bool spec_decoding_is_generation_length_variable,
    int32_t spec_decoding_max_generation_length, bool is_mla_enabled, int q_lora_rank, int kv_lora_rank,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim, bool skip_attn, int cp_size, int cp_rank,
    std::set<int32_t> cp_group)
    : mLayerIdx(layer_idx)
    , mNumHeads(num_heads)
    , mVisionStart(vision_start)
    , mVisionLength(vision_length)
    , mNumKVHeads(num_kv_heads)
    , mLayerIdxInCachePool(layer_idx_in_cache_pool)
    , mHeadSize(head_size)
    , mUnidirectional(unidirectional)
    , mQScaling(q_scaling)
    , mAttnLogitSoftcappingScale(attn_logit_softcapping_scale)
    , mRotaryEmbeddingDim(rotary_embedding_dim)
    , mRotaryEmbeddingBase(rotary_embedding_base)
    , mRotaryEmbeddingScaleType(rotary_embedding_scale_type)
    , mRotaryEmbeddingScale(rotary_embedding_scale)
    , mRotaryEmbeddingShortMscale(rotary_embedding_short_m_scale)
    , mRotaryEmbeddingLongMscale(rotary_embedding_long_m_scale)
    , mRotaryEmbeddingMaxPositions(rotary_embedding_max_positions)
    , mRotaryEmbeddingOriginalMaxPositions(rotary_embedding_original_max_positions)
    , mPositionEmbeddingType(position_embedding_type)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::DISABLED)
    , mFMHAForceFP32Acc(type == nvinfer1::DataType::kBF16)
    , mMaskType(mask_type)
    , mBlockSparseParams(block_sparse_params)
    , mType(type)
    , mMultiBlockMode(
          is_spec_decoding_enabled ? false : true)
    , mEnableXQA(true)
    , mKVCacheQuantMode(kv_cache_quant_mode)
    , mRemovePadding(remove_input_padding)
    , mPagedKVCache(paged_kv_cache)
    , mTokensPerBlock(tokens_per_block)
    , mTpSize(tp_size)
    , mTpRank(tp_rank)
    , mUnfuseQkvGemm(unfuse_qkv_gemm)
    , mUseLognScaling(use_logn_scaling)
    , mMaxContextLength(max_context_length)
    , mQKVBiasEnabled(qkv_bias_enabled)
    , mCrossAttention(cross_attention)
    , mMaxDistance(max_distance)
    , mPosShiftEnabled(pos_shift_enabled)
    , mDenseContextFMHA(dense_context_fmha)
    , mPagedContextFMHA(use_paged_context_fmha)
    , mFP8ContextFMHA(use_fp8_context_fmha)
    , mHasFullAttentionMask(has_full_attention_mask)
    , mUseKVCache(use_cache)
    , mIsSpecDecodingEnabled(is_spec_decoding_enabled)
    , mSpecDecodingIsGenerationLengthVariable(spec_decoding_is_generation_length_variable)
    , mSpecDecodingMaxGenerationLength(spec_decoding_max_generation_length)
    , mIsMLAEnabled(is_mla_enabled)
    , mMLAParams({q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim})
    , mCpSize(cp_size)
    , mCpRank(cp_rank)
    , mCpGroup(move(cp_group))
    , mDriver(CUDADriverWrapper::getInstance())
    , mSkipAttn(skip_attn)
{
    if (mEnableContextFMHA)
    {
        mEnableContextFMHA = false;
        if (!(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16))
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of unsupported data type.");
        }
        else if (mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kRELATIVE)
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of relative position embedding.");
        }
        else if (mSM == 70 && isALiBi())
        {
            TLLM_LOG_WARNING("Alibi is not supported for FMHA on Volta.");
        }
        else if (isCrossAttention() && !mPagedKVCache)
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of cross attention + contiguous kv cache.");
        }
        else
        {
            mEnableContextFMHA = true;
        }
    }

    if (mFP8ContextFMHA)
    {
        TLLM_CHECK_WITH_INFO(mEnableContextFMHA, "FP8 FMHA cannot be enabled because Context FMHA is not supported.");
        TLLM_CHECK_WITH_INFO(mSM == 89 || mSM == 90, "FP8 FMHA cannot be enabled except on Ada or Hopper Arch.");
    }

    TLLM_CHECK(isRoPE() == (rotary_embedding_dim != 0));
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (mType != nvinfer1::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");

    if (mSM == 70 && mEnableContextFMHA)
    {
        TLLM_CHECK_WITH_INFO(!mFMHAForceFP32Acc, "FP32 Acc is not supported on Volta");
    }

    if (!mmha_supported(getHeadSize()) && !mIsMLAEnabled)
    {
        TLLM_CHECK_WITH_INFO(false, "Head size %d is not supported by MMHA.", getHeadSize());
    }

    if (mIsMLAEnabled)
    {
        TLLM_CHECK_WITH_INFO(mEnableContextFMHA, "MLA(Deepseek v2) only support fmha");
        TLLM_CHECK_WITH_INFO(
            !mFP8ContextFMHA && !mDenseContextFMHA, "MLA(Deepseek v2) currently not support FP8 and dense fmha");
        TLLM_CHECK_WITH_INFO(
            mPagedKVCache && mUseKVCache && mRemovePadding, "MLA(Deepseek v2) only support paged kv cache");
        TLLM_CHECK_WITH_INFO(!mCrossAttention, "MLA(Deepseek v2) do not support cross attention right now");
        TLLM_CHECK_WITH_INFO(mMaskType != suggestify::kernels::AttentionMaskType::CUSTOM_MASK,
            "MLA(Deepseek v2) do not support custom mask right now");
        TLLM_CHECK_WITH_INFO(mMLAParams.qk_rope_head_dim == 64 && mMLAParams.kv_lora_rank == 512,
            "MLA(Deepseek v2) only support fixed kv_lora_rank(512) and fixed qk_rope_head_dim(64) right now.");
    }
}

int GPTAttentionPluginCommon::getHeadSize(bool checkInit) const
{
    if (checkInit)
    {
        TLLM_CHECK_WITH_INFO(mHeadSize > 0, "Trying to read mHeadSize before it's been initialized");
    }
    return mHeadSize;
}

GPTAttentionPluginCommon::GPTAttentionPluginCommon(void const* data, size_t length)
    : mDriver(CUDADriverWrapper::getInstance())
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    unsigned int kvCacheQuantMode;

    read(d, mLayerIdx);
    read(d, mNumHeads);
    read(d, mVisionStart);
    read(d, mVisionLength);
    read(d, mNumKVHeads);
    read(d, mLayerIdxInCachePool);
    read(d, mHeadSize);
    read(d, mUnidirectional);
    read(d, mQScaling);
    read(d, mAttnLogitSoftcappingScale);
    read(d, mPositionEmbeddingType);
    read(d, mRotaryEmbeddingDim);
    read(d, mRotaryEmbeddingBase);
    read(d, mRotaryEmbeddingScaleType);
    read(d, mRotaryEmbeddingScale);
    read(d, mRotaryEmbeddingShortMscale);
    read(d, mRotaryEmbeddingLongMscale);
    read(d, mRotaryEmbeddingMaxPositions);
    read(d, mRotaryEmbeddingOriginalMaxPositions);
    read(d, mTpSize);
    read(d, mTpRank);
    read(d, mUnfuseQkvGemm);
    read(d, mUseLognScaling);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mMultiBlockMode);
    read(d, mEnableXQA);
    read(d, kvCacheQuantMode);
    read(d, mRemovePadding);
    read(d, mMaskType);
    read(d, mBlockSparseParams);
    read(d, mPagedKVCache);
    read(d, mTokensPerBlock);
    read(d, mType);
    read(d, mMaxContextLength);
    read(d, mQKVBiasEnabled);
    read(d, mCrossAttention);
    read(d, mMaxDistance);
    read(d, mPosShiftEnabled);
    read(d, mDenseContextFMHA);
    read(d, mPagedContextFMHA);
    read(d, mFP8ContextFMHA);
    read(d, mHasFullAttentionMask);
    read(d, mUseKVCache);
    read(d, mIsSpecDecodingEnabled);
    read(d, mUseSpecDecoding);
    read(d, mSpecDecodingIsGenerationLengthVariable);
    read(d, mSpecDecodingMaxGenerationLength);
    read(d, mIsMLAEnabled);
    read(d, mMLAParams);
    read(d, mNbMultiBlockSemaphores);
    read(d, mSkipAttn);
    read(d, mCpSize);
    read(d, mCpRank);

    mKVCacheQuantMode = tc::QuantMode(kvCacheQuantMode);

    uint32_t decoderXQARunnerResourceSerializedSize;
    read(d, decoderXQARunnerResourceSerializedSize);
    DecoderXQARunner::getResourceGlobal()->merge(
        DecoderXQARunner::Resource(d, decoderXQARunnerResourceSerializedSize),true);
    d += decoderXQARunnerResourceSerializedSize;

    mCpGroup.clear();
    int32_t groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mCpGroup.insert(groupItem);
    }
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (mType != nvinfer1::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

size_t GPTAttentionPluginCommon::getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t max_num_seq,
    int32_t input_seq_length, int32_t cross_kv_length, int32_t max_num_tokens) const noexcept
{
    int const local_hidden_units_qo = mNumHeads / mCpSize * getHeadSize();
    int const local_hidden_units_kv = mNumKVHeads / mCpSize * getHeadSize();

    auto const size = suggestify::runtime::BufferDataType(type).getSize();

    size_t context_workspace_size = 0;

    auto const batch_size = static_cast<size_t>(max_num_seq);
    auto const kv_seq_length = (isCrossAttention() ? cross_kv_length : input_seq_length);
    size_t const attention_mask_size = mEnableContextFMHA ? 0 : size * max_num_tokens * kv_seq_length;
    size_t const cu_seqlens_size = sizeof(int) * (batch_size + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * batch_size * mRotaryEmbeddingDim / 2;
    size_t const q_buf_2_size = !mEnableContextFMHA || mFMHARunner->isSeparateQAndKvInput()
        ? (mFP8ContextFMHA ? 1 : size) * max_num_tokens * local_hidden_units_qo
        : 0;
    size_t const k_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const v_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const qk_buf_size
        = mEnableContextFMHA ? 0 : size * batch_size * mNumHeads * input_seq_length * kv_seq_length;
    size_t const qkv_buf_2_size = mEnableContextFMHA ? 0 : size * max_num_tokens * local_hidden_units_qo;
    size_t const qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_length * kv_seq_length;
    size_t const fp8_qkv_buffer_size = mFP8ContextFMHA && mEnableContextFMHA && !mFMHARunner->isSeparateQAndKvInput()
        ? max_num_tokens * size_t(local_hidden_units_qo + 2 * local_hidden_units_kv)
        : 0;
    size_t const padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * max_num_tokens;
    size_t const encoder_padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * max_num_tokens;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;
    size_t const fmha_bmm1_scale_size = mFP8ContextFMHA ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = mFP8ContextFMHA ? sizeof(float) : 0;

    size_t const cpMaxPaddedSequenceLength = max_num_tokens + batch_size * (mCpSize - 1);
    size_t const cpWorkspaceSize = mCpSize == 1
        ? 0
        : (2 * size * cpMaxPaddedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads) + cu_seqlens_size);

    int const NUM_BUFFERS = 19;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = CUBLAS_WORKSPACE_SIZE;
    workspaces[1] = attention_mask_size;
    workspaces[2] = cu_seqlens_size;
    workspaces[3] = cu_seqlens_size;
    workspaces[4] = cu_seqlens_size;
    workspaces[5] = rotary_inv_freq_size;
    workspaces[6] = q_buf_2_size;
    workspaces[7] = k_buf_2_size;
    workspaces[8] = v_buf_2_size;
    workspaces[9] = qk_buf_size;
    workspaces[10] = qkv_buf_2_size;
    workspaces[11] = qk_buf_float_size;
    workspaces[12] = fp8_qkv_buffer_size;
    workspaces[13] = padding_offset_size;
    workspaces[14] = encoder_padding_offset_size;
    workspaces[15] = fmha_scheduler_counter;
    workspaces[16] = fmha_bmm1_scale_size;
    workspaces[17] = fmha_bmm2_scale_size;
    workspaces[18] = cpWorkspaceSize;
    context_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);

    return context_workspace_size;
}

size_t GPTAttentionPluginCommon::getWorkspaceSizeForGeneration(
    nvinfer1::DataType type, int32_t max_num_seq, int32_t max_attention_window, int32_t max_num_tokens) const noexcept
{
    int const local_hidden_units_qo = mNumHeads * getHeadSize();
    int const local_hidden_units_kv = mNumKVHeads * getHeadSize();

    auto const size = suggestify::runtime::BufferDataType(type).getSize();

    size_t context_workspace_size = 0;
    size_t generation_workspace_size = 0;

    if (mIsMLAEnabled)
    {
        size_t cu_seqlens_size = sizeof(int) * (max_num_tokens + 1);
        size_t fmha_scheduler_counter = sizeof(uint32_t);
        size_t o_buffer_size = size * max_num_tokens * mNumHeads * mMLAParams.kv_lora_rank;
        int const NUM_BUFFERS = 5;
        size_t workspaces[NUM_BUFFERS];
        workspaces[0] = CUBLAS_WORKSPACE_SIZE;
        workspaces[1] = cu_seqlens_size;
        workspaces[2] = cu_seqlens_size;
        workspaces[3] = fmha_scheduler_counter;
        workspaces[4] = o_buffer_size;
        generation_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
        return generation_workspace_size;
    }

    int const batch_beam = max_num_seq;
    int32_t const maxSeqLenTile
        = std::max(getMaxNumSeqLenTile(batch_beam), (int) tc::divUp(mMultiProcessorCount, mNumHeads));

    size_t const partial_out_size = size * batch_beam * mNumHeads * mHeadSize * maxSeqLenTile;
    size_t const partial_sum_size = sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile;
    size_t const partial_max_size = sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile;
    size_t const shift_k_cache_size = (!mPosShiftEnabled || isCrossAttention())
        ? 0
        : size * batch_beam * mNumHeads * mHeadSize * max_attention_window;
    size_t const cpMaxPaddedSequenceLength = (batch_beam + mCpSize - 1) / mCpSize * mCpSize;
    size_t const cpWorkspaceSize
        = mCpSize == 1 ? 0 : (2 * size * cpMaxPaddedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads));

    int const NUM_BUFFERS = 5;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = partial_out_size;
    workspaces[1] = partial_sum_size;
    workspaces[2] = partial_max_size;
    workspaces[3] = shift_k_cache_size;
    workspaces[4] = cpWorkspaceSize;
    generation_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);

    size_t mqa_workspace_size = 0;
    if (mDecoderXQARunner.get())
    {
        int const XQA_NUM_BUFFERS = 3;
        size_t mqa_workspaces[XQA_NUM_BUFFERS];
        size_t const cu_seqlens_size = sizeof(int) * (batch_beam + 1);
        size_t const rotary_inv_freq_size = sizeof(float) * batch_beam * mRotaryEmbeddingDim / 2;
        mqa_workspaces[0] = mDecoderXQARunner->getWorkspaceSize(max_num_tokens);
        mqa_workspaces[1] = cu_seqlens_size;
        mqa_workspaces[2] = rotary_inv_freq_size;
        mqa_workspace_size = tc::calculateTotalWorkspaceSize(mqa_workspaces, XQA_NUM_BUFFERS);
    }

    return std::max(generation_workspace_size, mqa_workspace_size);
}

int GPTAttentionPluginCommon::getMaxNumSeqLenTile(int batch_beam_size) const
{
    if (mMultiBlockMode)
    {
        return tc::getEnvMmhaMultiblockDebug() ? std::max(kReservedMaxSeqLenTilePerSeq, getEnvMmhaBlocksPerSequence())
                                               : tc::divUp(mMultiProcessorCount, batch_beam_size * mNumHeads);
    }
    return 0;
}

template <typename T, typename KVCacheBuffer>
int GPTAttentionPluginCommon::mlaPreContext(
    mlaParams<T>& params, EnqueueContextParams<T> const& context_params, cudaStream_t stream)
{
    auto cublasHandle = mCublasWrapper->getCublasHandle();
    TLLM_CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(params.workspace);
    if constexpr (std::is_same_v<T, half>)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    auto rope_dim = mMLAParams.qk_rope_head_dim;
    auto c_q = mMLAParams.q_lora_rank;
    auto c_k = mMLAParams.kv_lora_rank;
    auto context_head_size = mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim;
    auto v_head_dim = mMLAParams.v_head_dim;

    {
        auto transa = CUBLAS_OP_T;
        auto transb = CUBLAS_OP_N;
        int m = params.head_num * context_head_size;
        int n = params.acc_q_len;
        int k = c_q;
        int lda = k, ldb = c_q + c_k + rope_dim;
        int ldc
            = params.head_num * (2 * context_head_size + v_head_dim);
        mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
        mCublasWrapper->Gemm(
            transa, transb, m, n, k, params.q_b_proj, lda, params.fused_a_input, ldb, params.attention_input_buf, ldc);
        mCublasWrapper->destroyDescriptors();
    }

    {
        auto transa = CUBLAS_OP_T;
        auto transb = CUBLAS_OP_N;
        int m = mMLAParams.qk_nope_head_dim;
        int n = params.acc_q_len;
        int k = c_k;
        int lda = k, ldb = c_q + c_k + rope_dim;
        int ldc = params.head_num * (context_head_size * 2 + v_head_dim);
        mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

        mCublasWrapper->stridedBatchedGemm(transa, transb, m, n, k, params.kv_b_proj, lda,
            mMLAParams.qk_nope_head_dim * c_k, params.fused_a_input + c_q, ldb, 0,
            params.attention_input_buf + static_cast<size_t>(params.head_num) * context_head_size,
            params.head_num * (context_head_size * 2 + v_head_dim), context_head_size, params.head_num, 1.0f, 0.0);
        mCublasWrapper->stridedBatchedGemm(transa, transb, m, n, k,
            params.kv_b_proj + mMLAParams.qk_nope_head_dim * c_k * params.head_num, lda, v_head_dim * c_k,
            params.fused_a_input + c_q, ldb, 0,
            params.attention_input_buf + 2 * static_cast<size_t>(params.head_num) * context_head_size,
            params.head_num * (context_head_size * 2 + v_head_dim), v_head_dim, params.head_num, 1.0f, 0.0);
        mCublasWrapper->destroyDescriptors();
    }

    return 0;
}

template <typename T, typename KVCacheBuffer>
int GPTAttentionPluginCommon::mlaGeneration(
    mlaParams<T>& params, EnqueueGenerationParams<T> const& generation_params, cudaStream_t stream)
{
    int const num_heads = mNumHeads;
    int const num_kv_heads = 1;
    int const head_size = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
    int32_t const batch_beam = generation_params.beam_width * generation_params.num_requests;

    auto const sizePerToken = num_kv_heads * head_size * sizeof(T);
    using BufferDataType = typename KVLinearBuffer::DataType;
    KVCacheBuffer kv_cache_buffer;
    if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
    {
        kv_cache_buffer = KVBlockArray(batch_beam, generation_params.max_blocks_per_sequence, mTokensPerBlock,
            sizePerToken, generation_params.cyclic_attention_window_size,
            generation_params.max_cyclic_attention_window_size, generation_params.sink_token_length,
            generation_params.can_use_one_more_block, generation_params.host_primary_pool_pointer,
            generation_params.host_secondary_pool_pointer, generation_params.block_offsets);
    }
    else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
    {
        using BufferDataType = typename KVCacheBuffer::DataType;
        kv_cache_buffer = KVLinearBuffer(batch_beam, generation_params.max_attention_window, sizePerToken,
            generation_params.cyclic_attention_window_size, generation_params.sink_token_length, true,
            reinterpret_cast<BufferDataType*>(generation_params.key_value_cache));
    }

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    TLLM_CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(params.workspace);
    if constexpr (std::is_same_v<T, half>)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    cudaDataType_t const gemm_data_type = tc::CudaDataType<T>::value;
    cudaDataType_t const gemm_out_data_type = CUDA_R_32F;
    float const qk_scale_gemm = 1.0f;
    auto const size = sizeof(T);
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    {
        auto transa = CUBLAS_OP_T;
        auto transb = CUBLAS_OP_N;
        int m = params.head_num * (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim);
        int n = params.acc_q_len;
        int k = mMLAParams.q_lora_rank;
        int lda = k, ldb = mMLAParams.q_lora_rank + mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
        int ldc = m;
        mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
        mCublasWrapper->Gemm(transa, transb, m, n, k, params.fused_q_proj, lda, params.fused_a_input, ldb,
            params.attention_input_buf, ldc);
        mCublasWrapper->destroyDescriptors();
    }

    size_t const cu_seqlens_size = sizeof(int) * (params.batch_size + 1);
    size_t const fmha_scheduler_counter = sizeof(uint32_t);
    size_t o_buffer_size = size * batch_beam * mNumHeads * mMLAParams.kv_lora_rank;
    int* cu_q_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    int* cu_kv_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    uint32_t* fmha_tile_counter_ptr
        = reinterpret_cast<uint32_t*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_scheduler_counter));
    T* o_buffer = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, o_buffer_size));

    params.seqQOffset = cu_q_seqlens;
    params.cu_kv_seqlens = cu_kv_seqlens;
    params.fmha_tile_counter = fmha_tile_counter_ptr;

    invokeMLARopeGeneration<T>(params, kv_cache_buffer, stream);

    int64_t enable_context_fmha_fp32_acc_val = generation_params.runtime_perf_knobs[1];
    mFMHAForceFP32Acc = mFMHAForceFP32Acc || enable_context_fmha_fp32_acc_val == 1;

    MHARunnerParams fmhaParams;
    memset(&fmhaParams, 0, sizeof(fmhaParams));
    fmhaParams.b = batch_beam;
    fmhaParams.qSeqLen = params.head_num;
    fmhaParams.kvSeqLen = generation_params.max_past_kv_length;
    fmhaParams.slidingWindowSize = generation_params.cyclic_attention_window_size;
    fmhaParams.totalQSeqLen = batch_beam * params.head_num;
    fmhaParams.qPtr = reinterpret_cast<void const*>(params.attention_input_buf);
    fmhaParams.kvPtr = nullptr;
    fmhaParams.outputPtr = o_buffer;
    fmhaParams.pagedKvCache = reinterpret_cast<KVBlockArray&>(kv_cache_buffer);
    fmhaParams.cuQSeqLenPtr = cu_q_seqlens;
    fmhaParams.cuKvSeqLenPtr = cu_kv_seqlens;
    fmhaParams.cuMaskRowsPtr = nullptr;
    fmhaParams.tileCounterPtr = fmha_tile_counter_ptr;
    fmhaParams.scaleBmm1Ptr = nullptr;
    fmhaParams.scaleBmm2Ptr = nullptr;
    fmhaParams.stream = stream;
    fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;

    mDecoderFMHARunner->run(fmhaParams);

    {
        auto transa = CUBLAS_OP_T;
        auto transb = CUBLAS_OP_N;
        int m = mMLAParams.v_head_dim;
        int n = params.batch_size;
        int k = mMLAParams.kv_lora_rank;
        int lda = k, ldb = k;
        int ldc = m;
        mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

        mCublasWrapper->stridedBatchedGemm(transa, transb, m, n, k,
            params.kv_b_proj + params.head_num * mMLAParams.kv_lora_rank * mMLAParams.qk_nope_head_dim, lda,
            mMLAParams.v_head_dim * mMLAParams.kv_lora_rank, o_buffer, ldb * params.head_num, ldb, params.context_buf,
            ldc * params.head_num, ldc, params.head_num, 1.0f, 0.0);
        mCublasWrapper->destroyDescriptors();
    }

    sync_check_cuda_error();
    return 0;
}

#define MLA_FUNC_DEFINE(T, KVCacheBuffer)                                                                              \
    template int GPTAttentionPluginCommon::mlaPreContext<T, KVCacheBuffer>(                                            \
        mlaParams<T> & params, EnqueueContextParams<T> const& context_params, cudaStream_t stream);                    \
    template int GPTAttentionPluginCommon::mlaGeneration<T, KVCacheBuffer>(                                            \
        mlaParams<T> & params, EnqueueGenerationParams<T> const& generation_params, cudaStream_t stream);

MLA_FUNC_DEFINE(float, KVBlockArray)
MLA_FUNC_DEFINE(half, KVBlockArray)
MLA_FUNC_DEFINE(float, KVLinearBuffer)
MLA_FUNC_DEFINE(half, KVLinearBuffer)
#ifdef ENABLE_BF16
MLA_FUNC_DEFINE(__nv_bfloat16, KVBlockArray)
MLA_FUNC_DEFINE(__nv_bfloat16, KVLinearBuffer)
#endif

template <typename T, typename KVCacheBuffer>
int GPTAttentionPluginCommon::enqueueContext(EnqueueContextParams<T> const& params, cudaStream_t stream)
{
    int const num_heads = mNumHeads;
    int const num_kv_heads = mNumKVHeads;
    int const head_size = getHeadSize();

    int const local_hidden_units_qo = num_heads * head_size;
    int const local_hidden_units_kv = num_kv_heads * head_size;
    PositionEmbeddingType const position_embedding_type = mPositionEmbeddingType;
    float const q_scaling = mQScaling;
    bool const* finished = nullptr;
    bool const has_ia3 = false;

    KVCacheBuffer kv_cache_buffer;
    auto const elemSize = mKVCacheQuantMode.hasKvCacheQuant() ? sizeof(int8_t) : sizeof(T);
    auto sizePerToken = num_kv_heads * head_size * elemSize;
    KVBlockArray::DataType* hostKvCacheBlockOffsets = nullptr;
    if (useKVCache())
    {
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            kv_cache_buffer = KVBlockArray(params.batch_size, params.max_blocks_per_sequence, mTokensPerBlock,
                sizePerToken, params.cyclic_attention_window_size, params.max_cyclic_attention_window_size,
                params.sink_token_length, params.can_use_one_more_block, params.host_primary_pool_pointer,
                params.host_secondary_pool_pointer, params.block_offsets);
            hostKvCacheBlockOffsets = params.host_block_offsets;
        }
        else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
        {
            using BufferDataType = typename KVCacheBuffer::DataType;
            kv_cache_buffer = KVLinearBuffer(params.batch_size,
                isCrossAttention() ? params.cross_kv_length : params.max_attention_window, sizePerToken,
                params.cyclic_attention_window_size, params.sink_token_length, false,
                reinterpret_cast<BufferDataType*>(params.key_value_cache));
        }
    }

    auto const quant_option = tc::QuantMode::fromDescription();
    float const* qkv_scale_out = nullptr;
    float const* attention_out_scale = nullptr;

    int const* ia3_tasks = nullptr;
    T const* ia3_key_weights = nullptr;
    T const* ia3_value_weights = nullptr;

    int const max_seq_len_tile = 0;
    T* partial_out = nullptr;
    float* partial_sum = nullptr;
    float* partial_max = nullptr;
    int* block_counter = nullptr;

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    TLLM_CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(params.workspace);
    if constexpr (std::is_same_v<T, half>)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    size_t const kv_seq_length = (isCrossAttention() ? params.cross_kv_length : params.input_seq_length);
    size_t const attention_mask_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * params.input_seq_length * kv_seq_length;
    size_t const cu_seqlens_size = sizeof(int) * (params.batch_size + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * params.batch_size * mRotaryEmbeddingDim / 2;
    size_t const q_buf_2_size = !mEnableContextFMHA || mFMHARunner->isSeparateQAndKvInput()
        ? (mFP8ContextFMHA ? 1 : sizeof(T)) * params.batch_size * params.input_seq_length * local_hidden_units_qo
        : 0;
    size_t const k_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const v_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const qk_buf_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * mNumHeads * params.input_seq_length * kv_seq_length;
    size_t const qkv_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * params.input_seq_length * local_hidden_units_qo;
    size_t const qk_buf_float_size = mEnableContextFMHA
        ? 0
        : sizeof(float) * params.batch_size * mNumHeads * params.input_seq_length * kv_seq_length;
    size_t const fp8_qkv_buffer_size = mEnableContextFMHA && mFP8ContextFMHA && !mFMHARunner->isSeparateQAndKvInput()
        ? params.batch_size * params.input_seq_length * (local_hidden_units_qo + 2 * local_hidden_units_kv)
        : 0;
    size_t const padding_offset_size
        = mEnableContextFMHA ? 0 : sizeof(int) * params.batch_size * params.input_seq_length;
    size_t const encoder_padding_offset_size
        = mEnableContextFMHA ? 0 : sizeof(int) * params.batch_size * params.cross_kv_length;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;
    size_t const fmha_bmm1_scale_size = mFP8ContextFMHA ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = mFP8ContextFMHA ? sizeof(float) : 0;

    size_t const cpMaxPadedSequenceLength = params.num_tokens + params.batch_size * (mCpSize - 1);
    size_t const cpWorkspaceSize
        = mCpSize == 1 ? 0 : 2 * sizeof(T) * cpMaxPadedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads);

    bool const is_qk_buf_float_ = true;

    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    T* attention_mask = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, attention_mask_size));
    int* cu_q_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    int* cu_kv_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    int* cu_mask_rows = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    float* rotary_inv_freq_buf
        = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, rotary_inv_freq_size));
    T* q_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, q_buf_2_size));
    T* k_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, k_buf_2_size));
    T* v_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, v_buf_2_size));
    T* qk_buf_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_size));
    T* qkv_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qkv_buf_2_size));
    float* qk_buf_float_ = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_float_size));
    __nv_fp8_e4m3* fp8_qkv_buffer
        = reinterpret_cast<__nv_fp8_e4m3*>(nextWorkspacePtr(workspace_byte_ptr, offset, fp8_qkv_buffer_size));
    int* padding_offset = mEnableContextFMHA
        ? nullptr
        : reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, padding_offset_size));
    int* encoder_padding_offset = (mEnableContextFMHA && !isCrossAttention())
        ? nullptr
        : reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, encoder_padding_offset_size));
    uint32_t* fmha_tile_counter_ptr
        = reinterpret_cast<uint32_t*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_scheduler_counter));
    float* fmha_bmm1_scale_ptr
        = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_bmm1_scale_size));
    float* fmha_bmm2_scale_ptr
        = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_bmm2_scale_size));

    T* gatherInBuffer = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, cpWorkspaceSize));
    T* gatherOutBuffer = gatherInBuffer + cpMaxPadedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads);
    int* cu_cp_partial_seqlens = reinterpret_cast<int*>(
        gatherOutBuffer + cpMaxPadedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads));

    BuildDecoderInfoParams<T> decoder_params;
    memset(&decoder_params, 0, sizeof(decoder_params));
    decoder_params.seqQOffsets = cu_q_seqlens;
    decoder_params.seqKVOffsets = cu_kv_seqlens;
    decoder_params.seqCpPartialOffsets = cu_cp_partial_seqlens;
    decoder_params.cpSize = mCpSize;
    decoder_params.packedMaskRowOffsets = cu_mask_rows;
    decoder_params.paddingOffsets = padding_offset;
    decoder_params.encoderPaddingOffsets
        = isCrossAttention() ? encoder_padding_offset : nullptr;
    decoder_params.attentionMask = isCrossAttention() ? nullptr : attention_mask;
    decoder_params.seqQLengths = params.q_seq_lengths;
    decoder_params.seqKVLengths = isCrossAttention() ? params.encoder_input_lengths : params.kv_seq_lengths;
    decoder_params.batchSize = params.batch_size;
    decoder_params.maxQSeqLength = params.input_seq_length;
    decoder_params.maxEncoderQSeqLength
        = isCrossAttention() ? params.cross_kv_length : 0;
    decoder_params.attentionWindowSize = params.cyclic_attention_window_size;
    decoder_params.sinkTokenLength = params.sink_token_length;
    decoder_params.numTokens = params.num_tokens;
    decoder_params.attentionMaskType = mMaskType;
    decoder_params.blockSparseParams = mBlockSparseParams;
    decoder_params.fmhaTileCounter = fmha_tile_counter_ptr;
    decoder_params.quantScaleO = params.attention_output_orig_quant;
    decoder_params.dequantScaleQkv = params.kv_scale_quant_orig;
    decoder_params.fmhaHostBmm1Scale = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling);
    decoder_params.fmhaBmm1Scale = fmha_bmm1_scale_ptr;
    decoder_params.fmhaBmm2Scale = fmha_bmm2_scale_ptr;
    decoder_params.rotaryEmbeddingScale = mRotaryEmbeddingScale;
    decoder_params.rotaryEmbeddingBase = mRotaryEmbeddingBase;
    decoder_params.rotaryEmbeddingDim = mRotaryEmbeddingDim;
    decoder_params.rotaryScalingType = mRotaryEmbeddingScaleType;
    decoder_params.rotaryEmbeddingInvFreq = rotary_inv_freq_buf;
    decoder_params.rotaryEmbeddingInvFreqCache = params.rotary_inv_freq;
    decoder_params.rotaryEmbeddingMaxPositions = mRotaryEmbeddingMaxPositions;

    invokeBuildDecoderInfo(decoder_params, stream);
    sync_check_cuda_error();

    if (isUnfusedCrossAttention())
    {
        {
            std::vector<T> h_attention_mask(params.batch_size * params.input_seq_length * params.cross_kv_length, 1.);
            std::vector<int32_t> h_encoder_input_lengths(params.batch_size);
            cudaMemcpyAsync(h_encoder_input_lengths.data(), params.encoder_input_lengths,
                sizeof(int32_t) * params.batch_size, cudaMemcpyDeviceToHost, stream);
            sync_check_cuda_error();

            for (int bi = 0; bi < params.batch_size; bi++)
            {
                int b_offset = bi * params.input_seq_length * params.cross_kv_length;
                for (int qi = 0; qi < params.input_seq_length; qi++)
                {
                    int q_offset = b_offset + qi * params.cross_kv_length;
                    if (h_encoder_input_lengths[bi] < params.cross_kv_length)
                    {
                        std::fill(h_attention_mask.begin() + q_offset + h_encoder_input_lengths[bi],
                            h_attention_mask.begin() + q_offset + params.cross_kv_length, 0.f);
                    }
                }
            }
            cudaMemcpyAsync(attention_mask, h_attention_mask.data(),
                sizeof(T) * params.batch_size * params.cross_kv_length * params.input_seq_length,
                cudaMemcpyHostToDevice, stream);
            sync_check_cuda_error();
        }
    }

    if (!mRemovePadding)
    {
        cudaMemsetAsync(params.context_buf, 0, params.num_tokens * local_hidden_units_qo * sizeof(T), stream);
        sync_check_cuda_error();
    }

    KvCacheDataType const cache_type = mKVCacheQuantMode.hasInt8KvCache()
        ? KvCacheDataType::INT8
        : (mKVCacheQuantMode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

    cudaDataType_t const gemm_data_type = tc::CudaDataType<T>::value;
    int const attention_seq_len_1 = params.input_seq_length;
    int const attention_seq_len_2 = isCrossAttention() ? params.cross_kv_length : params.input_seq_length;

    float const qk_scale
        = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling);
    float const qk_scale_gemm = isRelativePosition() ? qk_scale : 1.0f;
    T const qk_scale_softmax = static_cast<T>(isRelativePosition() ? 1.0f : qk_scale);

    if (mEnableContextFMHA)
    {
        T* attention_input = const_cast<T*>(params.attention_input);
        if (mCpSize > 1)
        {
            int32_t partialTokenNum = 0;
            int32_t maxPartialLength = 0;
            for (size_t batchIdx = 0; batchIdx < params.batch_size; ++batchIdx)
            {
                int32_t partialLength = (params.host_context_lengths[batchIdx] + mCpSize - 1) / mCpSize;
                maxPartialLength = std::max(maxPartialLength, partialLength);
                partialTokenNum += partialLength;
            }
            auto const totalHeads = mNumHeads + 2 * mNumKVHeads;
            auto const partialHeads = totalHeads / mCpSize;
            auto const partialQHeads = mNumHeads / mCpSize;
            auto const partialKVHeads = mNumKVHeads / mCpSize;


            invokeCpTranspose(gatherInBuffer, gatherOutBuffer, params.attention_input, partialTokenNum, mCpSize,
                partialQHeads, partialKVHeads, getHeadSize(), mCpRank, stream);
            sync_check_cuda_error();

#if ENABLE_MULTI_DEVICE
            ncclGroupStart();
            for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
            {
                if (cpIdx != mCpRank)
                {
                    NCCLCHECK(ncclSend(gatherInBuffer + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                        (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                        stream));
                    NCCLCHECK(ncclRecv(gatherOutBuffer + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                        (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                        stream));
                }
            }
            ncclGroupEnd();
            sync_check_cuda_error();
#endif

            invokeCpTranspose2(gatherInBuffer, gatherOutBuffer, params.q_seq_lengths, cu_q_seqlens,
                cu_cp_partial_seqlens, mCpSize, maxPartialLength, params.batch_size, partialHeads, getHeadSize(),
                stream);

            attention_input = gatherInBuffer;
        }
        sync_check_cuda_error();

        bool const enablePagedKVContextFMHA = mPagedKVCache && mPagedContextFMHA;
        TLLM_CHECK_WITH_INFO(!(mKVCacheQuantMode.hasInt8KvCache() && enablePagedKVContextFMHA),
            "Paged Context FMHA doesn't work with int8 kv cache currently.");
        TLLM_CHECK_WITH_INFO(
            !(mKVCacheQuantMode.hasFp8KvCache() && !mKVCacheQuantMode.hasFp8Qdq() && enablePagedKVContextFMHA),
            "FP8 Paged Context FMHA only works with fp8 quantization workflow currently.");
        TLLM_CHECK_WITH_INFO(!(params.sink_token_length > 0 && enablePagedKVContextFMHA),
            "Cannot support StreamingLLM now when enabling paged KV context FMHA.");

        int const max_kv_seq_len = isCrossAttention() ? params.cross_kv_length : params.max_past_kv_len;

        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParams;

        preprocessingParams.qkv_input = const_cast<T*>(attention_input);
        preprocessingParams.cross_kv_input = const_cast<T*>(params.cross_kv);
        preprocessingParams.quantized_qkv_output = fp8_qkv_buffer;
        preprocessingParams.q_output = q_buf_2_;
        preprocessingParams.kv_cache_buffer = kv_cache_buffer;
        preprocessingParams.qkv_bias = params.qkv_bias;
        preprocessingParams.seq_lens = params.q_seq_lengths;
        preprocessingParams.cache_seq_lens = params.kv_seq_lengths;
        preprocessingParams.encoder_seq_lens = params.encoder_input_lengths;
        preprocessingParams.cu_seq_lens = cu_q_seqlens;
        preprocessingParams.cu_kv_seq_lens = cu_kv_seqlens;
        preprocessingParams.rotary_embedding_inv_freq = rotary_inv_freq_buf;
        preprocessingParams.rotary_coef_cache_buffer = params.rotary_cos_sin;
        preprocessingParams.mrope_rotary_cos_sin = params.mrope_rotary_cos_sin;
        preprocessingParams.kvScaleOrigQuant = params.kv_scale_orig_quant;
        preprocessingParams.spec_decoding_position_offsets = nullptr;
        preprocessingParams.logn_scaling = params.logn_scaling_ptr;

        preprocessingParams.batch_size = params.batch_size;
        preprocessingParams.max_input_seq_len = params.input_seq_length;
        preprocessingParams.max_kv_seq_len = max_kv_seq_len;
        preprocessingParams.cyclic_kv_cache_len
            = isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size;
        preprocessingParams.sink_token_len = params.sink_token_length;
        preprocessingParams.token_num = params.num_tokens;
        preprocessingParams.remove_padding = mRemovePadding;
        preprocessingParams.cross_attention = isCrossAttention();
        preprocessingParams.head_num = mNumHeads / mCpSize;
        preprocessingParams.kv_head_num = mNumKVHeads / mCpSize;
        preprocessingParams.qheads_per_kv_head = mNumHeads / mNumKVHeads;
        preprocessingParams.size_per_head = getHeadSize();
        preprocessingParams.rotary_embedding_dim = mRotaryEmbeddingDim;
        preprocessingParams.rotary_embedding_base = mRotaryEmbeddingBase;
        preprocessingParams.rotary_scale_type = mRotaryEmbeddingScaleType;
        preprocessingParams.rotary_embedding_scale = mRotaryEmbeddingScale;
        preprocessingParams.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
        preprocessingParams.position_embedding_type = position_embedding_type;
        preprocessingParams.position_shift_enabled = mPosShiftEnabled;
        preprocessingParams.cache_type = cache_type;
        preprocessingParams.separate_q_kv_output = enablePagedKVContextFMHA || isCrossAttention();
        preprocessingParams.quantized_fp8_output = mFP8ContextFMHA;
        preprocessingParams.multi_processor_count = mMultiProcessorCount;

        preprocessingParams.rotary_vision_start = mVisionStart;
        preprocessingParams.rotary_vision_length = mVisionLength;

        {
            std::string const beforeRopeStr = "ctx attention before RoPE at layer " + std::to_string(mLayerIdx);
            TLLM_CHECK_DEBUG_WITH_INFO(suggestify::runtime::utils::tensorHasInvalid(params.num_tokens,
                                           (local_hidden_units_qo + 2 * local_hidden_units_kv), mType,
                                           const_cast<T*>(attention_input), stream, beforeRopeStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + beforeRopeStr);
        }
        if (mIsMLAEnabled)
        {
            params.mla_param->cu_q_seqlens = cu_q_seqlens;
            invokeMLARopeContext<T, KVCacheBuffer>(*params.mla_param, kv_cache_buffer, stream);
        }
        else
        {
            invokeQKVPreprocessing(preprocessingParams, stream);
        }
        {
            std::string const afterRopeStr = "ctx attention after RoPE at layer " + std::to_string(mLayerIdx);
            TLLM_CHECK_DEBUG_WITH_INFO(suggestify::runtime::utils::tensorHasInvalid(params.num_tokens,
                                           (local_hidden_units_qo + 2 * local_hidden_units_kv), mType,
                                           const_cast<T*>(attention_input), stream, afterRopeStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + afterRopeStr);
        }
        sync_check_cuda_error();
        int64_t enable_context_fmha_fp32_acc_val = params.runtime_perf_knobs[1];
        mFMHAForceFP32Acc = mFMHAForceFP32Acc || enable_context_fmha_fp32_acc_val == 1;


        MHARunnerParams fmhaParams;
        memset(&fmhaParams, 0, sizeof(fmhaParams));
        fmhaParams.b = params.batch_size;
        fmhaParams.qSeqLen = params.input_seq_length;
        fmhaParams.kvSeqLen = max_kv_seq_len;
        fmhaParams.slidingWindowSize
            = (mDenseContextFMHA || isCrossAttention()) ? max_kv_seq_len : params.cyclic_attention_window_size;
        fmhaParams.totalQSeqLen = params.num_tokens;
        fmhaParams.totalKvSeqLen = params.num_tokens;
        fmhaParams.qkvPtr = mFP8ContextFMHA ? reinterpret_cast<void const*>(fp8_qkv_buffer)
                                            : reinterpret_cast<void const*>(attention_input);
        fmhaParams.qPtr = reinterpret_cast<void const*>(q_buf_2_);
        fmhaParams.kvPtr = nullptr;
        fmhaParams.outputPtr
            = mCpSize > 1 ? gatherOutBuffer : params.context_buf;
        fmhaParams.packedMaskPtr = params.attention_packed_mask;
        fmhaParams.pagedKvCache = reinterpret_cast<KVBlockArray&>(kv_cache_buffer);
        fmhaParams.cuQSeqLenPtr = cu_q_seqlens;
        fmhaParams.cuKvSeqLenPtr = cu_kv_seqlens;
        fmhaParams.cuMaskRowsPtr = cu_mask_rows;
        fmhaParams.tileCounterPtr = fmha_tile_counter_ptr;
        fmhaParams.scaleBmm1Ptr = fmha_bmm1_scale_ptr;
        fmhaParams.scaleBmm2Ptr = fmha_bmm2_scale_ptr;
        fmhaParams.stream = stream;
        fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;

        mFMHARunner->run(fmhaParams);
        sync_check_cuda_error();

        invokeKvCachePostprocessing(preprocessingParams, stream);
        sync_check_cuda_error();

        if (mCpSize > 1)
        {

            int32_t maxPartialLength = 0;
            int32_t partialTokenNum = 0;
            for (size_t batchIdx = 0; batchIdx < params.batch_size; ++batchIdx)
            {
                int32_t partialLength = (params.host_context_lengths[batchIdx] + mCpSize - 1) / mCpSize;
                maxPartialLength = std::max(maxPartialLength, partialLength);
                partialTokenNum += partialLength;
            }
            auto partialHeads = mNumHeads / mCpSize;

            if (mFP8ContextFMHA)
            {
                invokeCpTransposeToSeqMajor2(reinterpret_cast<__nv_fp8_e4m3*>(gatherInBuffer),
                    reinterpret_cast<__nv_fp8_e4m3 const*>(gatherOutBuffer), params.q_seq_lengths, cu_q_seqlens,
                    cu_cp_partial_seqlens, mCpSize, maxPartialLength, params.batch_size, partialHeads, getHeadSize(),
                    stream);
            }
            else
            {
                invokeCpTransposeToSeqMajor2(gatherInBuffer, gatherOutBuffer, params.q_seq_lengths, cu_q_seqlens,
                    cu_cp_partial_seqlens, mCpSize, maxPartialLength, params.batch_size, partialHeads, getHeadSize(),
                    stream);
            }

#if ENABLE_MULTI_DEVICE
            const size_t elementNum = partialTokenNum * getHeadSize() * partialHeads;
            ncclGroupStart();
            for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
            {
                if (cpIdx != mCpRank)
                {
                    if (mFP8ContextFMHA)
                    {
                        NCCLCHECK(ncclSend(reinterpret_cast<__nv_fp8_e4m3*>(gatherInBuffer) + cpIdx * elementNum,
                            elementNum, ncclInt8, cpIdx, *mCpNcclComm, stream));
                        NCCLCHECK(ncclRecv(reinterpret_cast<__nv_fp8_e4m3*>(gatherOutBuffer) + cpIdx * elementNum,
                            elementNum, ncclInt8, cpIdx, *mCpNcclComm, stream));
                    }
                    else
                    {
                        NCCLCHECK(ncclSend(gatherInBuffer + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType],
                            cpIdx, *mCpNcclComm, stream));
                        NCCLCHECK(ncclRecv(gatherOutBuffer + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType],
                            cpIdx, *mCpNcclComm, stream));
                    }
                }
            }
            ncclGroupEnd();
#endif

            if (mFP8ContextFMHA)
            {
                invokeCpTransposeToSeqMajor<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3*>(params.context_buf),
                    reinterpret_cast<__nv_fp8_e4m3 const*>(gatherInBuffer),
                    reinterpret_cast<__nv_fp8_e4m3 const*>(gatherOutBuffer), partialTokenNum, mCpSize, partialHeads,
                    getHeadSize(), mCpRank, stream);
            }
            else
            {
                invokeCpTransposeToSeqMajor<T>((T*) params.context_buf, gatherInBuffer, gatherOutBuffer,
                    partialTokenNum, mCpSize, partialHeads, getHeadSize(), mCpRank, stream);
            }
            sync_check_cuda_error();
        }
    }
    else
    {
        TLLM_CHECK_DEBUG_WITH_INFO(params.logn_scaling_ptr == nullptr, "Unfused MHA does not support logn scaling");
        cudaMemsetAsync(k_buf_2_, 0,
            reinterpret_cast<int8_t*>(v_buf_2_) - reinterpret_cast<int8_t*>(k_buf_2_) + v_buf_2_size, stream);

        if (!isCrossAttention())
        {
            invokeAddFusedQKVBiasTranspose(q_buf_2_, k_buf_2_, v_buf_2_, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.q_seq_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
                mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error();
        }
        else
        {
            invokeAddFusedQKVBiasTranspose(q_buf_2_, (T*) nullptr, (T*) nullptr, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.q_seq_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
                mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error();

            invokeAddFusedQKVBiasTranspose((T*) nullptr, k_buf_2_, v_buf_2_, const_cast<T*>(params.cross_kv),
                const_cast<T*>(params.qkv_bias), params.encoder_input_lengths,
                mRemovePadding ? encoder_padding_offset : nullptr, params.batch_size, params.cross_kv_length,
                params.num_encoder_tokens, 0, mNumKVHeads, getHeadSize(), mRotaryEmbeddingDim,
                mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale, mRotaryEmbeddingMaxPositions,
                position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error();
        }

        if (useKVCache())
        {
            invokeTranspose4dBatchMajor(k_buf_2_, v_buf_2_, kv_cache_buffer, params.batch_size,
                isCrossAttention() ? params.cross_kv_length : params.input_seq_length,
                isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size, getHeadSize(),
                mNumKVHeads, cache_type, params.kv_scale_orig_quant,
                isCrossAttention() ? params.encoder_input_lengths : params.q_seq_lengths, stream);
        }
        sync_check_cuda_error();

        T const* linear_bias_slopes = isALiBi() ? params.alibi_slopes : nullptr;
        T const* relative_attention_bias = isRelativePosition() ? params.relative_attention_bias : nullptr;
        int const relative_attention_bias_stride = isRelativePosition() ? params.relative_attention_bias_stride : 0;
        int const max_distance = mMaxDistance;
        cudaDataType_t gemm_out_data_type = is_qk_buf_float_ ? CUDA_R_32F : gemm_data_type;
        void* gemm_out_buf_ = is_qk_buf_float_ ? static_cast<void*>(qk_buf_float_) : static_cast<void*>(qk_buf_);
        if (mNumKVHeads == 1)
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,
                attention_seq_len_1 * mNumHeads,
                getHeadSize(),
                qk_scale_gemm, k_buf_2_, gemm_data_type,
                getHeadSize(),
                attention_seq_len_2 * getHeadSize(),
                q_buf_2_, gemm_data_type,
                getHeadSize(),
                attention_seq_len_1 * mNumHeads * getHeadSize(),
                0.0f, gemm_out_buf_, gemm_out_data_type,
                attention_seq_len_2,
                attention_seq_len_1 * mNumHeads * attention_seq_len_2,
                params.batch_size,
                CUDA_R_32F);
        }
        else if (mNumKVHeads == mNumHeads)
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,
                attention_seq_len_1,
                getHeadSize(),
                qk_scale_gemm, k_buf_2_, gemm_data_type,
                getHeadSize(),
                attention_seq_len_2 * getHeadSize(),
                q_buf_2_, gemm_data_type,
                getHeadSize(),
                attention_seq_len_1 * getHeadSize(),
                0.0f, gemm_out_buf_, gemm_out_data_type,
                attention_seq_len_2,
                attention_seq_len_2 * attention_seq_len_1,
                params.batch_size * mNumHeads,
                CUDA_R_32F);
        }
        else
        {
            int const num_qheads_per_kv_head = mNumHeads / mNumKVHeads;
            for (int ki = 0; ki < mNumKVHeads; ++ki)
            {
                T* qptr = q_buf_2_ + (ki * num_qheads_per_kv_head * attention_seq_len_1 * getHeadSize());
                T* kptr = k_buf_2_ + (ki * attention_seq_len_2 * getHeadSize());
                int const qk_offset = ki * attention_seq_len_1 * num_qheads_per_kv_head * attention_seq_len_2;
                void* qkptr = is_qk_buf_float_ ? static_cast<void*>(qk_buf_float_ + qk_offset)
                                               : static_cast<void*>(qk_buf_ + qk_offset);
                mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                    attention_seq_len_2,
                    attention_seq_len_1 * num_qheads_per_kv_head,
                    getHeadSize(),
                    qk_scale_gemm, kptr, gemm_data_type,
                    getHeadSize(),
                    mNumKVHeads * attention_seq_len_2 * getHeadSize(),
                    qptr, gemm_data_type,
                    getHeadSize(),
                    attention_seq_len_1 * mNumHeads * getHeadSize(),
                    0.0f, qkptr, gemm_out_data_type,
                    attention_seq_len_2,
                    attention_seq_len_1 * mNumHeads * attention_seq_len_2,
                    params.batch_size,
                    CUDA_R_32F);
            }
        }

        if (is_qk_buf_float_ == true)
        {
            if (isRelativePosition())
            {
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_float_, relative_attention_bias, params.batch_size,
                    mNumHeads, attention_seq_len_1,
                    isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size, stream,
                    max_distance > 0, relative_attention_bias_stride, max_distance, false);
            }

            MaskedSoftmaxParam<T, float> param;
            param.attention_score = qk_buf_;
            param.qk = qk_buf_float_;
            param.attention_mask = attention_mask;
            param.batch_size = params.batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.attn_logit_softcapping_scale = mAttnLogitSoftcappingScale;
            param.attn_logit_softcapping_inverse_scale = 1.0f / mAttnLogitSoftcappingScale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);
            param.block_sparse_attn = mMaskType == AttentionMaskType::BLOCKSPARSE;
            param.block_sparse_params = mBlockSparseParams;
            param.q_seq_lengths = params.q_seq_lengths;
            invokeMaskedSoftmax(param, stream);
        }
        else
        {
            if (isRelativePosition())
            {
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_, relative_attention_bias, params.batch_size, mNumHeads,
                    attention_seq_len_1,
                    isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size, stream,
                    max_distance > 0, relative_attention_bias_stride, max_distance, false);
            }

            MaskedSoftmaxParam<T, T> param;
            param.attention_score = qk_buf_;
            param.qk = qk_buf_;
            param.attention_mask = attention_mask;
            param.batch_size = params.batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.attn_logit_softcapping_scale = mAttnLogitSoftcappingScale;
            param.attn_logit_softcapping_inverse_scale = 1.0f / mAttnLogitSoftcappingScale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);
            param.block_sparse_attn = mMaskType == AttentionMaskType::BLOCKSPARSE;
            param.block_sparse_params = mBlockSparseParams;
            param.q_seq_lengths = params.q_seq_lengths;
            invokeMaskedSoftmax(param, stream);
        }

        if (mNumKVHeads == 1)
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
                getHeadSize(),
                mNumHeads * attention_seq_len_1,
                attention_seq_len_2,
                v_buf_2_,
                getHeadSize(),
                getHeadSize() * attention_seq_len_2,
                qk_buf_,
                attention_seq_len_2,
                attention_seq_len_2 * mNumHeads * attention_seq_len_1,
                qkv_buf_2_,
                getHeadSize(),
                getHeadSize() * mNumHeads * attention_seq_len_1,
                params.batch_size
            );
        }
        else if (mNumKVHeads == mNumHeads)
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N, getHeadSize(), attention_seq_len_1,
                attention_seq_len_2, v_buf_2_, getHeadSize(), attention_seq_len_2 * getHeadSize(), qk_buf_,
                attention_seq_len_2, attention_seq_len_1 * attention_seq_len_2, qkv_buf_2_, getHeadSize(),
                attention_seq_len_1 * getHeadSize(), params.batch_size * mNumHeads);
        }
        else
        {
            int const num_qheads_per_kv_head = mNumHeads / mNumKVHeads;
            for (int ki = 0; ki < mNumKVHeads; ++ki)
            {
                T* qkptr = qk_buf_ + (ki * num_qheads_per_kv_head * attention_seq_len_1 * attention_seq_len_2);
                T* vptr = v_buf_2_ + (ki * attention_seq_len_2 * getHeadSize());
                T* qkvptr = qkv_buf_2_ + (ki * attention_seq_len_1 * num_qheads_per_kv_head * getHeadSize());
                mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
                    getHeadSize(),
                    num_qheads_per_kv_head * attention_seq_len_1,
                    attention_seq_len_2,
                    vptr,
                    getHeadSize(),
                    mNumKVHeads * getHeadSize() * attention_seq_len_2,
                    qkptr,
                    attention_seq_len_2,
                    attention_seq_len_2 * mNumHeads * attention_seq_len_1,
                    qkvptr,
                    getHeadSize(),
                    getHeadSize() * mNumHeads * attention_seq_len_1,
                    params.batch_size
                );
            }
        }

        if (!mRemovePadding)
        {
            invokeTransposeQKV(static_cast<T*>(params.context_buf), qkv_buf_2_, params.batch_size, attention_seq_len_1,
                mNumHeads, getHeadSize(), (float*) nullptr, 0, stream);
        }
        else
        {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_, static_cast<T*>(params.context_buf), params.num_tokens,
                params.batch_size, attention_seq_len_1, mNumHeads, getHeadSize(), padding_offset, (float*) nullptr, 0,
                stream);
        }
    }
    return 0;
}

template int GPTAttentionPluginCommon::enqueueContext<half, KVLinearBuffer>(
    EnqueueContextParams<half> const& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueContext<float, KVLinearBuffer>(
    EnqueueContextParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueContext<__nv_bfloat16, KVLinearBuffer>(
    EnqueueContextParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

template int GPTAttentionPluginCommon::enqueueContext<half, KVBlockArray>(
    EnqueueContextParams<half> const& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueContext<float, KVBlockArray>(
    EnqueueContextParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueContext<__nv_bfloat16, KVBlockArray>(
    EnqueueContextParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

bool GPTAttentionPluginCommon::mForceMultiBlockWarned = false;

template <typename T, typename KVCacheBuffer>
int GPTAttentionPluginCommon::enqueueGeneration(EnqueueGenerationParams<T> const& params, cudaStream_t stream)
{

    int const num_heads = mNumHeads;
    int const num_kv_heads = mNumKVHeads;
    int const head_size = getHeadSize();
    int const local_hidden_units_qo = num_heads * head_size;
    int const local_hidden_units_kv = num_kv_heads * head_size;
    PositionEmbeddingType const position_embedding_type = mPositionEmbeddingType;
    float const q_scaling = mQScaling;
    float const* logn_scaling_ptr = isLognScaling() ? params.logn_scaling_ptr : nullptr;
    T const* relative_attention_bias = isRelativePosition() ? params.relative_attention_bias : nullptr;
    int const relative_attention_bias_stride = isRelativePosition() ? params.relative_attention_bias_stride : 0;
    int const max_distance = mMaxDistance;
    bool const* finished = nullptr;
    bool const has_ia3 = false;

    auto const quant_option = tc::QuantMode::fromDescription();
    float const* qkv_scale_out = nullptr;

    int const* ia3_tasks = nullptr;
    T const* ia3_key_weights = nullptr;
    T const* ia3_value_weights = nullptr;

    int32_t const batch_beam = params.beam_width * params.num_requests;

    KVCacheBuffer kv_cache_buffer;
    auto const elemSize = mKVCacheQuantMode.hasKvCacheQuant() ? sizeof(int8_t) : sizeof(T);
    auto const sizePerToken = num_kv_heads * head_size * elemSize;
    if (useKVCache())
    {
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            using BufferDataType = typename KVCacheBuffer::DataType;
            kv_cache_buffer = KVBlockArray(batch_beam, params.max_blocks_per_sequence, mTokensPerBlock, sizePerToken,
                params.cyclic_attention_window_size, params.max_cyclic_attention_window_size, params.sink_token_length,
                params.can_use_one_more_block, params.host_primary_pool_pointer, params.host_secondary_pool_pointer,
                reinterpret_cast<BufferDataType*>(params.block_offsets));
        }
        else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
        {
            using BufferDataType = typename KVCacheBuffer::DataType;
            kv_cache_buffer = KVLinearBuffer(batch_beam, params.max_attention_window, sizePerToken,
                params.cyclic_attention_window_size, params.sink_token_length, false,
                reinterpret_cast<BufferDataType*>(params.key_value_cache));
        }
    }
    sync_check_cuda_error();

#ifndef NDEBUG
    debugCheckSemaphores(stream);
#endif

    if (!(mIsSpecDecodingEnabled && mUseSpecDecoding))
    {
        int64_t multi_block_mode_val = params.runtime_perf_knobs[0];
        mMultiBlockMode = multi_block_mode_val == 1;

        if (!mMultiBlockMode)
        {
            char* isForceMultiBlockModeChar = std::getenv("FORCE_MULTI_BLOCK_MODE");
            bool isForceMultiBlockMode
                = (isForceMultiBlockModeChar != nullptr && std::string(isForceMultiBlockModeChar) == "ON");
            mMultiBlockMode = isForceMultiBlockMode;
        }
    }

    {
        XQAParams xqaParams{};
        if (suggestify::kernels::XQADispatchHelper<T, KVCacheBuffer>::CanSupport && mDecoderXQARunner.get() != nullptr
            && this->template convertMMHAParamsToXQAParams<T, KVCacheBuffer>(
                xqaParams, params,false)
            && mDecoderXQARunner->shouldUse(xqaParams,false))
        {
            TLLM_LOG_DEBUG("XQA kernels are selected in the generation phase.");
            mDecoderXQARunner->template dispatch<KVCacheBuffer>(xqaParams, kv_cache_buffer, stream);
            return 0;
        }
        else if (mIsSpecDecodingEnabled && mUseSpecDecoding)
        {
            TLLM_CHECK_WITH_INFO(false, "No available XQA kernels are found for speculative decoding mode.");
        }
    }

    int timestep = params.max_past_kv_length;
    int const max_timesteps = std::min(timestep, params.cyclic_attention_window_size);
    int estimated_min_multi_block_count
        = estimate_min_multi_block_count<T>(max_timesteps, mMaxSharedMemoryPerBlockOptin - 2048);

    if (!mMultiBlockMode && !mForceMultiBlockWarned && estimated_min_multi_block_count > 1)
    {
        mForceMultiBlockWarned = true;
        TLLM_LOG_WARNING(
            "Force using MultiBlockMode in MMHA as shared memory is not enough, "
            "MultiBlockMode may have different accuracy compared to non-MultiBlockMode.");
    }

    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = 0;
    int32_t const max_num_seq_len_tiles = std::max(getMaxNumSeqLenTile(batch_beam), estimated_min_multi_block_count);
    int32_t const min_num_seq_len_tiles = std::max(1, estimated_min_multi_block_count);
    bool const enable_multi_block
        = (mMultiBlockMode && max_num_seq_len_tiles > 1) || estimated_min_multi_block_count > 1;
    size_t const partial_out_size
        = enable_multi_block ? sizeof(T) * batch_beam * mNumHeads * mHeadSize * max_num_seq_len_tiles : 0;
    size_t const partial_sum_size
        = enable_multi_block ? sizeof(float) * batch_beam * mNumHeads * max_num_seq_len_tiles : 0;
    size_t const partial_max_size
        = enable_multi_block ? sizeof(float) * batch_beam * mNumHeads * max_num_seq_len_tiles : 0;
    size_t const shift_k_cache_size = (!mPosShiftEnabled || isCrossAttention())
        ? 0
        : sizeof(T) * batch_beam * mNumHeads * mHeadSize * params.max_attention_window;
    size_t const cpMaxPaddedSequenceLength = (batch_beam + mCpSize - 1) / mCpSize * mCpSize;
    size_t const cpWorkspaceSize
        = mCpSize == 1 ? 0 : 2 * sizeof(T) * cpMaxPaddedSequenceLength * (mNumHeads + 2 * mNumKVHeads) * mHeadSize;

    T* partial_out = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_out_size));
    float* partial_sum = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_sum_size));
    float* partial_max = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_max_size));
    T* shift_k_cache = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, shift_k_cache_size));
    T* mhaOutput = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, cpWorkspaceSize));
    T* mhaInput = mhaOutput + cpMaxPaddedSequenceLength * (mNumHeads + 2 * mNumKVHeads) * mHeadSize;

    KVLinearBuffer shift_k_cache_buffer;
    if (useKVCache() && mPosShiftEnabled && !isCrossAttention())
    {
        shift_k_cache_buffer
            = KVLinearBuffer(batch_beam, params.max_attention_window, sizePerToken, params.cyclic_attention_window_size,
                params.sink_token_length, true, reinterpret_cast<int8_t*>(shift_k_cache));
        sync_check_cuda_error();
        KvCacheDataType const kv_cache_type = KvCacheDataType::BASE;
        using DataType = typename SATypeConverter<T>::Type;
        invokeShiftKCache<DataType, KVCacheBuffer>(kv_cache_buffer, shift_k_cache_buffer, kv_cache_type, getHeadSize(),
            timestep, batch_beam, mNumKVHeads, params.beam_width, params.cyclic_attention_window_size,
            params.sink_token_length, params.kv_scale_quant_orig, params.sequence_lengths, params.context_lengths,
            mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
            mRotaryEmbeddingMaxPositions, mPositionEmbeddingType, stream);
    }

    T* attention_input = const_cast<T*>(params.attention_input);
    if (mCpSize > 1)
    {
        auto const totalHeads = mNumHeads + 2 * mNumKVHeads;
        auto const partialHeads = totalHeads / mCpSize;
        auto const partialQHeads = mNumHeads / mCpSize;
        auto const partialKVHeads = mNumKVHeads / mCpSize;
        auto const partialTokenNum = (batch_beam + mCpSize - 1) / mCpSize;


        invokeCpTranspose(mhaOutput, mhaInput, params.attention_input, partialTokenNum, mCpSize, partialQHeads,
            partialKVHeads, mHeadSize, mCpRank, stream);
        sync_check_cuda_error();

#if ENABLE_MULTI_DEVICE
        ncclGroupStart();
        for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
        {
            if (cpIdx != mCpRank)
            {
                NCCLCHECK(ncclSend(mhaOutput + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                    (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                    stream));
                NCCLCHECK(ncclRecv(mhaInput + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                    (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                    stream));
            }
        }
        ncclGroupEnd();
        sync_check_cuda_error();
#endif

        attention_input = mhaInput;
    }
    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> dispatch_params;
    memset(&dispatch_params, 0, sizeof(dispatch_params));
    dispatch_params.mUnfuseQkvGemm = mUnfuseQkvGemm;
    dispatch_params.qkv_buf = attention_input;
    dispatch_params.qkv_bias = params.qkv_bias;
    dispatch_params.logn_scaling_ptr = logn_scaling_ptr;
    dispatch_params.relative_attention_bias = relative_attention_bias;
    dispatch_params.relative_attention_bias_stride = relative_attention_bias_stride;
    dispatch_params.attention_mask = params.attention_mask;
    dispatch_params.attention_mask_stride = params.attention_mask_stride;
    dispatch_params.max_distance = max_distance;
    dispatch_params.cache_indir = params.cache_indir;
    dispatch_params.context_buf = mCpSize > 1 ? mhaOutput : params.context_buf;
    dispatch_params.finished = finished;
    dispatch_params.sequence_lengths
        = params.sequence_lengths;
    dispatch_params.max_batch_size = batch_beam;
    dispatch_params.inference_batch_size = batch_beam;
    dispatch_params.beam_width = params.beam_width;
    dispatch_params.head_num = mNumHeads / mCpSize;
    dispatch_params.kv_head_num = mNumKVHeads / mCpSize;
    dispatch_params.size_per_head = getHeadSize();
    dispatch_params.rotary_embedding_dim = mRotaryEmbeddingDim;
    dispatch_params.position_embedding_type = mPositionEmbeddingType;
    dispatch_params.max_attention_window = params.max_attention_window;
    dispatch_params.cyclic_attention_window_size = params.cyclic_attention_window_size;
    dispatch_params.sink_token_length = isCrossAttention() ? 0 : params.sink_token_length;
    dispatch_params.input_lengths = params.context_lengths;
    dispatch_params.timestep = timestep;
    dispatch_params.q_scaling = q_scaling;
    dispatch_params.attn_logit_softcapping_scale = mAttnLogitSoftcappingScale;
    dispatch_params.linear_bias_slopes = isALiBi() ? params.alibi_slopes : nullptr;
    dispatch_params.ia3_tasks = ia3_tasks;
    dispatch_params.ia3_key_weights = ia3_key_weights;
    dispatch_params.ia3_value_weights = ia3_value_weights;
    dispatch_params.qkv_scale_out = qkv_scale_out;
    dispatch_params.fp8_context_fmha = mFP8ContextFMHA;
    dispatch_params.attention_out_scale = params.attention_output_orig_quant;
    dispatch_params.quant_option = quant_option;
    dispatch_params.multi_block_mode = enable_multi_block;
    dispatch_params.max_seq_len_tile = max_num_seq_len_tiles;
    dispatch_params.min_seq_len_tile = min_num_seq_len_tiles;
    dispatch_params.partial_out = partial_out;
    dispatch_params.partial_sum = partial_sum;
    dispatch_params.partial_max = partial_max;
    dispatch_params.block_counter = mMultiBlockSemaphores.get();
    dispatch_params.kv_cache_quant_mode = mKVCacheQuantMode;
    dispatch_params.kv_scale_orig_quant = params.kv_scale_orig_quant;
    dispatch_params.kv_scale_quant_orig = params.kv_scale_quant_orig;
    dispatch_params.kv_block_array = kv_cache_buffer;
    dispatch_params.shift_k_cache_buffer = shift_k_cache_buffer;
    dispatch_params.multi_processor_count = mMultiProcessorCount;
    dispatch_params.rotary_embedding_base = mRotaryEmbeddingBase;
    dispatch_params.rotary_embedding_scale_type = mRotaryEmbeddingScaleType;
    dispatch_params.rotary_embedding_scale = mRotaryEmbeddingScale;
    dispatch_params.rotary_embedding_inv_freq_cache = params.rotary_inv_freq;
    dispatch_params.rotary_embedding_short_m_scale = mRotaryEmbeddingShortMscale;
    dispatch_params.rotary_embedding_long_m_scale = mRotaryEmbeddingLongMscale;
    dispatch_params.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
    dispatch_params.rotary_embedding_original_max_positions = mRotaryEmbeddingOriginalMaxPositions;
    dispatch_params.position_shift_enabled = mPosShiftEnabled;
    dispatch_params.rotary_cogvlm_vision_start = mVisionStart;
    dispatch_params.rotary_cogvlm_vision_length = mVisionLength;
    dispatch_params.cross_attention = isCrossAttention();
    dispatch_params.memory_length_per_sample = params.encoder_input_lengths;
    dispatch_params.block_sparse_attention = mMaskType == AttentionMaskType::BLOCKSPARSE;
    dispatch_params.block_sparse_params = mBlockSparseParams;
    dispatch_params.mrope_position_deltas = params.mrope_position_deltas;

    using DataType = typename SATypeConverter<T>::Type;
    if (!isCrossAttention())
    {
        Masked_multihead_attention_params<DataType> mmha_params;
        fusedQKV_masked_attention_dispatch(mmha_params, dispatch_params, stream);
    }
    else
    {
        Cross_multihead_attention_params<DataType> mmhca_params;
        fusedQKV_masked_attention_dispatch(mmhca_params, dispatch_params, stream);
    }
    if (mCpSize > 1)
    {

        auto partialHeads = mNumHeads / mCpSize;
        auto const partialTokenNum = (batch_beam + mCpSize - 1) / mCpSize;

#if ENABLE_MULTI_DEVICE
        const size_t elementNum = partialTokenNum * getHeadSize() * partialHeads;
        ncclGroupStart();
        for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
        {
            if (cpIdx != mCpRank)
            {
                if (mFP8ContextFMHA)
                {
                    NCCLCHECK(ncclSend(reinterpret_cast<__nv_fp8_e4m3*>(mhaOutput) + cpIdx * elementNum, elementNum,
                        ncclInt8, cpIdx, *mCpNcclComm, stream));
                    NCCLCHECK(ncclRecv(reinterpret_cast<__nv_fp8_e4m3*>(mhaInput) + cpIdx * elementNum, elementNum,
                        ncclInt8, cpIdx, *mCpNcclComm, stream));
                }
                else
                {
                    NCCLCHECK(ncclSend(mhaOutput + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx,
                        *mCpNcclComm, stream));
                    NCCLCHECK(ncclRecv(mhaInput + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx,
                        *mCpNcclComm, stream));
                }
            }
        }
        ncclGroupEnd();
#endif

        if (mFP8ContextFMHA)
        {
            invokeCpTransposeToSeqMajor<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3*>(params.context_buf),
                reinterpret_cast<__nv_fp8_e4m3 const*>(mhaOutput), reinterpret_cast<__nv_fp8_e4m3 const*>(mhaInput),
                partialTokenNum, mCpSize, partialHeads, getHeadSize(), mCpRank, stream);
        }
        else
        {
            invokeCpTransposeToSeqMajor<T>((T*) params.context_buf, mhaOutput, mhaInput, partialTokenNum, mCpSize,
                partialHeads, getHeadSize(), mCpRank, stream);
        }
        sync_check_cuda_error();
    }
    return 0;
}

template int GPTAttentionPluginCommon::enqueueGeneration<half, KVLinearBuffer>(
    EnqueueGenerationParams<half> const& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueGeneration<float, KVLinearBuffer>(
    EnqueueGenerationParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueGeneration<__nv_bfloat16, KVLinearBuffer>(
    EnqueueGenerationParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

template int GPTAttentionPluginCommon::enqueueGeneration<half, KVBlockArray>(
    EnqueueGenerationParams<half> const& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueGeneration<float, KVBlockArray>(
    EnqueueGenerationParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueGeneration<__nv_bfloat16, KVBlockArray>(
    EnqueueGenerationParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

template <typename T, typename KVCacheBuffer>
void GPTAttentionPluginCommon::prepareEnqueueGeneration(EnqueueGenerationParams<T> const& params)
{
    XQAParams xqaParams{};
    if (suggestify::kernels::XQADispatchHelper<T, KVCacheBuffer>::CanSupport && mDecoderXQARunner.get() != nullptr
        && this->template convertMMHAParamsToXQAParams<T, KVCacheBuffer>(xqaParams, params,true)
        && mDecoderXQARunner->shouldUse(xqaParams,true))
    {
        TLLM_LOG_DEBUG("Preparing XQA kernels in prepareEnqueueGeneration.");
        mDecoderXQARunner->prepare(xqaParams);
    }
}

template void GPTAttentionPluginCommon::prepareEnqueueGeneration<half, KVLinearBuffer>(
    EnqueueGenerationParams<half> const& params);

template void GPTAttentionPluginCommon::prepareEnqueueGeneration<float, KVLinearBuffer>(
    EnqueueGenerationParams<float> const& params);

#ifdef ENABLE_BF16
template void GPTAttentionPluginCommon::prepareEnqueueGeneration<__nv_bfloat16, KVLinearBuffer>(
    EnqueueGenerationParams<__nv_bfloat16> const& params);
#endif

template void GPTAttentionPluginCommon::prepareEnqueueGeneration<half, KVBlockArray>(
    EnqueueGenerationParams<half> const& params);

template void GPTAttentionPluginCommon::prepareEnqueueGeneration<float, KVBlockArray>(
    EnqueueGenerationParams<float> const& params);

#ifdef ENABLE_BF16
template void GPTAttentionPluginCommon::prepareEnqueueGeneration<__nv_bfloat16, KVBlockArray>(
    EnqueueGenerationParams<__nv_bfloat16> const& params);
#endif

int GPTAttentionPluginCommon::initialize() noexcept
{
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();

    getEnvMmhaMultiblockDebug();
    getEnvMmhaBlocksPerSequence();

    mCublasWrapper.reset(new tc::CublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, nullptr));

    if (mEnableContextFMHA)
    {
        Data_type data_type;
        if (mType == nvinfer1::DataType::kHALF)
        {
            data_type = DATA_TYPE_FP16;
        }
        else if (mType == nvinfer1::DataType::kBF16)
        {
            data_type = DATA_TYPE_BF16;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "GPTAttentionPlugin received wrong data type.");
        }

        if (mFP8ContextFMHA)
        {
            data_type = DATA_TYPE_E4M3;
        }

        MHARunnerFixedParams fmhaParams{};
        fmhaParams.dataType = data_type;
        fmhaParams.forceFp32Acc = false;
        if (isCrossAttention())
        {
            fmhaParams.attentionMaskType = ContextAttentionMaskType::PADDING;
            fmhaParams.attentionInputLayout
                = mPagedKVCache ? AttentionInputLayout::Q_PAGED_KV : AttentionInputLayout::Q_CONTIGUOUS_KV;
        }
        else
        {
            fmhaParams.attentionMaskType = ContextAttentionMaskType::CAUSAL;
            fmhaParams.attentionInputLayout = (mPagedKVCache && mPagedContextFMHA && !mIsMLAEnabled)
                ? AttentionInputLayout::Q_PAGED_KV
                : AttentionInputLayout::PACKED_QKV;
        }
        if (useCustomMask())
        {
            fmhaParams.attentionMaskType = ContextAttentionMaskType::CUSTOM_MASK;
        }
        fmhaParams.isSPadded = !mRemovePadding;
        fmhaParams.numQHeads = mNumHeads / mCpSize;
        fmhaParams.numKvHeads = mNumKVHeads / mCpSize;
        fmhaParams.headSize = mHeadSize;
        if (mIsMLAEnabled)
        {
            fmhaParams.numKvHeads = mNumHeads;
            fmhaParams.headSize = mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim;
            fmhaParams.headSizeV = mMLAParams.v_head_dim;
        }
        fmhaParams.qScaling = mQScaling;
        fmhaParams.attnLogitSoftcappingScale = mAttnLogitSoftcappingScale;
        fmhaParams.hasAlibi = isALiBi();
        fmhaParams.scaleAlibi = isAliBiWithScale();
        if (mTpSize > 1)
        {
            fmhaParams.tpSize = mTpSize;
            fmhaParams.tpRank = mTpRank;
        }
        else if (mCpSize > 1)
        {
            fmhaParams.tpSize = mCpSize;
            fmhaParams.tpRank = mCpRank;
        }

        mFMHARunner.reset(new FusedMHARunnerV2(fmhaParams));

        if (mIsMLAEnabled)
        {
            MHARunnerFixedParams fmhaParams{};
            fmhaParams.dataType = data_type;
            fmhaParams.forceFp32Acc = false;
            fmhaParams.attentionMaskType
                = useCustomMask() ? ContextAttentionMaskType::CUSTOM_MASK : ContextAttentionMaskType::PADDING;
            fmhaParams.attentionInputLayout = AttentionInputLayout::Q_PAGED_KV;
            fmhaParams.isSPadded = !mRemovePadding;
            fmhaParams.numQHeads = 1;
            fmhaParams.numKvHeads = 1;
            fmhaParams.headSize = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
            fmhaParams.headSizeV = mMLAParams.kv_lora_rank;
            fmhaParams.qScaling = mQScaling * sqrt((float) (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim))
                / sqrtf((float) (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim));
            fmhaParams.attnLogitSoftcappingScale = mAttnLogitSoftcappingScale;
            fmhaParams.hasAlibi = isALiBi();
            fmhaParams.scaleAlibi = isAliBiWithScale();
            fmhaParams.tpSize = mTpSize;
            fmhaParams.tpRank = mTpRank;
            mDecoderFMHARunner.reset(new FusedMHARunnerV2(fmhaParams));

            TLLM_CHECK_WITH_INFO(mFMHARunner->isFmhaSupported() && mDecoderFMHARunner->isFmhaSupported(),
                "Deepseek should be supported by fmha in context and generation part.");
        }

        mEnableContextFMHA = mFMHARunner->isFmhaSupported();

        TLLM_CHECK_WITH_INFO(
            !useCustomMask() || mEnableContextFMHA, "Only Context FMHA supports custom mask input currently.");
    }

    bool useXQAKernels = (mEnableXQA || mIsSpecDecodingEnabled) && !isCrossAttention()
        && (mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16);

    if (useXQAKernels)
    {
        Data_type xqa_runner_data_type;
        if (mType == nvinfer1::DataType::kHALF)
        {
            xqa_runner_data_type = DATA_TYPE_FP16;
        }
        else if (mType == nvinfer1::DataType::kBF16)
        {
            xqa_runner_data_type = DATA_TYPE_BF16;
        }
        TLLM_LOG_DEBUG("Enabling XQA kernels for GPTAttention.");
        if (mIsSpecDecodingEnabled)
        {
            TLLM_CHECK_WITH_INFO(mNumHeads % mNumKVHeads == 0, "mNumHeads should be multiples of mNumKVHeads.");
            TLLM_CHECK_WITH_INFO(!mMultiBlockMode, "Medusa doesn't support multi-block mode.");
        }

        mDecoderXQARunner.reset(new DecoderXQARunner(
            xqa_runner_data_type, mNumHeads / mCpSize, mNumKVHeads / mCpSize, mHeadSize, mMultiBlockMode));
    }
    else if (mIsSpecDecodingEnabled)
    {
        TLLM_CHECK_WITH_INFO(false, "Speculative decoding mode doesn't support the data type or cross attention.");
    }

    if (mNbMultiBlockSemaphores != 0)
    {
        reserveSemaphoreArray(mNbMultiBlockSemaphores);
    }

    if (isBuilding())
    {
        return 0;
    }
#if ENABLE_MULTI_DEVICE
    if (mCpSize > 1 && COMM_SESSION.getSize() > 1)
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mCpNcclComm = getComm(mCpGroup);
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    }
#endif
    return 0;
}

void GPTAttentionPluginCommon::destroy() noexcept
{
    delete this;
}

size_t GPTAttentionPluginCommon::getCommonSerializationSize() const noexcept
{
    return sizeof(mLayerIdx) + sizeof(mNumHeads) + +sizeof(mVisionStart) + sizeof(mVisionLength) + sizeof(mNumKVHeads)
        + sizeof(mLayerIdxInCachePool) + sizeof(mHeadSize) + sizeof(mUnidirectional) + sizeof(mQScaling)
        + sizeof(mAttnLogitSoftcappingScale) + sizeof(mPositionEmbeddingType) + sizeof(mRotaryEmbeddingDim)
        + sizeof(mRotaryEmbeddingBase) + sizeof(mRotaryEmbeddingScaleType) + sizeof(mRotaryEmbeddingScale)
        + sizeof(mRotaryEmbeddingShortMscale) + sizeof(mRotaryEmbeddingLongMscale)
        + sizeof(mRotaryEmbeddingMaxPositions) + sizeof(mRotaryEmbeddingOriginalMaxPositions) + sizeof(mTpSize)
        + sizeof(mTpRank) + sizeof(mEnableContextFMHA) + sizeof(mFMHAForceFP32Acc) + sizeof(mMultiBlockMode)
        + sizeof(mEnableXQA) + sizeof(unsigned int)
        + sizeof(mRemovePadding) + sizeof(mMaskType) + sizeof(mBlockSparseParams) + sizeof(mPagedKVCache)
        + sizeof(mTokensPerBlock) + sizeof(mType) + sizeof(mMaxContextLength) + sizeof(mQKVBiasEnabled)
        + sizeof(mCrossAttention) + sizeof(mMaxDistance) + sizeof(mPosShiftEnabled) + sizeof(mDenseContextFMHA)
        + sizeof(mPagedContextFMHA) + sizeof(mFP8ContextFMHA) + sizeof(mHasFullAttentionMask) + sizeof(mUseKVCache)
        + sizeof(mUnfuseQkvGemm) + sizeof(mUseLognScaling) + sizeof(mIsSpecDecodingEnabled) + sizeof(mUseSpecDecoding)
        + sizeof(mSpecDecodingIsGenerationLengthVariable) + sizeof(mSpecDecodingMaxGenerationLength)
        + sizeof(mNbMultiBlockSemaphores) + sizeof(mIsMLAEnabled) + sizeof(mMLAParams) + sizeof(mSkipAttn)
        + sizeof(uint32_t)
        + sizeof(mCpSize) + sizeof(mCpRank) + sizeof(int32_t) * mCpGroup.size()
        + DecoderXQARunner::getResourceGlobal()->getSerializationSize();
}

void GPTAttentionPluginCommon::serializeCommon(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mLayerIdx);
    write(d, mNumHeads);
    write(d, mVisionStart);
    write(d, mVisionLength);
    write(d, mNumKVHeads);
    write(d, mLayerIdxInCachePool);
    write(d, mHeadSize);
    write(d, mUnidirectional);
    write(d, mQScaling);
    write(d, mAttnLogitSoftcappingScale);
    write(d, mPositionEmbeddingType);
    write(d, mRotaryEmbeddingDim);
    write(d, mRotaryEmbeddingBase);
    write(d, mRotaryEmbeddingScaleType);
    write(d, mRotaryEmbeddingScale);
    write(d, mRotaryEmbeddingShortMscale);
    write(d, mRotaryEmbeddingLongMscale);
    write(d, mRotaryEmbeddingMaxPositions);
    write(d, mRotaryEmbeddingOriginalMaxPositions);
    write(d, mTpSize);
    write(d, mTpRank);
    write(d, mUnfuseQkvGemm);
    write(d, mUseLognScaling);
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mMultiBlockMode);
    write(d, mEnableXQA);
    write(d, mKVCacheQuantMode.value());
    write(d, mRemovePadding);
    write(d, mMaskType);
    write(d, mBlockSparseParams);
    write(d, mPagedKVCache);
    write(d, mTokensPerBlock);
    write(d, mType);
    write(d, mMaxContextLength);
    write(d, mQKVBiasEnabled);
    write(d, mCrossAttention);
    write(d, mMaxDistance);
    write(d, mPosShiftEnabled);
    write(d, mDenseContextFMHA);
    write(d, mPagedContextFMHA);
    write(d, mFP8ContextFMHA);
    write(d, mHasFullAttentionMask);
    write(d, mUseKVCache);
    write(d, mIsSpecDecodingEnabled);
    write(d, mUseSpecDecoding);
    write(d, mSpecDecodingIsGenerationLengthVariable);
    write(d, mSpecDecodingMaxGenerationLength);
    write(d, mIsMLAEnabled);
    write(d, mMLAParams);
    write(d, mNbMultiBlockSemaphores);
    write(d, mSkipAttn);
    write(d, mCpSize);
    write(d, mCpRank);

    uint32_t decoderXQARunnerResourceSerializedSize = DecoderXQARunner::getResourceGlobal()->getSerializationSize();
    write(d, decoderXQARunnerResourceSerializedSize);
    DecoderXQARunner::getResourceGlobal()->serialize(d, decoderXQARunnerResourceSerializedSize);
    d += decoderXQARunnerResourceSerializedSize;

    for (auto it = mCpGroup.begin(); it != mCpGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getCommonSerializationSize());
}

void GPTAttentionPluginCommon::terminate() noexcept
{
}

void GPTAttentionPluginCommon::reserveSemaphoreArray(int32_t size)
{
    if (size == 0 || (size <= mNbMultiBlockSemaphores && mMultiBlockSemaphores != nullptr))
    {
        return;
    }
    int32_t* ptr;
    deviceMalloc(&ptr, size, false);
    deviceMemSetZero(ptr, size);
    mMultiBlockSemaphores.reset(ptr);
    mNbMultiBlockSemaphores = size;
}

void GPTAttentionPluginCommon::debugCheckSemaphores(cudaStream_t stream)
{
#ifdef NDEBUG
    TLLM_CHECK_WITH_INFO(false, "debugCheckSemaphores should not be called in release build");
#endif
    if (mNbMultiBlockSemaphores == 0)
    {
        return;
    }
    std::vector<uint32_t> hostBuf(mNbMultiBlockSemaphores);
    TLLM_CUDA_CHECK(cudaMemcpyAsync(hostBuf.data(), mMultiBlockSemaphores.get(),
        sizeof(uint32_t) * mNbMultiBlockSemaphores, cudaMemcpyDeviceToHost, stream));
    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
    TLLM_CHECK(std::count(hostBuf.begin(), hostBuf.end(), 0U) == mNbMultiBlockSemaphores);
}


GPTAttentionPluginCreatorCommon::GPTAttentionPluginCreatorCommon()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("vision_start", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("vision_length", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("unidirectional", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(
        PluginField("attn_logit_softcapping_scale", nullptr, PluginFieldType::kFLOAT32, 0.0));
    mPluginAttributes.emplace_back(PluginField("position_embedding_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_dim", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_base", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_scale_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(
        PluginField("rotary_embedding_short_m_scale", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(
        PluginField("rotary_embedding_long_m_scale", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_max_positions", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(
        PluginField("rotary_embedding_original_max_positions", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("tp_rank", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("unfuse_qkv_gemm", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("kv_cache_quant_mode", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("mask_type", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("paged_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("tokens_per_block", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_context_length", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("qkv_bias_enabled", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("do_cross_attention", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("max_distance", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("pos_shift_enabled", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("dense_context_fmha", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("use_paged_context_fmha", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("use_fp8_context_fmha", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("has_full_attention_mask", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("use_cache", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("is_spec_decoding_enabled", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(
        PluginField("spec_decoding_is_generation_length_variable", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(
        PluginField("spec_decoding_max_generation_length", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("is_mla_enabled", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("q_lora_rank", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("kv_lora_rank", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("qk_nope_head_dim", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("qk_rope_head_dim", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("v_head_dim", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("skip_attn", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("cp_size", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("cp_rank", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("cp_group", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

PluginFieldCollection const* GPTAttentionPluginCreatorCommon::getFieldNames() noexcept
{
    return &mFC;
}
