#pragma once

#include "suggestify/kernels/gptKernels.h"
#include "suggestify/kernels/kvCacheUtils.h"
#include "suggestify/kernels/mlaKernels.h"
#include "suggestify/runtime/iTensor.h"
#include <cuda_runtime_api.h>

namespace suggestify
{
namespace kernels
{

template <typename T>
void invokeAddQKVBiasIA3Transpose(T* q_buf, T* k_buf, T* v_buf, T* Q, T const* bias_Q, T* K, T const* bias_K, T* V,
    T const* bias_V, int const batch_size, int const seq_len, int const head_num, int const size_per_head,
    int const* ia3_tasks, T const* ia3_key_weights, T const* ia3_value_weights, cudaStream_t stream);

template <typename T, typename T_IN>
struct MaskedSoftmaxParam
{
    T* attention_score = nullptr;
    const T_IN* qk = nullptr;
    T const* attention_mask = nullptr;
    int batch_size = 0;
    int q_length = 0;
    int k_length = 0;
    int num_heads = 0;
    T qk_scale = T(0.0f);
    float attn_logit_softcapping_scale = 0.f;
    float attn_logit_softcapping_inverse_scale = 0.f;
    bool block_sparse_attn = false;
    BlockSparseParams block_sparse_params;
    int const* q_seq_lengths = nullptr;

    T const* linear_bias_slopes = nullptr;
};

enum class KvCacheDataType
{
    BASE = 0,
    INT8,
    FP8
};

enum class RotaryPositionEmbeddingType
{
    NONE = 0,
    GPTJ,
    GPT_NEOX,
};

template <typename T, typename KVCacheBuffer>
struct QKVPreprocessingParams
{
    T* qkv_input{nullptr};
    T* cross_kv_input{nullptr};
    void* quantized_qkv_output{nullptr};
    T* q_output{nullptr};
    KVCacheBuffer kv_cache_buffer{};
    T const* qkv_bias{nullptr};

    float const* logn_scaling{nullptr};
    int const* seq_lens{nullptr};
    int const* cache_seq_lens{nullptr};
    int const* encoder_seq_lens{nullptr};
    int const* cu_seq_lens{nullptr};
    int const* cu_kv_seq_lens{nullptr};
    float const* rotary_embedding_inv_freq{nullptr};
    float2 const* rotary_coef_cache_buffer{nullptr};
    float const* kvScaleOrigQuant{nullptr};
    int const* spec_decoding_position_offsets{nullptr};

    float2 const* mrope_rotary_cos_sin{nullptr};
    int32_t const* mrope_position_deltas{nullptr};

    int batch_size{0};
    int max_input_seq_len{0};
    int max_kv_seq_len{0};
    int cyclic_kv_cache_len{0};
    int sink_token_len{0};
    int token_num{0};
    bool remove_padding{true};
    bool cross_attention{false};
    int head_num{0};
    int kv_head_num{0};
    int qheads_per_kv_head{0};
    int size_per_head{0};
    int rotary_embedding_dim{0};
    float rotary_embedding_base{0.};
    RotaryScalingType rotary_scale_type{};
    float rotary_embedding_scale{0.};
    int rotary_embedding_max_positions{0};
    PositionEmbeddingType position_embedding_type{};
    bool position_shift_enabled{false};
    KvCacheDataType cache_type{};
    bool separate_q_kv_output{false};
    bool quantized_fp8_output{false};
    int multi_processor_count{0};
    int rotary_vision_start{0};
    int rotary_vision_length{0};
    int half_rotary_dim{0};
    int q_hidden_size{0};
    int kv_hidden_size{0};
    int hidden_size{0};

    void setCommonParameters()
    {
        half_rotary_dim = rotary_embedding_dim / 2;
        q_hidden_size = head_num * size_per_head;
        kv_hidden_size = kv_head_num * size_per_head;
        hidden_size = q_hidden_size + 2 * kv_hidden_size;
    }

    std::string toString() const
    {
        std::stringstream ss;

        ss << "QKVPreprocessingParams ====================" << std::endl;
        ss << "qkv_input: " << qkv_input << std::endl;
        ss << "cross_kv_input: " << cross_kv_input << std::endl;
        ss << "quantized_qkv_output: " << quantized_qkv_output << std::endl;
        ss << "q_output: " << q_output << std::endl;
        ss << "kv_cache_buffer: " << kv_cache_buffer.data << std::endl;
        ss << "qkv_bias: " << qkv_bias << std::endl;
        ss << "seq_lens: "
           << *(runtime::ITensor::wrap(
                  (void*) seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        ss << "cache_seq_lens: "
           << *(runtime::ITensor::wrap(
                  (void*) cache_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        ss << "encoder_seq_lens: "
           << *(runtime::ITensor::wrap(
                  (void*) encoder_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        ss << "cu_seq_lens: "
           << *(runtime::ITensor::wrap(
                  (void*) cu_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        ss << "cu_kv_seq_lens: "
           << *(runtime::ITensor::wrap(
                  (void*) cu_kv_seq_lens, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})));
        ss << "rotary_embedding_inv_freq: "
           << *(runtime::ITensor::wrap((void*) rotary_embedding_inv_freq, nvinfer1::DataType::kFLOAT,
                  runtime::ITensor::makeShape({batch_size, rotary_embedding_dim / 2})));
        ss << "rotary_coef_cache_buffer: " << rotary_coef_cache_buffer << std::endl;
        ss << "kvScaleOrigQuant: " << kvScaleOrigQuant << std::endl;
        ss << "spec_decoding_position_offsets: " << spec_decoding_position_offsets << std::endl;
        ss << "batch_size: " << batch_size << std::endl;
        ss << "max_input_seq_len: " << max_input_seq_len << std::endl;
        ss << "max_kv_seq_len: " << max_kv_seq_len << std::endl;
        ss << "cyclic_kv_cache_len: " << cyclic_kv_cache_len << std::endl;
        ss << "sink_token_len: " << sink_token_len << std::endl;
        ss << "token_num: " << token_num << std::endl;
        ss << "remove_padding: " << remove_padding << std::endl;
        ss << "cross_attention: " << cross_attention << std::endl;
        ss << "head_num: " << head_num << std::endl;
        ss << "kv_head_num: " << kv_head_num << std::endl;
        ss << "qheads_per_kv_head: " << qheads_per_kv_head << std::endl;
        ss << "size_per_head: " << size_per_head << std::endl;
        ss << "rotary_embedding_dim: " << rotary_embedding_dim << std::endl;
        ss << "rotary_embedding_base: " << rotary_embedding_base << std::endl;
        ss << "rotary_scale_type: " << static_cast<int>(rotary_scale_type) << std::endl;
        ss << "rotary_embedding_scale: " << rotary_embedding_scale << std::endl;
        ss << "rotary_embedding_max_positions: " << rotary_embedding_max_positions << std::endl;
        ss << "position_embedding_type: " << static_cast<int>(position_embedding_type) << std::endl;
        ss << "position_shift_enabled: " << std::boolalpha << position_shift_enabled << std::endl;
        ss << "cache_type: " << static_cast<int>(cache_type) << std::endl;
        ss << "separate_q_kv_output: " << std::boolalpha << separate_q_kv_output << std::endl;
        ss << "quantized_fp8_output: " << quantized_fp8_output << std::endl;
        ss << "multi_processor_count: " << multi_processor_count << std::endl;

        return ss.str();
    }
};

template <typename T, typename T_IN>
void invokeMaskedSoftmax(MaskedSoftmaxParam<T, T_IN>& param, cudaStream_t stream);

template <typename T>
void invokeTransposeQKV(T* dst, T* src, int const batch_size, int const seq_len, int const head_num,
    int const size_per_head, float const* scale, int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddQKVBiasIA3RebuildPadding(T* Q, T const* bias_Q, T* K, T const* bias_K, T* V, T const* bias_V, T* q_buf,
    T* k_buf, T* v_buf, int const batch_size, int const seq_len, int const head_num, int const size_per_head,
    int const valid_word_num, int const* mask_offset, int const* ia3_tasks, T const* ia3_key_weights,
    T const* ia3_value_weights, cudaStream_t stream);

template <typename T>
void invokeTransposeAttentionOutRemovePadding(T* src, T* dst, int const valid_word_num, int const batch_size,
    int const seq_len, int const head_num, int const size_per_head, int const* mask_offset, float const* scale,
    int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, T const* qkv_bias, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, float const* scale, int const int8_mode, cudaStream_t stream);

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, T const* qkv_bias, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, qkv_bias, seq_lens, padding_offset, batch_size, seq_len,
        token_num, head_num, kv_head_num, size_per_head, 0, false, (float*) nullptr, 0, stream);
}

template <typename T>
void invokeAddFusedQKVBiasTranspose(T* q_buf, T* k_buf, T* v_buf, T* QKV, int const* seq_lens,
    int const* padding_offset, int const batch_size, int const seq_len, int const token_num, int const head_num,
    int const kv_head_num, int const size_per_head, int const rotary_embedding_dim, float rotary_embedding_base,
    const RotaryScalingType rotary_scale_type, float rotary_embedding_scale, int const rotary_embedding_max_positions,
    PositionEmbeddingType const position_embedding_type, float const* scale, int const int8_mode, cudaStream_t stream)
{
    invokeAddFusedQKVBiasTranspose(q_buf, k_buf, v_buf, QKV, (T const*) nullptr, seq_lens, padding_offset, batch_size,
        seq_len, token_num, head_num, kv_head_num, size_per_head, rotary_embedding_dim, rotary_embedding_base,
        rotary_scale_type, rotary_embedding_scale, rotary_embedding_max_positions, position_embedding_type, scale,
        int8_mode, stream);
}

template <typename T, typename KVCacheBuffer>
void invokeTranspose4dBatchMajor(T const* k_src, T const* v_src, KVCacheBuffer& kvTable, int const local_batch_size,
    int const seq_len, int const max_attention_window_size, int const size_per_head, int const local_head_num,
    const KvCacheDataType cache_type, float const* kvScaleOrigQuant, int const* sequence_lengths, cudaStream_t stream);

template <typename T, typename T_cache, typename KVCacheBuffer>
void invokeApplyBiasRopeUpdateKVCacheDispatch(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

template <typename T, typename KVCacheBuffer>
void invokeQKVPreprocessing(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    params.setCommonParameters();
    if (params.cache_type == KvCacheDataType::INT8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, int8_t, KVCacheBuffer>(params, stream);
    }
#ifdef ENABLE_FP8
    else if (params.cache_type == KvCacheDataType::FP8)
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, __nv_fp8_e4m3, KVCacheBuffer>(params, stream);
    }
#endif
    else
    {
        invokeApplyBiasRopeUpdateKVCacheDispatch<T, T, KVCacheBuffer>(params, stream);
    }
}

template <typename T, typename T_cache, typename KVCacheBuffer>
void invokeUpdateCyclicKvCacheAfterFmha(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

template <typename T, typename KVCacheBuffer>
void invokeKvCachePostprocessing(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    params.setCommonParameters();
    if (params.cache_type == KvCacheDataType::INT8)
    {
        invokeUpdateCyclicKvCacheAfterFmha<T, int8_t, KVCacheBuffer>(params, stream);
    }
#ifdef ENABLE_FP8
    else if (params.cache_type == KvCacheDataType::FP8)
    {
        invokeUpdateCyclicKvCacheAfterFmha<T, __nv_fp8_e4m3, KVCacheBuffer>(params, stream);
    }
#endif
    else
    {
        invokeUpdateCyclicKvCacheAfterFmha<T, T, KVCacheBuffer>(params, stream);
    }
}

template <typename T, typename BT>
void invokeAddRelativeAttentionBiasUnaligned(T* qk_buf, const BT* relative_attention_bias, int const batch_size,
    int const head_num, int const seq_len, int const max_seq_len, cudaStream_t stream, bool implicit = false,
    int num_buckets = 0, int max_distance = 0, bool bidirectional = true);

template <typename T, typename KVCacheBuffer>
void invokeShiftKCache(KVCacheBuffer const& kvCacheBuffer, KVLinearBuffer const& shiftKCacheBuffer,
    const KvCacheDataType cache_type, int const sizePerHead, int const timestep, int const batch_beam,
    int const kv_head_num, int const beam_width, int const maxKCacheLen, int const sinkTokenLen,
    float const* kScaleQuantOrig, int const* sequence_lengths, int const* input_lengths, int const rotary_embedding_dim,
    float rotary_embedding_base, RotaryScalingType const rotary_scale_type, float rotary_embedding_scale,
    int const rotary_embedding_max_positions, PositionEmbeddingType const position_embedding_type, cudaStream_t stream);

template <typename Dst, typename Src>
void invokeConversion(Dst* dst, Src const* src, int64_t size, float const* __restrict__ scale, cudaStream_t stream);

template <typename T>
void invokeCpTranspose(T* dst, T* dst2, T const* src, int64_t partialLength, int64_t cpSize, int64_t partialQHeads,
    int64_t partialKVHeads, int64_t headSize, int64_t rank, cudaStream_t stream);

template <typename T>
void invokeCpTransposeToSeqMajor(T* dst, T const* srcMyRank, T const* srcOtherRank, int64_t partialLength,
    int64_t cpSize, int64_t newPartialHeads, int64_t headSize, int64_t rank, cudaStream_t stream);

template <typename T>
void invokeCpTranspose2(T* dst, T const* src, int32_t const* q_seq_lengths, int32_t const* cu_q_seqlens,
    int32_t const* cu_cp_partial_seqlens, int64_t cpSize, int64_t maxPartalLength, int64_t batchSize,
    int64_t partialHeads, int64_t headSize, cudaStream_t stream);

template <typename T>
void invokeCpTransposeToSeqMajor2(T* dst, T const* src, int32_t const* q_seq_lengths, int32_t const* cu_q_seqlens,
    int32_t const* cu_cp_partial_seqlens, int64_t cpSize, int64_t maxPartalLength, int64_t batchSize,
    int64_t partialHeads, int64_t headSize, cudaStream_t stream);

}

}
