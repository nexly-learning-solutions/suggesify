#pragma once

#include "checkMacrosPlugin.h"
#include "../common/cublasMMWrapper.h"
#include "../common/logger.h"
#include "../common/quantization.h"
#include "../common/stringUtils.h"
#include "../src/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "../src/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "../src/gptKernels.h"
#include "../plugins/common/plugin.h"
#include "../plugins/gptAttentionCommon/gptAttentionCommon.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace suggestify::plugins
{


class GPTAttentionPlugin : public GPTAttentionPluginCommon
{
public:
    GPTAttentionPlugin(int layer_idx, int num_heads, int vision_start, int vision_length, int num_kv_heads,
        int layer_idx_in_cache_pool, int head_size, int unidirectional, float q_scaling,
        float attn_logit_softcapping_scale, suggestify::kernels::PositionEmbeddingType position_embedding_type,
        int rotary_embedding_dim,
        float rotary_embedding_base, suggestify::kernels::RotaryScalingType rotary_embedding_scale_type,
        float rotary_embedding_scale, float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale,
        int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, int tp_size,
        int tp_rank,
        bool unfuse_qkv_gemm,
        bool use_logn_scaling,
        suggestify::kernels::ContextFMHAType context_fmha_type, int kv_cache_quant_mode, bool remove_input_padding,
        suggestify::kernels::AttentionMaskType mask_type,
        suggestify::kernels::BlockSparseParams block_sparse_params, bool paged_kv_cache, int tokens_per_block,
        nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled, bool cross_attention = false,
        int max_distance = 0, bool pos_shift_enabled = false, bool dense_context_fmha = false,
        bool use_paged_context_fmha = false, bool use_fp8_context_fmha = false, bool has_full_attention_mask = false,
        bool use_cache = true, bool is_spec_decoding_enabled = false,
        bool spec_decoding_is_generation_length_variable = false, int spec_decoding_max_generation_length = 1,
        bool is_mla_enabled = false, int q_lora_rank = 0, int kv_lora_rank = 0, int qk_nope_head_dim = 0,
        int qk_rope_head_dim = 0, int v_head_dim = 0, bool skip_attn = false, int cp_size = 1, int cp_rank = 0,
        std::set<int32_t> cp_group = {});

    GPTAttentionPlugin(void const* data, size_t length);

    ~GPTAttentionPlugin() override = default;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    template <typename T, typename AttentionOutT, typename KVCacheBuffer>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    template <typename T, typename AttentionOutT = T>
    int enqueueDispatchKVCacheType(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    void configurePluginImpl(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept;
    template <typename T>
    void configurePluginDispatchKVCacheType(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    GPTAttentionPlugin* clone() const noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

private:
    template <typename T, typename AttentionOutT>
    kernels::mlaParams<T> enqueueMLAPreprocess(int32_t localNbSeq, int32_t localNbTokens,
        nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void*& workspace, bool is_context, cudaStream_t stream);

    template <typename T, typename AttentionOutT, typename KVCacheBuffer>
    int enqueueSome(int32_t seqIdxBeg, int32_t localNbSeq, int32_t tokenIdxBeg, int32_t localNbTokens,
        nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    using IndexType = std::int32_t;

    std::vector<size_t> mEntryIdx;
    enum class IdxEntry : size_t
    {
        QKV_TENSOR,
        K_TENSOR,
        V_TENSOR,
        ATTENTION_MASK,
        ATTENTION_PACKED_MASK,
        SEQUENCE_LENGTH,
        HOST_PAST_KEY_VALUE_LENGTHS,
        HOST_MAX_ATTENTION_WINDOW,
        HOST_SINK_TOKEN_LENGTH,
        CONTEXT_LENGTHS,
        CACHE_INDIR,
        REQUEST_TYPES,
        KV_CACHE_BLOCK_OFFSETS,
        HOST_KV_CACHE_BLOCK_OFFSETS,
        HOST_KV_CACHE_POOL_POINTERS,
        HOST_KV_CACHE_POOL_MAPPING,
        PAST_KEY_VALUE,
        KV_CACHE_QUANTIZATION_SCALE,
        KV_CACHE_DEQUANTIZATION_SCALE,
        ATTENTION_OUTPUT_QUANTIZATION_SCALE,
        ROTARY_INV_FREQ,
        ROTARY_COS_SIN,
        ALIBI_SLOPES,
        RELATIVE_ATTENTION_BIAS,
        CROSS_KV,
        CROSS_KV_LENGTH,
        ENCODER_INPUT_LENGTH,
        HOST_CONTEXT_LENGTH,
        QKV_BIAS_TENSOR,
        SPEC_DECODING_GENERATION_LENGTHS,
        SPEC_DECODING_PACKED_MASK,
        SPEC_DECODING_POSITION_OFFSETS,
        SPEC_DECODING_USE,
        LONG_ROPE_ROTARY_INV_FREQ,
        LONG_ROPE_ROTARY_COS_SIN,
        MROPE_ROTARY_COS_SIN,
        MROPE_POSITION_DELTAS,
        HOST_RUNTIME_PERF_KNOBS,
        HOST_CONTEXT_PROGRESS,
        MLA_FUSED_Q_PROJ_TENSOR,
        MLA_Q_B_PROJ_TENSOR,
        MLA_KV_B_PROJ_TENSOR,
        SKIP_ATTN,
        LOGN_SCALING,
        ENUM_SIZE,
    };

    std::string toString(IdxEntry const& entry) const;
    bool isEntryUsed(IdxEntry const& entry) const;
    void initEntryIdx();
    IndexType getIdx(IdxEntry const& entry) const;

    int getGenerationInputSequenceLength(
        nvinfer1::PluginTensorDesc const* inputDesc, int32_t localNbSeq, int32_t localNbTokens) const;
};

class GPTAttentionPluginCreator : public GPTAttentionPluginCreatorCommon
{
public:
    GPTAttentionPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

}
