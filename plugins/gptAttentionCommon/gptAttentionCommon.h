#pragma once

#include "suggestify/common/cublasMMWrapper.h"
#include "suggestify/common/quantization.h"
#include "../src/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "../src/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "../src/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "../src/gptKernels.h"
#include "../src/kvCacheUtils.h"
#include "../src/mlaKernels.h"
#include "../plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace suggestify::plugins
{

class GPTAttentionPluginCommon : public BasePlugin
{
public:
    GPTAttentionPluginCommon() = delete;

    GPTAttentionPluginCommon(int layer_idx, int num_heads, int vision_start, int vision_length, int num_kv_heads,
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
        bool spec_decoding_is_generation_length_variable = false, int32_t spec_decoding_max_generation_length = 1,
        bool is_mla_enabled = false, int q_lora_rank = 0, int kv_lora_rank = 0, int qk_nope_head_dim = 0,
        int qk_rope_head_dim = 0, int v_head_dim = 0, bool skip_attn = false, int cp_size = 1, int cp_rank = 0,
        std::set<int32_t> cp_group = {});

    GPTAttentionPluginCommon(void const* data, size_t length);

    ~GPTAttentionPluginCommon() override = default;

    template <typename T>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    int initialize() noexcept override;
    void terminate() noexcept override;

    template <typename T>
    T* cloneImpl() const noexcept;

    void destroy() noexcept override;

    size_t getCommonSerializationSize() const noexcept;
    void serializeCommon(void* buffer) const noexcept;
    int getHeadSize(bool checkInit = true) const;

protected:
    int getMaxNumSeqLenTile(int batch_beam_size = 1) const;
    size_t getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t nbReq, int32_t max_input_length,
        int32_t cross_kv_length = 0, int32_t max_num_tokens = 0) const noexcept;
    size_t getWorkspaceSizeForGeneration(nvinfer1::DataType type, int32_t total_num_seq, int32_t max_kv_cache_length,
        int32_t max_num_tokens) const noexcept;

    size_t getWorkspaceSizeForMLAPreProcess(
        nvinfer1::DataType type, size_t& remaining_size, int32_t total_token_length, int32_t rope_dim) const noexcept;

    template <typename T>
    struct EnqueueContextParams
    {
        T const* attention_input;
        T const* qkv_bias;
        bool const* attention_mask;
        uint32_t const* attention_packed_mask;
        float const* rotary_inv_freq;
        float2 const* rotary_cos_sin;
        int32_t input_seq_length;
        int32_t max_past_kv_len;
        int32_t max_attention_window;
        int32_t cyclic_attention_window_size;
        int32_t max_cyclic_attention_window_size;
        bool can_use_one_more_block;
        int32_t sink_token_length;
        int32_t const* q_seq_lengths;
        int32_t const* kv_seq_lengths;
        float const* kv_scale_orig_quant;
        float const* kv_scale_quant_orig;
        float const* attention_output_orig_quant;
        T const* alibi_slopes;
        void* context_buf;
        void* key_value_cache;
        kernels::KVBlockArray::DataType* block_offsets;
        kernels::KVBlockArray::DataType* host_block_offsets;
        void* host_primary_pool_pointer;
        void* host_secondary_pool_pointer;
        int32_t batch_size;
        int32_t num_tokens;
        int32_t max_blocks_per_sequence;
        int32_t const* host_context_lengths;
        void* workspace;
        float2 const* mrope_rotary_cos_sin = nullptr;

        float const* logn_scaling_ptr = nullptr;
        T const* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        T const* cross_kv = nullptr;
        int32_t cross_kv_length = 0;
        int32_t const* encoder_input_lengths = nullptr;
        int32_t num_encoder_tokens = 0;
        int64_t const* runtime_perf_knobs = nullptr;
        kernels::mlaParams<T>* mla_param;

        std::string enqueueContextParamsToString() const
        {
            std::stringstream ss;
            ss << "EnqueueContextParams ====================" << std::endl;

            ss << "attention_input: " << attention_input << std::endl;
            ss << "qkv_bias: " << qkv_bias << std::endl;
            ss << "attention_mask: " << attention_mask << std::endl;
            ss << "attention_packed_mask: " << attention_packed_mask << std::endl;
            ss << "rotary_inv_freq: " << rotary_inv_freq << std::endl;
            ss << "rotary_cos_sin: " << rotary_cos_sin << std::endl;
            ss << "input_seq_length: " << input_seq_length << std::endl;
            ss << "max_past_kv_len: " << max_past_kv_len << std::endl;
            ss << "max_attention_window: " << max_attention_window << std::endl;
            ss << "cyclic_attention_window_size: " << cyclic_attention_window_size << std::endl;
            ss << "max_cyclic_attention_window_size: " << max_cyclic_attention_window_size << std::endl;
            ss << "can_use_one_more_block: " << (can_use_one_more_block ? "true" : "false") << std::endl;
            ss << "sink_token_length: " << sink_token_length << std::endl;
            ss << "q_seq_lengths: "
               << *(runtime::ITensor::wrap(
                      (void*) q_seq_lengths, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})))
               << std::endl;
            ss << "kv_seq_lengths: "
               << *(runtime::ITensor::wrap(
                      (void*) kv_seq_lengths, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})))
               << std::endl;
            ss << "kv_scale_orig_quant: " << kv_scale_orig_quant << std::endl;
            ss << "kv_scale_quant_orig: " << kv_scale_quant_orig << std::endl;
            ss << "attention_output_orig_quant: " << attention_output_orig_quant << std::endl;
            ss << "alibi_slopes: " << alibi_slopes << std::endl;
            ss << "context_buf: " << context_buf << std::endl;
            ss << "key_value_cache: " << (half*) key_value_cache << std::endl;
            ss << "block_offsets: " << block_offsets << std::endl;
            ss << "host_block_offsets: " << host_block_offsets << std::endl;
            ss << "host_primary_pool_pointer: " << host_primary_pool_pointer << std::endl;
            ss << "host_secondary_pool_pointer: " << host_secondary_pool_pointer << std::endl;
            ss << "batch_size: " << batch_size << std::endl;
            ss << "num_tokens: " << num_tokens << std::endl;
            ss << "max_blocks_per_sequence: " << max_blocks_per_sequence << std::endl;
            ss << "workspace: " << workspace << std::endl;
            ss << "logn_scaling_ptr: " << logn_scaling_ptr << std::endl;
            ss << "relative_attention_bias: " << relative_attention_bias << std::endl;
            ss << "relative_attention_bias_stride: " << relative_attention_bias_stride << std::endl;
            ss << "cross_kv: " << cross_kv << std::endl;
            ss << "cross_kv_length: " << cross_kv_length << std::endl;
            ss << "encoder_input_lengths: " << encoder_input_lengths << std::endl;
            ss << "num_encoder_tokens: " << num_encoder_tokens << std::endl;
            return ss.str();
        }
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(EnqueueContextParams<T> const& params, cudaStream_t stream);

    template <typename T>
    struct EnqueueGenerationParams
    {
        T const* attention_input;
        T const* qkv_bias;
        bool const* attention_mask;
        float const* rotary_inv_freq;
        int32_t input_seq_length;
        int32_t const* sequence_lengths;
        int32_t max_past_kv_length;
        int32_t beam_width;
        int32_t const* context_lengths;
        float const* kv_scale_orig_quant;
        float const* kv_scale_quant_orig;
        float const* attention_output_orig_quant;
        T const* alibi_slopes;
        void* context_buf;
        void* key_value_cache;
        kernels::KVBlockArray::DataType* block_offsets;
        void* host_primary_pool_pointer;
        void* host_secondary_pool_pointer;
        int32_t attention_mask_stride;
        int32_t max_attention_window;
        int32_t cyclic_attention_window_size;
        int32_t max_cyclic_attention_window_size;
        bool can_use_one_more_block;
        int32_t sink_token_length;
        int32_t num_requests;
        int32_t max_blocks_per_sequence;
        int32_t const* cache_indir;
        int32_t* semaphores;
        void* workspace;
        int32_t const* host_past_key_value_lengths;
        int32_t const* mrope_position_deltas = nullptr;

        float const* logn_scaling_ptr = nullptr;
        T const* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        int32_t const* encoder_input_lengths = nullptr;
        int32_t const* host_context_lengths = nullptr;
        bool const* spec_decoding_mask = nullptr;
        int32_t const* spec_decoding_packed_mask = nullptr;
        int32_t const* spec_decoding_position_offsets = nullptr;
        int32_t const* spec_decoding_generation_lengths = nullptr;
        bool spec_decoding_is_generation_length_variable = false;
        int32_t spec_decoding_max_generation_length = 1;
        int32_t total_num_input_tokens;
        int64_t const* runtime_perf_knobs = nullptr;
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(EnqueueGenerationParams<T> const& params, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    int mlaPreContext(
        kernels::mlaParams<T>& params, EnqueueContextParams<T> const& context_params, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    int mlaGeneration(
        kernels::mlaParams<T>& params, EnqueueGenerationParams<T> const& generation_params, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    void prepareEnqueueGeneration(EnqueueGenerationParams<T> const& params);

    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(suggestify::kernels::XQAParams& xqaParams,
        EnqueueGenerationParams<T> const& generationsParams, bool forConfigurePlugin);

    bool isRelativePosition() const
    {
        return mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kRELATIVE;
    }

    bool isALiBi() const
    {
        return mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kALIBI
            || mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    bool isAliBiWithScale() const
    {
        return mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    bool isRoPE() const
    {
        return mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kROPE_GPTJ
            || mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kROPE_GPT_NEOX
            || mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kLONG_ROPE
            || mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kROPE_M;
    }

    bool isLongRoPE() const
    {
        return mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kLONG_ROPE;
    }

    bool isUnfusedCrossAttention() const
    {
        return !mEnableContextFMHA && mCrossAttention;
    }

    bool isMRoPE() const
    {
        return mPositionEmbeddingType == suggestify::kernels::PositionEmbeddingType::kROPE_M;
    }

    bool isLognScaling() const
    {
        return mUseLognScaling;
    }

    bool isCrossAttention() const
    {
        return mCrossAttention;
    }

    bool useKVCache() const
    {
        return mUseKVCache;
    }

    bool useCustomMask() const
    {
        return mMaskType == suggestify::kernels::AttentionMaskType::CUSTOM_MASK;
    }

    bool useFullCustomMask() const
    {
        return useCustomMask() && mHasFullAttentionMask;
    }

    bool usePackedCustomMask() const
    {
        return useCustomMask() && mEnableContextFMHA;
    }

    void reserveSemaphoreArray(int32_t size);

    void debugCheckSemaphores(cudaStream_t stream);

protected:
    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;

    const std::string mLayerName;

    int mLayerIdx;
    int mNumHeads;
    int mVisionStart;
    int mVisionLength;
    int mNumKVHeads;
    int mLayerIdxInCachePool;
    int mHeadSize;
    int mUnidirectional;
    float mQScaling;
    float mAttnLogitSoftcappingScale;
    int mRotaryEmbeddingDim;
    float mRotaryEmbeddingBase;
    suggestify::kernels::RotaryScalingType mRotaryEmbeddingScaleType;
    float mRotaryEmbeddingScale;
    float mRotaryEmbeddingShortMscale;
    float mRotaryEmbeddingLongMscale;
    int mRotaryEmbeddingMaxPositions;
    int mRotaryEmbeddingOriginalMaxPositions;
    suggestify::kernels::PositionEmbeddingType mPositionEmbeddingType;
    bool mUseLognScaling = false;
    bool mRemovePadding = false;
    suggestify::kernels::AttentionMaskType mMaskType;
    suggestify::kernels::BlockSparseParams mBlockSparseParams;

    bool mPagedKVCache = false;
    int mTokensPerBlock = 0;
    suggestify::common::QuantMode mKVCacheQuantMode;
    int mTpSize = 1;
    int mTpRank = 0;
    bool mUnfuseQkvGemm = false;
    nvinfer1::DataType mType;
    int32_t mMaxContextLength;
    bool mQKVBiasEnabled;
    bool mCrossAttention = false;
    int mMaxDistance = 0;
    bool mPosShiftEnabled = false;
    bool mPagedContextFMHA = false;
    bool mFP8ContextFMHA = false;
    bool mDenseContextFMHA = false;
    bool mHasFullAttentionMask = false;
    bool mIsSpecDecodingEnabled = false;
    bool mUseSpecDecoding = false;
    bool mSpecDecodingIsGenerationLengthVariable = false;
    int32_t mSpecDecodingMaxGenerationLength = 1;
    bool mIsMLAEnabled = false;
    suggestify::kernels::mlaMetaParams mMLAParams;
    int mCpSize = 1;
    int mCpRank = 0;
    std::set<int32_t> mCpGroup = {};
#if ENABLE_MULTI_DEVICE
    std::shared_ptr<ncclComm_t> mCpNcclComm;
#endif

    uint4* mSpecDecodingPackedMask;
    uint4* mSpecDecodingPackedHostMask;

    bool mEnableContextFMHA = false;
    bool mFMHAForceFP32Acc = false;
    int mSM = suggestify::common::getSMVersion();
    int mMultiProcessorCount = suggestify::common::getMultiProcessorCount();
    int mMaxSharedMemoryPerBlockOptin = suggestify::common::getMaxSharedMemoryPerBlockOptin();
    std::shared_ptr<CUDADriverWrapper> mDriver;
    UniqPtrWNullCopy<suggestify::kernels::FusedMHARunnerV2> mFMHARunner;
    UniqPtrWNullCopy<suggestify::kernels::FusedMHARunnerV2> mDecoderFMHARunner;
    UniqPtrWNullCopy<suggestify::kernels::DecoderXQARunner> mDecoderXQARunner;

    bool mMultiBlockMode;
    bool mEnableXQA;
    int mDeviceId = -1;
    static bool mForceMultiBlockWarned;
    UniqPtrWNullCopy<suggestify::common::CublasMMWrapper> mCublasWrapper;
    bool mUseKVCache = true;

    int32_t mNbMultiBlockSemaphores = 0;
    bool mSkipAttn = false;

    struct Deleter
    {
        void operator()(void* ptr)
        {
            cudaFree(ptr);
        }
    };

    UniqPtrWNullCopy<int32_t[], Deleter> mMultiBlockSemaphores = {};

    std::string toString() const
    {
        std::stringstream ss;
        ss << "gptAttentionCommon members ====================" << std::endl;
        ss << "mNumHeads: " << mNumHeads << std::endl;
        ss << "mNumKVHeads: " << mNumKVHeads << std::endl;
        ss << "mLayerIdxInCachePool " << mLayerIdxInCachePool << std::endl;
        ss << "mHeadSize: " << mHeadSize << std::endl;
        ss << "mUnidirectional: " << mUnidirectional << std::endl;
        ss << "mQScaling: " << mQScaling << std::endl;
        ss << "mRotaryEmbeddingDim: " << mRotaryEmbeddingDim << std::endl;
        ss << "mRotaryEmbeddingBase: " << mRotaryEmbeddingBase << std::endl;
        ss << "mRotaryEmbeddingScaleType: " << static_cast<int>(mRotaryEmbeddingScaleType) << std::endl;
        ss << "mRotaryEmbeddingScale: " << mRotaryEmbeddingScale << std::endl;
        ss << "mRotaryEmbeddingMaxPositions: " << mRotaryEmbeddingMaxPositions << std::endl;
        ss << "mPositionEmbeddingType: " << static_cast<int>(mPositionEmbeddingType) << std::endl;
        ss << "mUseLognScaling: " << std::boolalpha << mUseLognScaling << std::endl;
        ss << "mRemovePadding: " << std::boolalpha << mRemovePadding << std::endl;
        ss << "mMaskType: " << static_cast<int>(mMaskType) << std::endl;
        ss << "mPagedKVCache: " << std::boolalpha << mPagedKVCache << std::endl;
        ss << "mTokensPerBlock: " << mTokensPerBlock << std::endl;
        ss << "mKVCacheQuantMode: " << static_cast<int>(mKVCacheQuantMode.value()) << std::endl;
        ss << "mTpSize: " << mTpSize << std::endl;
        ss << "mTpRank: " << mTpRank << std::endl;
        ss << "mUnfuseQkvGemm: " << std::boolalpha << mUnfuseQkvGemm << std::endl;
        ss << "mType: " << static_cast<int>(mType) << std::endl;
        ss << "mMaxContextLength: " << mMaxContextLength << std::endl;
        ss << "mQKVBiasEnabled: " << std::boolalpha << mQKVBiasEnabled << std::endl;
        ss << "mCrossAttention: " << std::boolalpha << mCrossAttention << std::endl;
        ss << "mMaxDistance: " << mMaxDistance << std::endl;
        ss << "mPosShiftEnabled: " << std::boolalpha << mPosShiftEnabled << std::endl;
        ss << "mPagedContextFMHA: " << std::boolalpha << mPagedContextFMHA << std::endl;
        ss << "mFP8ContextFMHA: " << std::boolalpha << mFP8ContextFMHA << std::endl;
        ss << "mDenseContextFMHA: " << std::boolalpha << mDenseContextFMHA << std::endl;
        ss << "mEnableContextFMHA: " << std::boolalpha << mEnableContextFMHA << std::endl;
        ss << "mFMHAForceFP32Acc: " << std::boolalpha << mFMHAForceFP32Acc << std::endl;
        ss << "mSM: " << mSM << std::endl;
        ss << "mMultiProcessorCount: " << mMultiProcessorCount << std::endl;
        ss << "mMaxSharedMemoryPerBlockOptin: " << mMaxSharedMemoryPerBlockOptin << std::endl;
        ss << "mMultiBlockMode: " << std::boolalpha << mMultiBlockMode << std::endl;
        ss << "mEnableXQA: " << std::boolalpha << mEnableXQA << std::endl;
        ss << "mDeviceId: " << mDeviceId << std::endl;
        ss << "mUseKVCache: " << std::boolalpha << mUseKVCache << std::endl;
        ss << "mForceMultiBlockWarned: " << mForceMultiBlockWarned << std::endl;
        ss << "mSkipAttn: " << std::boolalpha << mSkipAttn << std::endl;
        ss << "mCpSize: " << mCpSize << std::endl;
        ss << "mCpRank: " << mCpRank << std::endl;
        ss << "mCpGroup: [";
        for (auto it = mCpGroup.begin(); it != mCpGroup.end(); it++)
        {
            if (it != mCpGroup.begin())
            {
                ss << ", ";
            }
            ss << *it;
        }
        ss << "]" << std::endl;

        return ss.str();
    }
};

class GPTAttentionPluginCreatorCommon : public BaseCreator
{
public:
    GPTAttentionPluginCreatorCommon();

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    template <typename T>
    T* deserializePluginImpl(char const* name, void const* serialData, size_t serialLength) noexcept;

protected:
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC{};
};

}
