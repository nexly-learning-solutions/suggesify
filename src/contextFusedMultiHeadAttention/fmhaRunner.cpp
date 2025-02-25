
#include "fmhaRunner.h"
#include "../common/mathUtils.h"
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>


namespace suggestify
{
namespace kernels
{


union __half2_uint32_t_union
{
    half2 fp162;
    uint32_t u32;
};

union __float_uint32_t_union
{
    float fp32;
    uint32_t u32;
};

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_FP32)
    {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<uint32_t const&>(inorm);
    }
    else if (dtype == DATA_TYPE_BF16)
    {
        alpha = reinterpret_cast<uint32_t const&>(norm);
    }
    else
    {
        assert(false);
    }
}


FusedMHARunnerV2::FusedMHARunnerV2(MHARunnerFixedParams fixedParams)
    : mFixedParams(fixedParams)
{
    TLLM_CHECK_WITH_INFO(
        (mSM == kSM_80 || mSM == kSM_86 || mSM == kSM_89 || mSM == kSM_90), "Unsupported architecture");
    TLLM_CHECK_WITH_INFO((mFixedParams.dataType == DATA_TYPE_FP16 || mFixedParams.dataType == DATA_TYPE_BF16
                             || mFixedParams.dataType == DATA_TYPE_E4M3),
        "Unsupported data type");
    xmmaKernel = getXMMAKernelsV2(mFixedParams.dataType, mSM);

    if (mFixedParams.headSizeV == 0)
    {
        mFixedParams.headSizeV = mFixedParams.headSize;
    }
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, device_id);
    cudaDeviceGetAttribute(&mDeviceL2CacheSize, cudaDevAttrL2CacheSize, device_id);
    auto const [free_memory, total_memory] = suggestify::common::getDeviceMemoryInfo(false);
    mTotalDeviceMemory = total_memory;
}


void FusedMHARunnerV2::setupKernelParams(MHARunnerParams runnerParams)
{
    memset(&mKernelParams, 0, sizeof(mKernelParams));

    mKernelParams.b = runnerParams.b;
    mKernelParams.s = runnerParams.qSeqLen;
    mKernelParams.sliding_window_size = runnerParams.slidingWindowSize;
    mKernelParams.d = mFixedParams.headSize;
    mKernelParams.dv = mFixedParams.headSizeV;
    TLLM_CHECK_WITH_INFO(mFixedParams.numQHeads % mFixedParams.numKvHeads == 0,
        "number of Query heads should be multiple of KV heads !");
    mKernelParams.h = mFixedParams.numQHeads;
    mKernelParams.h_kv = mFixedParams.numKvHeads;
    mKernelParams.h_q_per_kv = mFixedParams.numQHeads / mFixedParams.numKvHeads;
    mKernelParams.is_s_padded = mFixedParams.isSPadded;

    mKernelParams.qkv_stride_in_bytes = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSize
            + mFixedParams.numKvHeads * mFixedParams.headSize + mFixedParams.numKvHeads * mFixedParams.headSizeV,
        mFixedParams.dataType);
    mKernelParams.q_stride_in_bytes
        = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSize, mFixedParams.dataType);
    if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
    {
        mKernelParams.kv_stride_in_bytes = get_size_in_bytes(
            runnerParams.pagedKvCache.mTokensPerBlock * mFixedParams.headSize, mFixedParams.dataType);
        mKernelParams.v_stride_in_bytes = mKernelParams.kv_stride_in_bytes;
    }
    else if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_CONTIGUOUS_KV)
    {
        mKernelParams.kv_stride_in_bytes = get_size_in_bytes(mFixedParams.headSize, mFixedParams.dataType);
    }
    mKernelParams.o_stride_in_bytes
        = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSizeV, mFixedParams.dataType);
    if (mFixedParams.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
    {
        mKernelParams.packed_mask_stride_in_bytes
            = (suggestify::common::divUp(int64_t(runnerParams.kvSeqLen), int64_t(FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT))
                  * FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT)
            / 8;
    }

    float const inv_sqrt_scale = (1.f / (sqrtf(mFixedParams.headSize) * mFixedParams.qScaling));
    float const scale_after_alibi = mFixedParams.scaleAlibi ? inv_sqrt_scale : 1.0f;
    float scale_bmm1 = mFixedParams.scaleAlibi ? 1.0f : inv_sqrt_scale;
    scale_bmm1 = mFixedParams.attnLogitSoftcappingScale != 0.f ? scale_bmm1 / mFixedParams.attnLogitSoftcappingScale
                                                               : scale_bmm1;
    float const scale_softmax = 1.f;
    float const scale_bmm2 = 1.f;

    Data_type scale_type = mLaunchParams.force_fp32_acc ? DATA_TYPE_FP32 : mFixedParams.dataType;
    if (mLaunchParams.useBase2ExpTrick)
    {
        constexpr float kLog2e = 1.4426950408889634074;
        set_alpha(mKernelParams.scale_bmm1, scale_bmm1 * float(kLog2e), DATA_TYPE_FP32);
    }
    else
    {
        set_alpha(mKernelParams.scale_bmm1, scale_bmm1, scale_type);
    }
    set_alpha(mKernelParams.scale_softmax, scale_softmax, scale_type);
    set_alpha(mKernelParams.scale_bmm2, scale_bmm2, scale_type);
    mKernelParams.softcapping_scale_bmm1 = mFixedParams.attnLogitSoftcappingScale;

    if (mFixedParams.hasAlibi && mSM > kSM_70)
    {
        mKernelParams.has_alibi = true;
        mKernelParams.alibi_params = AlibiParams(
            mFixedParams.numQHeads, runnerParams.kvSeqLen, mFixedParams.tpSize, mFixedParams.tpRank, scale_after_alibi);
    }

    mKernelParams.qkv_ptr = runnerParams.qkvPtr;
    mKernelParams.q_ptr = runnerParams.qPtr;
    mKernelParams.kv_ptr = runnerParams.kvPtr;
    mKernelParams.o_ptr = runnerParams.outputPtr;
    if (mFixedParams.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
    {
        mKernelParams.packed_mask_ptr = runnerParams.packedMaskPtr;
        mKernelParams.cu_mask_rows = reinterpret_cast<int const*>(runnerParams.cuMaskRowsPtr);
    }
    mKernelParams.cu_q_seqlens = reinterpret_cast<int const*>(runnerParams.cuQSeqLenPtr);
    mKernelParams.tile_id_counter_ptr = reinterpret_cast<uint32_t*>(runnerParams.tileCounterPtr);
    int64_t scaleBmm1PtrOffset = (mLaunchParams.useBase2ExpTrick ? 1 : 0);
    if (mFixedParams.dataType == DATA_TYPE_E4M3)
    {
        mKernelParams.scale_bmm1_d = reinterpret_cast<uint32_t const*>(runnerParams.scaleBmm1Ptr + scaleBmm1PtrOffset);
        mKernelParams.scale_bmm2_d = reinterpret_cast<uint32_t const*>(runnerParams.scaleBmm2Ptr);
    }

    if (mFixedParams.attentionInputLayout != AttentionInputLayout::PACKED_QKV)
    {
        mKernelParams.cu_kv_seqlens = reinterpret_cast<int const*>(runnerParams.cuKvSeqLenPtr);
    }

    if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
    {
        mKernelParams.paged_kv_cache = runnerParams.pagedKvCache.copyKVBlockArrayForContextFMHA();
    }
}


void FusedMHARunnerV2::setupLaunchParams(MHARunnerParams runnerParams)
{

    mLaunchParams = {};

    mLaunchParams.multi_processor_count = mMultiProcessorCount;
    mLaunchParams.device_l2_cache_size = mDeviceL2CacheSize;
    mLaunchParams.total_device_memory = mTotalDeviceMemory;

    TLLM_CHECK_WITH_INFO(
        (mFixedParams.headSize == 128 || mFixedParams.headSize == 256) || !mFixedParams.attnLogitSoftcappingScale,
        "FMHA only supports head_size = 128 or 256 with attention logit softcapping scale currently.");
    mLaunchParams.enableAttnLogitSoftcapping = mFixedParams.attnLogitSoftcappingScale != 0.f;
    mLaunchParams.force_fp32_acc = mFixedParams.dataType == DATA_TYPE_BF16 || mFixedParams.dataType == DATA_TYPE_E4M3
        || mFixedParams.forceFp32Acc || runnerParams.forceFp32Acc;
    mLaunchParams.attention_mask_type = mFixedParams.attentionMaskType;
    mLaunchParams.attention_input_layout = mFixedParams.attentionInputLayout;

    mLaunchParams.total_q_seqlen
        = mFixedParams.isSPadded ? runnerParams.b * runnerParams.qSeqLen : runnerParams.totalQSeqLen;
    mLaunchParams.total_kv_seqlen
        = mFixedParams.isSPadded ? runnerParams.b * runnerParams.kvSeqLen : runnerParams.totalKvSeqLen;

    TLLM_CHECK_WITH_INFO(mFixedParams.headSize > 0, "Head size should be greater than 0.");
    mLaunchParams.padded_d = (mFixedParams.headSize & (mFixedParams.headSize - 1)) == 0
        ? mFixedParams.headSize
        : pow(2, int(log2(mFixedParams.headSize)) + 1);

    bool const isSm70 = (mSM == kSM_70);
    bool const isSm90 = (mSM == kSM_90);
    bool const isSm8x = (mSM == kSM_86 || mSM == kSM_89);
    bool const isSm80 = (mSM == kSM_80);
    bool const isSm89 = (mSM == kSM_89);

    if (runnerParams.kvSeqLen > runnerParams.slidingWindowSize
        && mLaunchParams.attention_mask_type == ContextAttentionMaskType::CAUSAL)
    {
        mLaunchParams.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
    }

    bool const separateQKvInput = mFixedParams.attentionInputLayout != AttentionInputLayout::PACKED_QKV;
    bool const paddingOrCausalMask = mFixedParams.attentionMaskType == ContextAttentionMaskType::PADDING
        || mFixedParams.attentionMaskType == ContextAttentionMaskType::CAUSAL;

    if (isSm90 && (mFixedParams.dataType == DATA_TYPE_E4M3 || (separateQKvInput && runnerParams.kvSeqLen > 512)))
    {
        mLaunchParams.flash_attention = true;
        mLaunchParams.force_unroll = true;
    }
    else if (isSm70)
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported architecture");
    }
    else if (isSm90 && !separateQKvInput && paddingOrCausalMask
        && (mFixedParams.headSize == 32 || mFixedParams.headSize == 64) && runnerParams.qSeqLen <= 256)
    {
        mLaunchParams.flash_attention = false;
        mLaunchParams.kernel_s = getSFromMaxSeqLen(runnerParams.qSeqLen);
    }
    else
    {
        mLaunchParams.flash_attention = true;
        mLaunchParams.kernel_s = 0;
        mLaunchParams.force_unroll = true;
        if (isSm89 && mFixedParams.dataType == DATA_TYPE_E4M3)
        {
            mLaunchParams.granular_tiling = false;
        }
        else if (mLaunchParams.flash_attention && runnerParams.kvSeqLen <= 64)
        {
            mLaunchParams.granular_tiling = false;
        }
        else if (isSm8x && mFixedParams.headSize < 256)
        {
            mLaunchParams.granular_tiling = false;
        }
        else if (isSm80 || isSm8x)
        {
            mLaunchParams.granular_tiling = true;
        }
    }

    if (isSm90 && mLaunchParams.flash_attention)
    {
        mLaunchParams.warp_specialization = true;
        mLaunchParams.use_tma = true;
        mLaunchParams.dynamic_scheduler = true;
    }

    if (mLaunchParams.warp_specialization && !mFixedParams.hasAlibi)
    {
        mLaunchParams.useKernelWithoutAlibi = true;
        mLaunchParams.useBase2ExpTrick = !mLaunchParams.enableAttnLogitSoftcapping;
    }

    if (mFixedParams.headSize == mFixedParams.headSizeV + 64)
    {
        mLaunchParams.flash_attention = true;
        mLaunchParams.force_unroll = true;
        mLaunchParams.kernel_s = 0;
        mLaunchParams.granular_tiling = true;
        mLaunchParams.warp_specialization = false;
        mLaunchParams.useKernelWithoutAlibi = false;
        mLaunchParams.useBase2ExpTrick = false;
        mLaunchParams.use_tma = false;
        mLaunchParams.dynamic_scheduler = false;
    }
}


void FusedMHARunnerV2::setPackedQkvTmaDescriptors(MHARunnerParams runnerParams)
{
    uint32_t const d_in_bytes = get_size_in_bytes(mLaunchParams.padded_d, mFixedParams.dataType);
    uint32_t const d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

    Multiple_tma_descriptor<4> qkv_tma_descriptor;

    uint32_t tensor_size_qkv[4];
    if (mKernelParams.h_kv < mKernelParams.h)
    {
        tensor_size_qkv[2] = 1;
        tensor_size_qkv[1] = (mKernelParams.h + 2 * mKernelParams.h_kv);
        tensor_size_qkv[0] = mKernelParams.d;
    }
    else
    {
        tensor_size_qkv[2] = 3;
        tensor_size_qkv[1] = mKernelParams.h;
        tensor_size_qkv[0] = mKernelParams.d;
    }

    uint32_t tensor_size_o[4];
    tensor_size_o[0] = mKernelParams.d;
    tensor_size_o[1] = mKernelParams.h;
    tensor_size_o[2] = 1;

    uint32_t box_size[4];
    box_size[2] = 1;
    box_size[1] = 1;
    box_size[0] = mLaunchParams.padded_d / d_groups;

    uint64_t tensor_stride_qkv[3];
    tensor_stride_qkv[0] = get_size_in_bytes(tensor_size_qkv[0], mFixedParams.dataType);
    tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0];
    tensor_stride_qkv[2] = tensor_size_qkv[2] * tensor_stride_qkv[1];

    uint64_t tensor_stride_o[3];
    tensor_stride_o[0] = get_size_in_bytes(tensor_size_o[0], mFixedParams.dataType);
    tensor_stride_o[1] = tensor_size_o[1] * tensor_stride_o[0];
    tensor_stride_o[2] = tensor_size_o[2] * tensor_stride_o[1];

    uint32_t traversal_stride_qkv[4] = {1, 1, 1, 1};
    uint32_t traversal_stride_o[4] = {1, 1, 1, 1};

    uint32_t oob_fill = 0;

    uint32_t fp32_to_tf32 = 0;

    uint32_t const d_bytes_per_group = d_in_bytes / d_groups;
    cudaTmaDescSwizzle const swizzle_mode = (d_bytes_per_group > 64
            ? cudaTmaDescSwizzle::SWIZZLE_128B
            : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

    uint32_t q_step = 0, kv_step = 0;
    xmmaKernel->getStepSize(q_step, kv_step, mKernelParams, mLaunchParams);

    auto const* qkv_ptr = static_cast<char const*>(mKernelParams.qkv_ptr);
    tensor_size_qkv[3] = mLaunchParams.total_q_seqlen;
    auto* o_ptr = static_cast<char*>(mKernelParams.o_ptr);
    tensor_size_o[3] = mLaunchParams.total_q_seqlen;

    box_size[3] = q_step;
    cudaTmaDescFormat const desc_format
        = (get_size_in_bytes(mFixedParams.dataType) == 1) ? cudaTmaDescFormat::U8 : cudaTmaDescFormat::F16_RN;
    qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
        swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_stride_qkv,
        traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_q);

    box_size[3] = kv_step;
    qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
        swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_stride_qkv,
        traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_kv);

    box_size[3] = 16;
    if ((get_size_in_bytes(mFixedParams.dataType) == 1)
        && mLaunchParams.attention_mask_type != ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL)
    {
        qkv_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_o, tensor_stride_o, traversal_stride_o,
            box_size, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_o);
    }
}


void FusedMHARunnerV2::setSeparateQKvTmaDescriptors(MHARunnerParams runnerParams)
{
    uint32_t const d_in_bytes = get_size_in_bytes(mLaunchParams.padded_d, mFixedParams.dataType);
    uint32_t const d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

    uint32_t q_step = 0, kv_step = 0;
    xmmaKernel->getStepSize(q_step, kv_step, mKernelParams, mLaunchParams);

    Multiple_tma_descriptor<4> qo_tma_descriptor;
    Multiple_tma_descriptor<4> kv_tma_descriptor;
    uint32_t tensor_size_qo[4];
    tensor_size_qo[3] = mLaunchParams.total_q_seqlen;
    tensor_size_qo[2] = 1;
    tensor_size_qo[1] = mKernelParams.h;
    tensor_size_qo[0] = mKernelParams.d;

    uint32_t box_size_qo[4];
    box_size_qo[3] = q_step;
    box_size_qo[2] = 1;
    box_size_qo[1] = 1;
    box_size_qo[0] = mLaunchParams.padded_d / d_groups;

    uint64_t tensor_stride_qo[3];
    tensor_stride_qo[0] = get_size_in_bytes(tensor_size_qo[0], mFixedParams.dataType);
    tensor_stride_qo[1] = tensor_size_qo[1] * tensor_stride_qo[0];
    tensor_stride_qo[2] = tensor_size_qo[2] * tensor_stride_qo[1];

    uint32_t traversal_stride[4] = {1, 1, 1, 1};

    uint32_t oob_fill = 0;

    uint32_t fp32_to_tf32 = 0;

    cudaTmaDescFormat const desc_format
        = (get_size_in_bytes(mFixedParams.dataType) == 1) ? cudaTmaDescFormat::U8 : cudaTmaDescFormat::F16_RN;

    uint32_t const d_bytes_per_group = d_in_bytes / d_groups;
    cudaTmaDescSwizzle const swizzle_mode = (d_bytes_per_group > 64
            ? cudaTmaDescSwizzle::SWIZZLE_128B
            : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

    auto const* q_ptr = static_cast<char const*>(mKernelParams.q_ptr);

    qo_tma_descriptor.set_tma_desctriptor(q_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
        cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo, traversal_stride, box_size_qo,
        oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_q);

    auto const* o_ptr = static_cast<char const*>(mKernelParams.o_ptr);

    box_size_qo[3] = 16;
    if ((get_size_in_bytes(mFixedParams.dataType) == 1)
        && mLaunchParams.attention_mask_type != ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL)
    {
        qo_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo, traversal_stride,
            box_size_qo, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_o);
    }

    if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_CONTIGUOUS_KV)
    {
        uint32_t tensor_size_kv[4];
        tensor_size_kv[3] = mLaunchParams.total_kv_seqlen;
        tensor_size_kv[2] = 2;
        tensor_size_kv[1] = mKernelParams.h_kv;
        tensor_size_kv[0] = mKernelParams.d;

        uint32_t box_size_kv[4];
        box_size_kv[3] = kv_step;
        box_size_kv[2] = 1;
        box_size_kv[1] = 1;
        box_size_kv[0] = mLaunchParams.padded_d / d_groups;

        uint64_t tensor_stride_kv[3];
        tensor_stride_kv[0] = get_size_in_bytes(tensor_size_kv[0], mFixedParams.dataType);
        tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];
        tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];

        kv_tma_descriptor.set_tma_desctriptor(runnerParams.kvPtr, desc_format,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_kv, tensor_stride_kv, traversal_stride, box_size_kv, oob_fill, fp32_to_tf32,
            &mKernelParams.tma_desc_kv);
    }
    else if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
    {
        uint32_t tokens_per_block = uint32_t(mKernelParams.paged_kv_cache.mTokensPerBlock);
        uint32_t tensor_size_kv[4];
        tensor_size_kv[3] = mLaunchParams.total_device_memory / mKernelParams.paged_kv_cache.mBytesPerBlock;
        tensor_size_kv[2] = mKernelParams.h_kv;
        tensor_size_kv[1] = tokens_per_block;
        tensor_size_kv[0] = mKernelParams.d;

        uint32_t box_size_kv[4];
        box_size_kv[3] = 1;
        box_size_kv[2] = 1;
        box_size_kv[1] = std::min(tokens_per_block, kv_step);
        box_size_kv[0] = mLaunchParams.padded_d / d_groups;

        TLLM_CHECK_WITH_INFO(
            tokens_per_block % 2 == 0, "FMHA with paged kv cache needs tokens_per_block to be power of 2 !");
        mKernelParams.blocks_per_tma_load = std::max(1, int32_t(kv_step / tokens_per_block));
        mKernelParams.blocks_per_tma_load_log2 = log2(mKernelParams.blocks_per_tma_load);

        uint64_t tensor_stride_kv[3];
        tensor_stride_kv[0] = get_size_in_bytes(tensor_size_kv[0], mFixedParams.dataType);
        tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];
        tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];

        kv_tma_descriptor.set_tma_desctriptor(runnerParams.pagedKvCache.mPrimaryPoolPtr, desc_format,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_kv, tensor_stride_kv, traversal_stride, box_size_kv, oob_fill, fp32_to_tf32,
            &mKernelParams.tma_desc_kv);
    }
}

void FusedMHARunnerV2::run(MHARunnerParams runnerParams)
{
    setupLaunchParams(runnerParams);
    setupKernelParams(runnerParams);
    if (mSM == kSM_90 && mLaunchParams.use_tma)
    {
        switch (mFixedParams.attentionInputLayout)
        {
        case AttentionInputLayout::PACKED_QKV: setPackedQkvTmaDescriptors(runnerParams); break;
        case AttentionInputLayout::Q_CONTIGUOUS_KV:
        case AttentionInputLayout::Q_PAGED_KV: setSeparateQKvTmaDescriptors(runnerParams); break;
        default: TLLM_CHECK_WITH_INFO(false, "Unsupported attention input layout.");
        }
    }
    if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV
        && mLaunchParams.attention_mask_type == ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL)
    {
        uint32_t q_step = 0, kv_step = 0;
        xmmaKernel->getStepSize(q_step, kv_step, mKernelParams, mLaunchParams);
        TLLM_CHECK_WITH_INFO(mKernelParams.sliding_window_size % kv_step == 0,
            "The sliding window size doesn't work with paged context fmha kv_step_size = %d.", kv_step);
    }

    xmmaKernel->run(mKernelParams, mLaunchParams, runnerParams.stream);
}


bool FusedMHARunnerV2::isValidS(int s) const
{
    return xmmaKernel->isValid(s);
}


int FusedMHARunnerV2::getSFromMaxSeqLen(int const max_seq_len) const
{
    int S = 1024;

    if (max_seq_len <= 64)
    {
        S = 64;
    }
    else if (max_seq_len <= 128)
    {
        S = 128;
    }
    else if (max_seq_len <= 256)
    {
        S = 256;
    }
    else if (max_seq_len <= 384)
    {
        S = 384;
    }
    else if (max_seq_len <= 512)
    {
        S = 512;
    }
    else if (max_seq_len > 512)
    {
        S = max_seq_len;
    }

    return S;
}


bool FusedMHARunnerV2::isFmhaSupported()
{
    bool foundKernels = xmmaKernel->checkIfKernelExist(mFixedParams);

    if (!foundKernels)
    {
        TLLM_LOG_WARNING("Fall back to unfused MHA for %s in sm_%d.", mFixedParams.convertToStrOutput().c_str(), mSM);
    }

    return foundKernels;
}

}
}
