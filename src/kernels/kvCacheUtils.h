#pragma once

#include "../common/assert.h"
#include "../src/kvCacheIndex.h"

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

namespace suggestify::kernels
{

enum class KVIdxType : int32_t
{
    K_IDX = 0,
    V_IDX = 1
};

struct KVBlockArrayForContextFMHA
{
    using DataType = KVCacheIndex const;

    int32_t mMaxSeqs;
    int32_t mMaxBlocksPerSeq;
    int32_t mTokensPerBlock;
    int32_t mTokensPerBlockLog2;

    int32_t mBytesPerBlock;
    void* mPrimaryPoolPtr;
    DataType* data;

    KVBlockArrayForContextFMHA()
        : mMaxSeqs{0}
        , mMaxBlocksPerSeq{0}
        , mTokensPerBlock{0}
        , mTokensPerBlockLog2{0}
        , mBytesPerBlock{0}
        , mPrimaryPoolPtr{nullptr}
        , data{nullptr}
    {
    }

    KVBlockArrayForContextFMHA(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock,
        int32_t bytesPerToken, void* primaryPoolPtr, DataType* data)
        : mMaxSeqs(batchSize)
        , mMaxBlocksPerSeq(maxBlocksPerSeq)
        , mTokensPerBlock(tokensPerBlock)
        , mBytesPerBlock{tokensPerBlock * bytesPerToken}
        , mPrimaryPoolPtr{primaryPoolPtr}
        , data{data}
    {
        float const tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
        CHECK_WITH_INFO(
            ceil(tokensPerBlockSeqLog2) == floor(tokensPerBlockSeqLog2), "tokensPerBlock must be power of 2");
        CHECK_WITH_INFO(static_cast<int64_t>(mMaxSeqs - 1) * mMaxBlocksPerSeq * 2 + maxBlocksPerSeq
                <= std::numeric_limits<int32_t>::max(),
            "kv cache is too large for gpt_attention_plugin");
        mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
    }
};

struct KVBlockArray : public KVBlockArrayForContextFMHA
{
    void* mSecondaryPoolPtr;
    int32_t mMaxAttentionWindow;
    int32_t mSinkTokens;
    int32_t mCyclicCacheLen;
    int32_t mBubbleLen;
    bool mEnableOneMoreBlock;

    KVBlockArray()
        : mSecondaryPoolPtr(nullptr)
        , mMaxAttentionWindow{0}
        , mSinkTokens{0}
        , mCyclicCacheLen{0}
        , mBubbleLen{0}
        , mEnableOneMoreBlock{false}
    {
    }

    KVBlockArray(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t bytesPerToken,
        int32_t maxAttentionWindow, int32_t maxAttentionWindowAllLayer, int32_t sinkTokenLen, bool canUseOneMoreBlock,
        void* primaryPoolPtr, void* secondaryPoolPtr, DataType* data)
        : KVBlockArrayForContextFMHA(batchSize, maxBlocksPerSeq, tokensPerBlock, bytesPerToken, primaryPoolPtr, data)
        , mSecondaryPoolPtr{secondaryPoolPtr}
        , mMaxAttentionWindow(maxAttentionWindow)
        , mSinkTokens(sinkTokenLen)
    {
        auto sinkTokensInLastBlock = mSinkTokens % mTokensPerBlock;
        mBubbleLen = sinkTokensInLastBlock == 0 ? 0 : mTokensPerBlock - sinkTokensInLastBlock;
        mEnableOneMoreBlock = (maxBlocksPerSeq - 1) * tokensPerBlock >= maxAttentionWindowAllLayer + mBubbleLen;
        mEnableOneMoreBlock &= canUseOneMoreBlock;
        mCyclicCacheLen = (mEnableOneMoreBlock) ? mMaxAttentionWindow + mTokensPerBlock - mSinkTokens
                                                : mMaxAttentionWindow - mSinkTokens;
    }

    [[nodiscard]] KVBlockArrayForContextFMHA copyKVBlockArrayForContextFMHA() const
    {
        return KVBlockArrayForContextFMHA{
            mMaxSeqs, mMaxBlocksPerSeq, mTokensPerBlock, mBytesPerBlock / mTokensPerBlock, mPrimaryPoolPtr, data};
    }

    __host__ __device__ [[nodiscard]] inline bool isSinkToken(int32_t tokenIdx) const
    {
        return tokenIdx < mSinkTokens;
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVTokenIdx(int32_t tokenIdx) const
    {
        if (!isSinkToken(tokenIdx))
        {
            return mSinkTokens + mBubbleLen + (tokenIdx - mSinkTokens) % mCyclicCacheLen;
        }
        return tokenIdx;
    }

    __host__ __device__ [[nodiscard]] inline DataType const* getRowPtr(KVIdxType kvIdx, int32_t seqIdx) const
    {
        return data + (seqIdx * mMaxBlocksPerSeq * 2 + static_cast<int32_t>(kvIdx) * mMaxBlocksPerSeq);
    }

    __host__ __device__ inline void* getBlockPtr(DataType const* offsets, int32_t tokenIdx) const
    {
        auto const offset = offsets[tokenIdx >> mTokensPerBlockLog2];
        return reinterpret_cast<void*>(
            reinterpret_cast<char*>(getPoolPtr(offset)) + offset.get() * static_cast<uint64_t>(mBytesPerBlock));
    }

    __host__ __device__ [[nodiscard]] inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx) const
    {
        return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
    }

    __host__ __device__ [[nodiscard]] inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx) const
    {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
    }

    __host__ __device__ [[nodiscard]] inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx) const
    {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
    }

    __host__ __device__ [[nodiscard]] inline int32_t getLocalIdx(int32_t globalIdx) const
    {
        return globalIdx & ((1 << mTokensPerBlockLog2) - 1);
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVLocalIdx(
        int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const
    {
        return headIdx * mTokensPerBlock * dimsPerHead + getLocalIdx(globalTokenIdx) * dimsPerHead + channelIdx;
    }

private:
    __host__ __device__ [[nodiscard]] void* getPoolPtr(DataType offset) const
    {
        return offset.isPrimary() ? mPrimaryPoolPtr : mSecondaryPoolPtr;
    }
};

struct KVLinearBuffer
{
    using DataType = int8_t;

    int32_t mMaxSeqs;
    int32_t mMaxSeqLen;
    int32_t mBytesPerSeq;
    int32_t mMaxAttentionWindow;
    int32_t mSinkTokens;
    int32_t mCyclicCacheLen;
    int32_t mBubbleLen;
    int32_t mValidRowsPerSeq;
    bool mEnableOneMoreBlock;
    DataType* data;

    KVLinearBuffer()
        : mMaxSeqs{0}
        , mMaxSeqLen{0}
        , mBytesPerSeq{0}
        , mMaxAttentionWindow{0}
        , mSinkTokens{0}
        , mCyclicCacheLen{0}
        , mBubbleLen{0}
        , mValidRowsPerSeq{0}
        , mEnableOneMoreBlock{false}
        , data{nullptr}
    {
    }

    KVLinearBuffer(int32_t batchSize, int32_t tokensPerBlock, int32_t sizePerToken, int32_t maxAttentionWindow,
        int32_t sinkTokenLen, bool onlyKorV, DataType* data)
        : mMaxSeqs(batchSize)
        , mMaxSeqLen(tokensPerBlock)
        , mBytesPerSeq(tokensPerBlock * sizePerToken)
        , mMaxAttentionWindow(maxAttentionWindow)
        , mSinkTokens(sinkTokenLen)
        , data(data)
    {
        CHECK_WITH_INFO(
            static_cast<int64_t>(mMaxSeqs - 1) * mBytesPerSeq * 2 + mBytesPerSeq <= std::numeric_limits<int32_t>::max(),
            "kv cache is too large for gpt_attention_plugin");
        mCyclicCacheLen = mMaxAttentionWindow - mSinkTokens;
        mBubbleLen = 0;
        mValidRowsPerSeq = (onlyKorV) ? 1 : 2;
        mEnableOneMoreBlock = false;
    }

    __host__ __device__ [[nodiscard]] inline bool isSinkToken(int32_t tokenIdx) const
    {
        return tokenIdx < mSinkTokens;
    }

    __host__ __device__ [[nodiscard]] inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx) const
    {
        return reinterpret_cast<void**>(data + seqIdx * mBytesPerSeq * mValidRowsPerSeq
            + static_cast<int32_t>(kvIdx) * mBytesPerSeq * (mValidRowsPerSeq - 1));
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVTokenIdx(int32_t tokenIdx) const
    {
        if (!isSinkToken(tokenIdx))
        {
            return mSinkTokens + (tokenIdx - mSinkTokens) % mCyclicCacheLen;
        }
        return tokenIdx;
    }

    __host__ __device__ static inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
    {
        return reinterpret_cast<void*>(pointer);
    }

    __host__ __device__ [[nodiscard]] inline void* getKBlockPtr(int32_t seqIdx, int32_t) const
    {
        return reinterpret_cast<void*>(getRowPtr(KVIdxType::K_IDX, seqIdx));
    }

    __host__ __device__ [[nodiscard]] inline void* getVBlockPtr(int32_t seqIdx, int32_t) const
    {
        return reinterpret_cast<void*>(getRowPtr(KVIdxType::V_IDX, seqIdx));
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVLocalIdx(
        int32_t tokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const
    {
        return headIdx * mMaxSeqLen * dimsPerHead + tokenIdx * dimsPerHead + channelIdx;
    }
};

}
