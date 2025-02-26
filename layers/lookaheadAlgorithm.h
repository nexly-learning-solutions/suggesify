
#pragma once

#include "lookaheadPoolManager.h"
#include "baseLayer.h"
#include "decodingParams.h"
#include "suggestify/runtime/common.h"
#include <curand_kernel.h>
#include <tuple>

namespace suggestify::layers
{

class LookaheadAlgorithm
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;

    LookaheadAlgorithm(
        runtime::SizeType32 maxW, runtime::SizeType32 maxN, runtime::SizeType32 maxG, runtime::SizeType32 id = 0);

    void setup(TensorConstPtr const& prompt, runtime::SizeType32 w, runtime::SizeType32 n, runtime::SizeType32 g,
        uint64_t seed);

    void accept(TensorConstPtr const& generatedTokens);

    void prepare(TensorPtr const& draftTokens, TensorPtr const& positionIds, TensorPtr const& draftLengthPtr,
        TensorPtr const& attentionMask, runtime::SizeType32 attentionMaskOffset,
        TensorConstPtr const& lastPositionIdPtr, TensorConstPtr const& lastTokenPtr);

    void update(TensorPtr const& acceptedTokens, TensorPtr const& acceptedOffsets, TensorPtr const& acceptedLength,
        TensorConstPtr const& sampledTokens, TensorConstPtr const& endToken);

    static void posIdsToMask(TensorPtr const& mask, TensorConstPtr const& posIds);

    static runtime::SizeType32 treeEncode(
        TensorPtr const& tokens, TensorPtr const& posIds, TensorPtr const& masks, TensorPtr const& encodeMap);

private:
    runtime::SizeType32 lookahead(
        TensorPtr const& draftTokens, TensorPtr const& positionIds, runtime::SizeType32 startPosId);

    runtime::SizeType32 guess(TensorPtr const& guessTokens, TensorPtr const& guessIds, runtime::SizeType32 startPosId,
        runtime::TokenIdType lastToken);

    void verify(TensorPtr const& accepted, TensorPtr const& acceptedOffsets, TensorPtr const& acceptedLength,
        runtime::TokenIdType newLastToken, TensorConstPtr const& sampledTokens, TensorConstPtr const& endToken);

private:
    LookaheadPoolManager mPoolManager;
    TensorPtr mPrefillsMax;
    TensorPtr mPrefills;
    TensorPtr mPastTokensMax;
    TensorPtr mPastTokens;
    TensorPtr mKeyTokensMax;
    TensorPtr mKeyTokens;
    TensorPtr mGoldenTokensMax;
    TensorPtr mGoldenTokens;
    TensorPtr mGuessTokensMax;
    TensorPtr mGuessTokens;
    TensorPtr mDraftTokensMax;
    TensorPtr mDraftTokens;
    TensorPtr mAttentionMask;
    TensorPtr mEncodeMapMax;
    TensorPtr mEncodeMap;
    TensorPtr mSampledTokensMax;
    TensorPtr mSampledTokens;

    runtime::SizeType32 const mMaxW{0};
    runtime::SizeType32 const mMaxN{0};
    runtime::SizeType32 const mMaxG{0};
    runtime::SizeType32 mW{0};
    runtime::SizeType32 mN{0};
    runtime::SizeType32 mG{0};
    runtime::SizeType32 mRuntimeMaxDraftLen{0};
    runtime::SizeType32 mRuntimeMaxDraftPathLen{0};
    runtime::SizeType32 mFilling;

    runtime::TokenIdType mCurrentToken;
    runtime::SizeType32 mId;
};

}
