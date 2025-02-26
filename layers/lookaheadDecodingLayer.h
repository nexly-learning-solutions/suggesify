
#pragma once

#include "lookaheadAlgorithm.h"
#include "baseLayer.h"
#include "decodingParams.h"
#include "suggestify/runtime/common.h"

namespace suggestify::layers
{

template <typename T>
class LookaheadDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using Base::mBufferManager;

    LookaheadDecodingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& baseSetupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputParams,
        std::shared_ptr<BaseDecodingInputs> const& inputParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void forwardSyncCPU(std::shared_ptr<LookaheadDecodingOutputs> const& outputs,
        std::shared_ptr<LookaheadDecodingInputs> const& inputs);

private:
    using Base::mDecoderDomain;

    size_t mWorkspaceSize{};
    size_t mSetupWorkspaceSize{};
    TensorPtr mCurandStatesDevice;
    TensorPtr mTargetTokensDevice;

    struct CpuAlgorithmResources
    {
        explicit CpuAlgorithmResources(DecoderDomain const& decoderDomain);

        std::vector<LookaheadAlgorithm> mAlgos;
        std::vector<TensorPtr> mPrompts;
        TensorPtr mBatchSlots;
        TensorPtr mTargetTokens;
        TensorPtr mTokensPerStep;
        TensorPtr mEndIds;

        TensorPtr mOutputIds;
        TensorPtr mPathsOffsets;
        TensorPtr mPathsOffsetsBatch;
        TensorPtr mNumNewTokens;
        TensorPtr mNumNewTokensCumSum;
        TensorPtr mNewTokens;

        TensorPtr mNextDraftTokens;
        TensorPtr mNextDraftPosIds;
        TensorPtr mNextDraftLengths;
        TensorPtr mSequenceLengths;
        TensorPtr mGenerationLengths;
        TensorPtr mAttentionMask;
        TensorPtr mPackedMask;
        TensorPtr mPositionOffsets;
        TensorPtr mPositionIds;
    };

    std::optional<CpuAlgorithmResources> mCpuAlgo;

    runtime::SizeType32 mGlobalSteps{0};
};

}
