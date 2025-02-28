
#pragma once

#include "../common/mpiUtils.h"
#include "../runtime/eagleBuffers.h"
#include "../runtime/explicitDraftTokensBuffers.h"
#include "../runtime/iTensor.h"
#include "../runtime/lookaheadBuffers.h"
#include "../runtime/modelConfig.h"
#include "../runtime/worldConfig.h"

#include <optional>
#include <vector>

namespace sugesstify::runtime
{
class TllmRuntime;
}

namespace sugesstify::batch_manager
{

class DecoderStepAsyncSend
{
public:
    using BufferPtr = runtime::IBuffer::SharedPtr;

    DecoderStepAsyncSend(std::shared_ptr<mpi::MpiComm> const& commSession, BufferPtr const& newOutputTokensHost,
        BufferPtr const& finished, BufferPtr const& sequenceLengthsHost, BufferPtr const& cumLogProbsHost,
        BufferPtr const& logProbsHost, BufferPtr const& cacheIndirectionOutput, BufferPtr const& acceptedCumSum,
        BufferPtr const& packedPaths, BufferPtr const& finishReasonsHost, int peer);

    ~DecoderStepAsyncSend();

    static auto constexpr kMpiTagOffset = 0;
    static auto constexpr kMpiTagUpperBound = kMpiTagOffset + 9;

private:
    std::shared_ptr<mpi::MpiRequest> mRequest1;
    std::shared_ptr<mpi::MpiRequest> mRequest2;
    std::shared_ptr<mpi::MpiRequest> mRequest3;
    std::shared_ptr<mpi::MpiRequest> mRequest4;
    std::shared_ptr<mpi::MpiRequest> mRequest5;
    std::shared_ptr<mpi::MpiRequest> mRequest6;
    std::shared_ptr<mpi::MpiRequest> mRequest7;
    std::shared_ptr<mpi::MpiRequest> mRequest8;
    std::shared_ptr<mpi::MpiRequest> mRequest9;
};

class DecoderSlotAsyncSend
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;

    DecoderSlotAsyncSend(std::shared_ptr<mpi::MpiComm> const& commSession, TensorPtr const& outputIdsView,
        TensorPtr const& sequenceLengthView, TensorPtr const& cumLogProbsView, TensorPtr const& logProbsView,
        bool returnLogProbs, int peer);

    ~DecoderSlotAsyncSend();

    static auto constexpr kMpiTagOffset = 9;
    static auto constexpr kMpiTagUpperBound = kMpiTagOffset + 4;
    static_assert(kMpiTagOffset >= DecoderStepAsyncSend::kMpiTagUpperBound);

private:
    std::shared_ptr<mpi::MpiRequest> mRequest1;
    std::shared_ptr<mpi::MpiRequest> mRequest2;
    std::shared_ptr<mpi::MpiRequest> mRequest3;
    std::shared_ptr<mpi::MpiRequest> mRequest4;
};

class DecoderBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    std::vector<TensorPtr> logits;
    TensorPtr slotOutputIds;
    TensorPtr slotOutputIdsHost;
    TensorPtr cacheIndirectionInput;
    TensorPtr cacheIndirectionOutput;
    TensorPtr sequenceLengths;
    TensorPtr sequenceLengthsHost;
    TensorPtr finished;
    TensorPtr newOutputTokens;
    TensorPtr newOutputTokensHost;
    TensorPtr cumLogProbs;
    TensorPtr cumLogProbsHost;
    TensorPtr logProbs;
    TensorPtr logProbsHost;
    TensorPtr finishReasonsHost;

    class DraftBuffers
    {
    public:
        TensorPtr nextDraftTokensDevice;
        TensorPtr nextDraftTokensHost;
        TensorPtr prevDraftTokensLengthsDevice;
        TensorPtr prevDraftTokensLengthsHost;
        TensorPtr nextDraftTokensLengthsDevice;
        TensorPtr nextDraftTokensLengthsHost;
        TensorPtr acceptedLengthsCumSumDevice;
        TensorPtr acceptedPackedPathsDevice;
        std::vector<std::vector<runtime::ITensor::SharedPtr>>
            predictedDraftLogits;

        void create(SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, runtime::TllmRuntime const& runtime,
            runtime::ModelConfig const& modelConfig);
    };

    DraftBuffers draftBuffers;
    runtime::ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers;
    runtime::EagleBuffers::Inputs eagleBuffers;
    std::optional<runtime::LookaheadDecodingBuffers> lookaheadBuffers;

    DecoderBuffers(SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        SizeType32 maxSeqLen, SizeType32 maxTokensPerStep, runtime::TllmRuntime const& runtime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    std::unique_ptr<DecoderStepAsyncSend> asyncSend(std::shared_ptr<mpi::MpiComm> const& commSession,
        bool returnLogProbs, SizeType32 maxBeamWidth, bool useMedusa, int peer);

    void recv(std::shared_ptr<mpi::MpiComm> const& commSession, bool returnLogProbs, SizeType32 maxBeamWidth,
        bool useMedusa, int peer);

    void bcast(std::shared_ptr<mpi::MpiComm> const& commSession, bool returnLogProbs, SizeType32 maxBeamWidth,
        bool useMedusa, int root);

    void enableLookaheadDecoding(SizeType32 maxNumSequences, SizeType32 maxTokensPerStep);
    void disableLookaheadDecoding(SizeType32 maxNumSequences);
};

class SlotDecoderBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;

    TensorPtr outputIds;
    TensorPtr outputIdsHost;
    TensorPtr sequenceLengthsHost;
    TensorPtr cumLogProbs;
    TensorPtr cumLogProbsHost;
    TensorPtr logProbs;
    TensorPtr logProbsHost;
    TensorPtr finishReasonsHost;

    SlotDecoderBuffers(SizeType32 maxBeamWidth, SizeType32 maxSeqLen, runtime::TllmRuntime const& runtime);

    static std::unique_ptr<DecoderSlotAsyncSend> asyncSend(std::shared_ptr<mpi::MpiComm> const& commSession,
        TensorPtr const& outputIdsView, TensorPtr const& sequenceLengthView, TensorPtr const& cumLogProbsView,
        TensorPtr const& logProbsView, bool returnLogProbs, int peer);

    std::unique_ptr<DecoderSlotAsyncSend> asyncSend(std::shared_ptr<mpi::MpiComm> const& commSession,
        TensorPtr const& sequenceLengthView, bool returnLogProbs, int peer);

    void recv(std::shared_ptr<mpi::MpiComm> const& commSession, TensorPtr const& sequenceLengthView,
        bool returnLogProbs, int peer);
};

}
