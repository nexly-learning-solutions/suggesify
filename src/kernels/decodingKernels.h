
#pragma once

#include "gptKernels.h"
#include "../src/beamSearchKernels.h"
#include "../src/decodingCommon.h"
#include "../runtime/common.h"
#include "../runtime/decodingInput.h"
#include "../runtime/decodingOutput.h"
#include "../runtime/samplingConfig.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace suggestify
{

namespace kernels
{

struct gatherTreeParam
{
    int32_t* beams = nullptr;
    int32_t* sequenceLengths = nullptr;
    int32_t maxSequenceLengthFinalStep = 0;
    int32_t const* inputLengths = nullptr;
    int32_t* responseInputLengths = nullptr;
    int32_t maxSeqLen = 0;
    int32_t batchSize = 0;
    int32_t beamWidth = 0;
    int32_t const* stepIds = nullptr;
    int32_t const* parentIds = nullptr;
    int32_t const* endTokens = nullptr;
    int32_t* outputIds = nullptr;
    cudaStream_t stream;
    float* cumLogProbs = nullptr;
    float lengthPenalty = 1.0f;
    int earlyStopping = 1;
};

void invokeGatherTree(gatherTreeParam param);

void invokeInsertUnfinishedPath(BeamHypotheses& bh, cudaStream_t stream);

void invokeFinalize(BeamHypotheses& bh, cudaStream_t stream);

void invokeInitializeOutput(runtime::TokenIdType* finalOutputIds, runtime::TokenIdType const* endIds,
    runtime::SizeType32 batch, runtime::SizeType32 beam, runtime::SizeType32 maxSeqLen, cudaStream_t stream);

void invokeCopyBeamHypotheses(runtime::DecodingOutput::BeamHypotheses const& src,
    runtime::DecodingOutput::BeamHypotheses const& dst, runtime::ITensor& srcCumLogProbs,
    runtime::ITensor& dstCumLogProbs, runtime::CudaStream const& stream, int numSMs);

void invokeCopyNextStepIds(runtime::TokenIdType* nextStepIds, runtime::TokenIdType const* const* outputIdsPtr,
    runtime::SizeType32 const* sequenceLengths, runtime::SizeType32 const* numNewTokens,
    runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 maxBatchSize,
    runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen, runtime::SizeType32 maxTokensPerStep,
    cudaStream_t stream);

void invokeTransposeLogProbs(float* output_log_probs, float* output_log_probs_tiled,
    runtime::SizeType32 const* sequence_lengths, runtime::SizeType32 const* batchSlots, runtime::SizeType32 batch_size,
    runtime::SizeType32 max_batch_size, runtime::SizeType32 beam_width, runtime::SizeType32 max_seq_len,
    cudaStream_t stream);

}

namespace runtime::kernels
{

void gatherTree(DecodingOutput const& decodingOutput, DecodingInput const& decodingInput, BufferManager const& manager,
    SamplingConfig const& samplingConfig);
}

}
