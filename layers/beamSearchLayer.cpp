
#include "beamSearchLayer.h"
#include "../src/beamSearchKernels/beamSearchKernelsTemplate.h"

#include "../common/cudaUtils.h"
#include "../common/stringUtils.h"
#include "../src/beamSearchKernels.h"
#include "defaultDecodingParams.h"
#include "layerUtils.h"
#include <limits>

using namespace suggestify::runtime;
using namespace suggestify::kernels;

namespace suggestify::layers
{

#define GET_INFO_STAGE1(nPBM)                                                                                          \
    {                                                                                                                  \
        int constexpr nBlock = (nPBM < 16) ? ((nPBM < 8) ? nThreadForSmallBeamWidth : 128) : 64;                       \
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                 \
            &nMaxActiveBlock, beamStage1Kernel<T, 2 * nPBM, nBlock>, nBlock, 0));                                      \
        CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage1Kernel<T, 2 * nPBM, nBlock>));                          \
        break;                                                                                                         \
    }

#define GET_INFO_STAGE2(nPBM)                                                                                          \
    {                                                                                                                  \
        if (nByteDynamicSharedMemoryStage2 > nByteMaxSharedMemoryPerBlock)                                             \
        {                                                                                                              \
            CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, nPBM, 128, false>));                      \
        }                                                                                                              \
        else if (nVPart <= 32)                                                                                         \
        {                                                                                                              \
            CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, nPBM, 32, true>));                        \
        }                                                                                                              \
        else if (nVPart <= 64)                                                                                         \
        {                                                                                                              \
            CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, nPBM, 64, true>));                        \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage2Kernel<T, nPBM, 128, true>));                       \
        }                                                                                                              \
        break;                                                                                                         \
    }

#define GET_INFO_STAGE3(nPBM, bV2)                                                                                     \
    {                                                                                                                  \
        int constexpr nThreadStage3 = (nPBM + 31) / 32 * 32;                                                           \
        CUDA_CHECK(cudaFuncGetAttributes(&attr, beamStage3Kernel<T, nPBM, nThreadStage3, true, bV2>));            \
        break;                                                                                                         \
    }

template <typename T>
BeamSearchLayer<T>::BeamSearchLayer(DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 const nBS{mDecoderDomain.getBatchSize()};
    SizeType32 const nBM{mDecoderDomain.getBeamWidth()};
    SizeType32 const nV{mDecoderDomain.getVocabSize()};
    CHECK_WITH_INFO(nBM <= nMaxBeamWidth, "Beam width is larger than the maximum supported (%d > %d)", int(nBM),
        int(nMaxBeamWidth));

    allocateBuffer();
    configureBeamSearchLayer();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::allocateBuffer()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mBeamSearchDiversityRateHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mLengthPenaltyHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mEarlyStoppingHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<int>::value);
    mBeamSearchDiversityRateDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mLengthPenaltyDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mEarlyStoppingDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<int>::value);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BeamSearchLayer<T>::configureBeamSearchLayer()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 const nBS{mDecoderDomain.getBatchSize()};
    SizeType32 const nBM{mDecoderDomain.getBeamWidth()};
    SizeType32 const nV{mDecoderDomain.getVocabSize()};
    SizeType32 const nPBM{padToNextPowerOfTwo(nBM)};
    cudaFuncAttributes attr;

    int nByteMaxSharedMemoryPerSM = -1, nByteMaxSharedMemoryPerBlock = -1;
    int const device = suggestify::common::getDevice();
    CUDA_CHECK(
        cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&nByteMaxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    this->mByteMaxSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock;
    int const nByteReservedSharedMemoryPerBlock = nByteMaxSharedMemoryPerSM - nByteMaxSharedMemoryPerBlock;

    if (nBM <= nMaxBeamWidthForV1)
    {
        int nMaxActiveBlock = -1;
        switch (nPBM)
        {
        case 1: GET_INFO_STAGE1(1);
        case 2: GET_INFO_STAGE1(2);
        case 4: GET_INFO_STAGE1(4);
        case 8: GET_INFO_STAGE1(8);
        default: break;
        }
        int nByteStaticSharedMemory = attr.sharedSizeBytes;
        int nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        CHECK_WITH_INFO(nByteMaxDynamicSharedMemoryPerBlock * nMaxVPartStage1 >= sizeof(T) * nV,
            "vocab_size is too large for Beam search.");
        int nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        int nBlock = nMaxActiveBlock;
        int nVPart = nMaxVPartStage1 + 1;
        for (; nBlock > 0 && nVPart > nMaxVPartStage1; --nBlock)
        {
            int nByteDynamicSharedMemoryStage1 = nByteMaxSharedMemoryPerSM / nBlock - nByteExtralSharedMemory;
            nByteDynamicSharedMemoryStage1 -= nByteDynamicSharedMemoryStage1 % sizeof(T);
            nVPart = ceilDiv(sizeof(T) * nV, nByteDynamicSharedMemoryStage1);
        }
        CHECK_WITH_INFO(nBlock >= 0, "No enough active blocks for Beam Search stage 1 kernel.");

        int const nByteDynamicSharedMemoryStage1 = sizeof(T) * ceilDiv(nV, nVPart);
        this->mVPart = nVPart;
        this->mByteSharedMemoryStage1 = nByteDynamicSharedMemoryStage1;

        CHECK_WITH_INFO(nBS * nBM * nPBM < (1 << 21),
            "max_batch_size or max_beam_width of TRT-LLM engine is too large for Beam search, try to decrease the "
            "parameters while building.");
        size_t const nByteDynamicSharedMemoryStage2
            = common::roundUp(sizeof(float) * nVPart * (nPBM * 4) + sizeof(cub::KeyValuePair<int, T>) * nPBM * 2, 4);
        switch (nPBM)
        {
        case 1: GET_INFO_STAGE2(1);
        case 2: GET_INFO_STAGE2(2);
        case 4: GET_INFO_STAGE2(4);
        case 8: GET_INFO_STAGE2(8);
        default: break;
        }
        nByteStaticSharedMemory = attr.sharedSizeBytes;
        nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        bool const bUseGlobalMemoryStage2 = (nByteDynamicSharedMemoryStage2 > nByteMaxDynamicSharedMemoryPerBlock);

        size_t const nByteDynamicSharedMemoryStage3 = common::roundUp(sizeof(T) * nPBM * nPBM * 2, 4);
        switch (nPBM)
        {
        case 1: GET_INFO_STAGE3(1, false);
        case 2: GET_INFO_STAGE3(2, false);
        case 4: GET_INFO_STAGE3(4, false);
        case 8: GET_INFO_STAGE3(8, false);
        }
        nByteStaticSharedMemory = attr.sharedSizeBytes;
        nByteMaxDynamicSharedMemoryPerBlock = nByteMaxSharedMemoryPerBlock - nByteStaticSharedMemory;
        nByteExtralSharedMemory = nByteReservedSharedMemoryPerBlock + nByteStaticSharedMemory;
        bool const bUseGlobalMemoryStage3 = (nByteDynamicSharedMemoryStage3 > nByteMaxDynamicSharedMemoryPerBlock);
        this->mByteSharedMemoryStage3 = nByteStaticSharedMemory;

        size_t const nByteA = common::roundUp(sizeof(T) * nBS * nPBM * nPBM * 4, 4);
        size_t const nByteB = common::roundUp(sizeof(T) * nBS * nPBM * nMaxVPartStage1 * nPBM * 4, 4);
        size_t const nByteC = (bUseGlobalMemoryStage2) ? nByteDynamicSharedMemoryStage2 : 0;
        size_t const nByteD = (bUseGlobalMemoryStage3) ? nByteDynamicSharedMemoryStage3 : 0;
        this->mWorkspaceSize = nByteA + std::max(nByteB + nByteC, nByteD);
    }
    else
    {
        this->mV2 = true;
        switch (nPBM)
        {
        case 1: GET_INFO_STAGE3(1, true);
        case 2: GET_INFO_STAGE3(2, true);
        case 4: GET_INFO_STAGE3(4, true);
        case 8: GET_INFO_STAGE3(8, true);
        case 16: GET_INFO_STAGE3(16, true);
        case 32: GET_INFO_STAGE3(32, true);
        case 64: GET_INFO_STAGE3(64, true);
        case 128: GET_INFO_STAGE3(128, true);
        case 256: GET_INFO_STAGE3(256, true);
        case 512: GET_INFO_STAGE3(512, true);
        case 1024: GET_INFO_STAGE3(1024, true);
        }
        this->mByteSharedMemoryStage3 = attr.sharedSizeBytes;

        SizeType32 const nBS{mDecoderDomain.getBatchSize()};
        SizeType32 const nBM{mDecoderDomain.getBeamWidth()};
        SizeType32 const nV{mDecoderDomain.getVocabSize()};
        SizeType32 const nPBM = padToNextPowerOfTwo(nBM);
        size_t const nByteStage1LogProbs = roundUp(sizeof(T) * nBS * nPBM * nPBM * 2, 4);
        size_t const nByteStage1Ids = roundUp(sizeof(int) * nBS * nPBM * nPBM * 2, 4);
        size_t const nByteStage2LogProbs = roundUp(sizeof(T) * nBS * nPBM * 2, 4);
        size_t const nByteStage2Ids = roundUp(sizeof(int) * nBS * nPBM * 2, 4);
        size_t const nByteStage1TopK = invokeComputeTopkLastDimWorkspaceSize<T>(nBS * nBM, nV, nPBM * 2, true);
        size_t const nByteStage2TopK = invokeComputeTopkLastDimWorkspaceSize<T>(nBS, nPBM * nPBM * 2, nBM * 2, true);
        size_t const nByteStage3 = sizeof(T) * nBM * nBM * 2;
        this->mWorkspaceSize = nByteStage2LogProbs + nByteStage2Ids
            + max(nByteStage1LogProbs + nByteStage1Ids + max(nByteStage1TopK, nByteStage2TopK), nByteStage3);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t BeamSearchLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template <typename T>
void BeamSearchLayer<T>::setup(SizeType32 const batchSize, SizeType32 const beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 const nBM{mDecoderDomain.getBeamWidth()};
    CHECK_WITH_INFO(
        beamWidth <= nBM, "Beam width is larger than the constructed for (%d > %d).", int(beamWidth), int(nBM));

    auto setupParams = std::dynamic_pointer_cast<BeamSearchSetupParams>(baseSetupParams);

    auto constexpr fltMax = std::numeric_limits<float>::max();
    auto constexpr fltMin = std::numeric_limits<float>::lowest();
    auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};
    fillBuffers(setupParams->beamSearchDiversityRate, DefaultDecodingParams::getBeamSearchDiversity(),
        mBeamSearchDiversityRateHost, mBeamSearchDiversityRateDevice, batchSlots, std::make_pair(-fltEpsilon, fltMax),
        "diversity rate");
    fillBuffers(setupParams->lengthPenalty, DefaultDecodingParams::getLengthPenalty(), mLengthPenaltyHost,
        mLengthPenaltyDevice, batchSlots, std::make_pair(fltMin, fltMax), "length penalty");
    fillBuffers(setupParams->earlyStopping, DefaultDecodingParams::getEarlyStopping(), mEarlyStoppingHost,
        mEarlyStoppingDevice, batchSlots, std::make_pair(-fltEpsilon, std::numeric_limits<int>::max()),
        "early stopping");

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

__global__ void updateCacheIndirectionKernel(
    int* tgtCI, int const* srcCI, BeamHypotheses bh, int const nMaxAttentionWindow, int const nSinkTokenLength)
{
    int const step = threadIdx.x + blockIdx.x * blockDim.x;
    size_t const nBM{bh.nBeamWidth};
    size_t const nMSL{bh.nMaxSeqLen};
    int const indexBatch = blockIdx.y;
    int const batchSlot = bh.batchSlots[indexBatch];
    int const indexBeam = blockIdx.z;
    int const indexBatchBeam = batchSlot * nBM + indexBeam;
    int const lastStep{bh.sequenceLengths[indexBatchBeam] - 1};

    if (step >= nMSL || step < bh.inputLengths[indexBatchBeam] || step < (nMSL - nMaxAttentionWindow)
        || bh.finished[indexBatchBeam].isFinished())
    {
        return;
    }

    int const indexBeamSrc = bh.parentIdsPtr[batchSlot][indexBeam * nMSL + lastStep];
    int const stepCirc = (step >= nSinkTokenLength)
        ? nSinkTokenLength + (step - nSinkTokenLength) % (nMaxAttentionWindow - nSinkTokenLength)
        : step;
    uint32_t const tgtOffset = batchSlot * nBM * nMaxAttentionWindow + indexBeam * nMaxAttentionWindow + stepCirc;
    uint32_t const srcOffset = batchSlot * nBM * nMaxAttentionWindow + indexBeamSrc * nMaxAttentionWindow + stepCirc;
    tgtCI[tgtOffset] = (step == lastStep) ? indexBeam : srcCI[srcOffset];
}

template <typename T>
void BeamSearchLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto ip = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    auto op = std::dynamic_pointer_cast<BeamSearchOutputs>(baseOutputs);
    auto const localDecoderDomain = getLocalDecoderDomain(ip, mDecoderDomain);

    CHECK_WITH_INFO(localDecoderDomain.getBeamWidth() > 1, "Use beamWidth <= 1 (%d <= 1) in Beam Search mode",
        localDecoderDomain.getBeamWidth());
    CHECK_WITH_INFO(ip->srcCacheIndirection.has_value(), "srcCacheIndirection is mandatory in beam search.");
    CHECK_WITH_INFO(op->parentIds.has_value(), "parentIds tensor is mandatory in beam search.");
    CHECK_WITH_INFO(op->finished.has_value(), "finished tensor is mandatory in beam search.");
    CHECK_WITH_INFO(op->cumLogProbs.has_value(), "cumLogProbs tensor is mandatory in beam search.");
    CHECK_WITH_INFO(op->beamHypotheses, "Output BeamHypotheses is not set.");
    CHECK_WITH_INFO(bufferCastOrNull<int>(*op->sequenceLength) != nullptr || mLengthPenaltyDevice == nullptr,
        "Current sequence lengths must be set for length penalty computation.");
    CHECK_WITH_INFO(ip->ite == 0, "Pipeline Parallelism is not supported yet!");

    BeamHypotheses bh;
    bh.nMaxBatchSize = static_cast<std::int32_t>(op->outputIdsPtr->getDimension<0>());
    bh.nBatchSize = ip->localBatchSize;
    bh.nBeamWidth = op->outputIds->getDimension<1>();
    bh.nMaxSeqLen = op->outputIds->getDimension<2>();
    bh.nVocabSize = mDecoderDomain.getVocabSizePadded();
    bh.nVPart = this->mVPart;
    bh.nByteMaxSharedMemoryPerBlock = this->mByteMaxSharedMemoryPerBlock;
    bh.nByteSharedMemoryStage1 = this->mByteSharedMemoryStage1;
    bh.nByteSharedMemoryStage3 = this->mByteSharedMemoryStage3;

    bh.diversityRates = bufferCast<float>(*mBeamSearchDiversityRateDevice);
    bh.lengthPenalties = bufferCast<float>(*mLengthPenaltyDevice);
    bh.earlyStoppings = bufferCast<int>(*mEarlyStoppingDevice);

    bh.inputLengths = bufferCast<SizeType32>(*ip->inputLengths.value());
    bh.endIds = bufferCast<TokenIdType>(*ip->endIds);
    bh.batchSlots = workspace->getDeviceBatchSlotsPtr();

    bh.logProbsTiled = bufferCastOrNull<float>(op->outputLogProbsTiled);
    bh.sequenceLengths = bufferCast<SizeType32>(*op->sequenceLength.value());
    bh.cumLogProbs = bufferCast<float>(*op->cumLogProbs.value());

    bh.outputIdsCBA = op->beamHypotheses->outputIdsCBA;
    bh.logProbsCBA = op->beamHypotheses->logProbsCBA;
    bh.sequenceLengthsCBA = op->beamHypotheses->sequenceLengthsCBA;
    bh.cumLogProbsCBA = op->beamHypotheses->cumLogProbsCBA;
    bh.normedScoresCBA = op->beamHypotheses->normedScoresCBA;
    bh.numBeamsCBA = op->beamHypotheses->numBeamsCBA;
    bh.minNormedScoresCBA = op->beamHypotheses->minNormedScoresCBA;

    bh.batchDones = op->beamHypotheses->batchDones;
    bh.finished = reinterpret_cast<FinishedState*>(bufferCast<FinishedState::UnderlyingType>(*op->finished.value()));

    bh.outputIdsPtr = bufferCast<TokenIdType*>(*op->outputIdsPtr);
    bh.parentIdsPtr = bufferCast<TokenIdType*>(*op->parentIdsPtr);

    T const* logProbs = bufferCast<T>(*workspace->getDeviceRuntimeLogits());
    T const* bias = static_cast<T const*>(nullptr);
    CHECK_WITH_INFO(getWorkspaceSize() >= 2 * bh.nBatchSize * bh.nBeamWidth * bh.nBeamWidth * 2,
        "Workspace size (%lu) is not enough for topk softmax required (%lu).", (uint64_t) getWorkspaceSize(),
        (uint64_t) (2 * bh.nMaxBatchSize * bh.nBeamWidth * bh.nBeamWidth * 2));

    if (this->mV2)
    {
        invokeTopkBeamSearch<T, true>(logProbs, bias, workspace->getRawWorkspaceDevicePtr(), bh, getStream());
    }
    else
    {
        invokeTopkBeamSearch<T, false>(logProbs, bias, workspace->getRawWorkspaceDevicePtr(), bh, getStream());
    }

    auto tgtCI = bufferCast<int>(*op->tgtCacheIndirection);
    auto srcCI = bufferCast<int>(*ip->srcCacheIndirection.value());
    dim3 const grid(common::roundUp(bh.nMaxSeqLen, 32), bh.nBatchSize, bh.nBeamWidth);
    updateCacheIndirectionKernel<<<grid, 32, 0, getStream()>>>(
        tgtCI, srcCI, bh, ip->maxAttentionWindow, ip->sinkTokenLength);
    sync_check_cuda_error();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BeamSearchLayer<float>;
template class BeamSearchLayer<half>;

}
