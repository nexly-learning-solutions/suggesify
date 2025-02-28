
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/reduceKernelUtils.cuh"
#include "../common/stringUtils.h"
#include "../src/beamSearchKernels.h"
#include "../src/decodingCommon.h"

using namespace suggestify::common;

namespace suggestify
{
namespace kernels
{

#pragma nv_diag_suppress static_var_with_dynamic_init

template <typename T, int PBM, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void beamStage1Kernel(T const* __restrict logProbs, T const* __restrict bias,
    float* __restrict pStage3, int const* __restrict endIds, FinishedState const* __restrict finished, int const nV,
    runtime::SizeType32 const* batchSlots)
{
    int const nBM = gridDim.y;
    int const tid = threadIdx.x;
    int const slot = batchSlots[blockIdx.x];
    int const nVLocal = (nV + gridDim.z - 1) / gridDim.z;
    int const indexLeft = nVLocal * blockIdx.z;
    int const indexRight = std::min(indexLeft + nVLocal, nV);
    int const nVOffset = (blockIdx.x * nBM + blockIdx.y) * nV;
    int const nVChunk = indexRight - indexLeft;
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    using KVPair = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<KVPair, BLOCK_SIZE>;
    cub::ArgMax argmax;

    __shared__ float smemOutput[PBM * 4];
    __shared__ int threadToUpdate;
    __shared__ typename BlockReduceTopK::TempStorage smemReduceBuffer;
    extern __shared__ char smem[];
    T* smemLogProbs = reinterpret_cast<T*>(smem);

    KVPair kvLocal{-1, -MAX_T_VAL};
    for (int i = indexLeft + tid; i < indexRight; i += BLOCK_SIZE)
    {
        T const b{bias == nullptr ? (T) 0.0f : bias[i]};
        int const index = i - indexLeft;
        T const value = (finished[slot * nBM + blockIdx.y].isFinished()) ? (i == endIds[slot] ? MAX_T_VAL : -MAX_T_VAL)
                                                                         : (logProbs[nVOffset + i] + b);
        smemLogProbs[index] = value;
        kvLocal = argmax(kvLocal, {index, value});
    }
    __syncthreads();

    for (int i = 0; i < 2 * nBM; ++i)
    {
        KVPair kv = BlockReduceTopK(smemReduceBuffer).Reduce(kvLocal, argmax);
        if (tid == 0)
        {
            int const index = nVOffset + indexLeft + kv.key;
            reinterpret_cast<int*>(smemOutput)[i] = index;
            smemOutput[PBM * 2 + i] = kv.value;
            smemLogProbs[kv.key] = -MAX_T_VAL;
            threadToUpdate = kv.key % BLOCK_SIZE;
        }
        __syncthreads();

        if (tid == threadToUpdate && i < 2 * nBM - 1)
        {
            kvLocal.key = nV - 1;
            kvLocal.value = -MAX_T_VAL;
            for (int index = tid; index < nVChunk; index += BLOCK_SIZE)
            {
                kvLocal = argmax(kvLocal, {index, smemLogProbs[index]});
            }
        }
        __syncthreads();
    }
    pStage3 += (blockIdx.x * nBM + blockIdx.y) * gridDim.z * PBM * 4 + blockIdx.z * PBM * 4;
    for (int i = tid; i < PBM * 4; i += BLOCK_SIZE)
    {
        pStage3[i] = smemOutput[i];
    }
}

template <typename T, int PBM, int BLOCK_SIZE, bool IS_FAST>
__launch_bounds__(BLOCK_SIZE) __global__
    void beamStage2Kernel(int* __restrict pStage2Ids, T* __restrict pStage2LogProbs, float* __restrict pStage3,
        float const* __restrict cumLogProbs, runtime::SizeType32 const* batchSlots, int const nV, int const nVPart)
{
    int const nBM = gridDim.y;
    int const gbid = blockIdx.x * gridDim.y + blockIdx.y;
    int const tid = threadIdx.x;
    int const slot = batchSlots[blockIdx.x];
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;

    using KVPair = cub::KeyValuePair<int, T>;
    using BlockReduceTopK = cub::BlockReduce<KVPair, BLOCK_SIZE>;
    cub::ArgMax argmax;

    __shared__ KVPair smemOutput[PBM * 2];
    __shared__ typename BlockReduceTopK::TempStorage smemReduceBuffer;

    float* pStage2Temp = pStage3 + PBM * 4 * gbid * nVPart;
    if constexpr (IS_FAST)
    {
        extern __shared__ char smem[];
        float* smemVal = reinterpret_cast<float*>(smem);
        for (int idx = tid; idx < PBM * 4 * nVPart; idx += BLOCK_SIZE)
        {
            smemVal[idx] = pStage2Temp[idx];
        }
        pStage2Temp = smemVal;
        __syncthreads();
    }

    for (int k = 0; k < 2 * nBM; ++k)
    {
        KVPair kvLocal{nV - 1, -MAX_T_VAL};
        if (tid < nVPart)
        {
            for (int i = 0; i < 2 * nBM; ++i)
            {
                int const index = tid * PBM * 4 + i;
                T const topValue = pStage2Temp[index + PBM * 2];
                kvLocal = argmax(kvLocal, {index, topValue});
            }
        }
        KVPair kv = BlockReduceTopK(smemReduceBuffer).Reduce(kvLocal, argmax);
        if (tid == 0)
        {
            int const offsetLocal = kv.key;
            kv.key = reinterpret_cast<int*>(pStage2Temp)[offsetLocal];
            smemOutput[k] = kv;
            reinterpret_cast<int*>(pStage2Temp)[offsetLocal] = nV - 1;
            pStage2Temp[offsetLocal + PBM * 2] = -MAX_T_VAL;
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        auto const cumLogProb = cumLogProbs[slot * nBM + blockIdx.y];
        for (int i = 0; i < 2 * nBM; ++i)
        {
            pStage2Ids[gbid * 2 * nBM + i] = smemOutput[i].key;
            pStage2LogProbs[gbid * 2 * nBM + i] = (float) smemOutput[i].value + cumLogProb;
        }
    }
}

template <typename T, int PBM, int BLOCK_SIZE, bool IS_FAST, bool IS_V2>
__launch_bounds__(BLOCK_SIZE) __global__ void beamStage3Kernel(
    int const* __restrict pStage2Ids, T const* __restrict pStage2LogProbs, float* __restrict pStage3, BeamHypotheses bh)
{
    T const MAX_T_VAL = std::is_same_v<T, half> ? HALF_FLT_MAX : FLT_MAX;
    int const bid = blockIdx.x;
    int const tid = threadIdx.x;
    int const slot = bh.batchSlots[bid];
    size_t const nMBS{bh.nMaxBatchSize};
    size_t const nBM{bh.nBeamWidth};
    size_t const nV{bh.nVocabSize};
    float const diversityRate{bh.diversityRates[slot]};
    float const lengthPenalty{bh.lengthPenalties[slot]};
    int const earlyStopping{bh.earlyStoppings[slot]};

    using KVPair = cub::KeyValuePair<int, T>;
    __shared__ float smemCumLogProbs[PBM];
    __shared__ int smemSeqLen[PBM];
    __shared__ KVPair smemTopKV[(IS_V2) ? 1 : PBM * 2];

    if (bh.numBeamsCBA != nullptr)
    {
        if (bh.numBeamsCBA[slot] == 0 && tid == 0)
        {
            bh.minNormedScoresCBA[slot] = FLT_MAX;
        }
        else if (earlyStopping == 1 && bh.numBeamsCBA[slot] == nBM
            || earlyStopping != 1 && bh.finished[slot * nBM].isFinished())
        {
            return;
        }
    }

    if constexpr (IS_V2)
    {
        pStage2Ids += bid * nBM * 2;
        pStage2LogProbs += bid * nBM * 2;
    }
    else
    {
        int const nCandidate = nBM * nBM * 2;
        pStage2Ids += bid * nCandidate;
        pStage2LogProbs += bid * nCandidate;
        KVPair kvLocal{nCandidate - 1, -MAX_T_VAL};
        cub::ArgMax argmax;
        extern __shared__ char smem[];
        T* smemVal = nullptr;
        if constexpr (IS_FAST)
        {
            smemVal = reinterpret_cast<T*>(smem);
        }
        else
        {
            smemVal = reinterpret_cast<T*>(pStage3);
        }

        for (int i = tid; i < nCandidate; i += BLOCK_SIZE)
        {
            int const index = bh.numBeamsCBA == nullptr ? i % nBM : i / 2 / nBM;
            T const value = pStage2LogProbs[i] + static_cast<T>(diversityRate * index);
            kvLocal = argmax(kvLocal, {i, value});
            smemVal[i] = value;
        }
        __syncthreads();

        using BlockReduce = cub::BlockReduce<KVPair, BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage smemReduceBuffer;
        __shared__ int threadToUpdate;

        for (int i = 0; i < 2 * nBM; ++i)
        {
            KVPair kv = BlockReduce(smemReduceBuffer).Reduce(kvLocal, argmax);
            if (tid == 0)
            {
                smemTopKV[i] = kv;
                smemVal[kv.key] = -MAX_T_VAL;
                threadToUpdate = kv.key % BLOCK_SIZE;
            }
            __syncthreads();
            if (tid == threadToUpdate && i < 2 * nBM - 1)
            {
                kvLocal.key = nCandidate - 1;
                kvLocal.value = -MAX_T_VAL;
                for (int index = tid; index < nCandidate; index += BLOCK_SIZE)
                {
                    kvLocal = argmax(kvLocal, {index, smemVal[index]});
                }
            }
        }
    }

    if (tid < nBM)
    {
        smemCumLogProbs[tid] = bh.cumLogProbs[slot * nBM + tid];
    }
    __syncthreads();

    if (tid == 0)
    {
        int nBeamForNextStep{0};
        for (int i = 0; i < 2 * nBM; ++i)
        {
            int topId;
            T topLogProb;
            if constexpr (IS_V2)
            {
                topId = pStage2Ids[i];
                topLogProb = pStage2LogProbs[i];
            }
            else
            {
                int const key = smemTopKV[i].key;
                topId = pStage2Ids[key];
                topLogProb = pStage2LogProbs[key];
            }
            bool const isEndToken = (topId % nV == bh.endIds[slot]);
            if (i < nBM && bh.numBeamsCBA != nullptr && isEndToken)
            {
                int const nSeqLen = bh.sequenceLengths[slot * nBM + i] + 1 - bh.inputLengths[slot * nBM + i];
                float const score = applyLengthPenalty(topLogProb, nSeqLen, lengthPenalty);
                int nCBA = bh.numBeamsCBA[slot];
                if (nCBA == nBM)
                {
                    if (score < bh.minNormedScoresCBA[slot])
                    {
                        if (earlyStopping)
                        {
                            break;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < nBM; j++)
                        {
                            if (bh.normedScoresCBA[slot * (nBM * 2) + j] == bh.minNormedScoresCBA[slot])
                            {
                                nCBA = j;
                                bh.numBeamsCBA[slot]--;
                                bh.minNormedScoresCBA[slot] = FLT_MAX;
                                bh.normedScoresCBA[slot * (nBM * 2) + j] = score;
                                for (int l = 0; l < nBM; l++)
                                {
                                    bh.minNormedScoresCBA[slot]
                                        = min(bh.minNormedScoresCBA[slot], bh.normedScoresCBA[slot * (nBM * 2) + l]);
                                }
                                break;
                            }
                        }
                    }
                }
                int indexPrev = (topId / nV) % nBM;
                int const step = bh.sequenceLengths[slot * nBM + indexPrev];
                int const offsetCBA = (slot * nBM * 2 + nCBA) * bh.nMaxSeqLen;
                bh.outputIdsCBA[offsetCBA + step] = bh.endIds[slot];
                if (bh.logProbsCBA != nullptr)
                {
                    bh.logProbsCBA[offsetCBA + step] = (float) topLogProb - smemCumLogProbs[(topId / nV) % nBM];
                }
                for (int j = step - 1; j >= 0; j--)
                {
                    bh.outputIdsCBA[offsetCBA + j] = bh.outputIdsPtr[slot][indexPrev * bh.nMaxSeqLen + j];
                    indexPrev = bh.parentIdsPtr[slot][indexPrev * bh.nMaxSeqLen + j];
                }
                if (bh.logProbsCBA != nullptr && bh.logProbsTiled != nullptr)
                {
                    indexPrev = (topId / nV) % nBM;
                    for (int j = step - 1; j >= 0; j--)
                    {
                        int const index = (j * nMBS + slot) * nBM + indexPrev;
                        bh.logProbsCBA[offsetCBA + j] = bh.logProbsTiled[index];
                        indexPrev = bh.parentIdsPtr[slot][indexPrev * bh.nMaxSeqLen + j];
                    }
                }
                int const index = slot * (nBM * 2) + nCBA;
                bh.sequenceLengthsCBA[index] = step;
                bh.normedScoresCBA[index] = score;
                bh.minNormedScoresCBA[slot] = min(bh.minNormedScoresCBA[slot], bh.normedScoresCBA[index]);
                bh.numBeamsCBA[slot]++;
                bh.cumLogProbsCBA[index] = (float) topLogProb;
            }
            else if (i < nBM || bh.numBeamsCBA != nullptr && !isEndToken)
            {
                int const step = bh.sequenceLengths[slot * nBM + nBeamForNextStep];
                bh.outputIdsPtr[slot][nBeamForNextStep * bh.nMaxSeqLen + step] = topId;
                if (bh.logProbsTiled != nullptr)
                {
                    int const index = step * nMBS * nBM + slot * nBM + nBeamForNextStep;
                    int const indexBeam = topId / nV % nBM;
                    bh.logProbsTiled[index] = (float) topLogProb - smemCumLogProbs[indexBeam];
                }
                bh.cumLogProbs[slot * nBM + nBeamForNextStep] = (float) topLogProb;
                nBeamForNextStep++;
            }
            else
            {
            }

            if (nBeamForNextStep >= nBM)
            {
                break;
            }
        }
    }

    if (tid == 0 && bh.numBeamsCBA != nullptr)
    {
        if (bh.numBeamsCBA[slot] < nBM)
        {
            bh.batchDones[slot] = false;
        }
        else if (earlyStopping == 1)
        {
            bh.batchDones[slot] = true;
        }
        else
        {
            int nSeqLen = bh.sequenceLengths[slot * nBM] + 1 - bh.inputLengths[slot * nBM];
            float const bestCumLogProbs = (IS_V2) ? pStage2LogProbs[0] : smemTopKV[0].value;
            if (earlyStopping != 0 && lengthPenalty > 0.0f)
            {
                nSeqLen = bh.nMaxSeqLen - bh.inputLengths[slot * nBM];
            }
            float const bestAttainableScore = applyLengthPenalty(bestCumLogProbs, nSeqLen, lengthPenalty);
            bh.batchDones[slot] = bh.minNormedScoresCBA[slot] >= bestAttainableScore;
        }
    }
    __syncthreads();

    if (tid < nBM)
    {
        smemSeqLen[tid] = bh.sequenceLengths[slot * nBM + tid];
    }
    __syncthreads();

    if (tid < nBM)
    {
        int const indexBatchBeam = slot * nBM + tid;
        int const step = smemSeqLen[tid];
        if (!bh.finished[indexBatchBeam].isFinished())
        {
            smemSeqLen[tid]++;
        }
        int const newId = bh.outputIdsPtr[slot][tid * bh.nMaxSeqLen + step];
        int const newBeamId = (newId / nV) % nBM;
        int const newTokenId = newId % nV;
        bh.sequenceLengths[indexBatchBeam] = smemSeqLen[newBeamId];
        if (newTokenId == bh.endIds[slot])
        {
            bh.finished[indexBatchBeam].setFinishedEOS();
        }
        bh.parentIdsPtr[slot][tid * bh.nMaxSeqLen + step] = newBeamId;
        bh.outputIdsPtr[slot][tid * bh.nMaxSeqLen + step] = newTokenId;

        if ((earlyStopping == 1) && (bh.numBeamsCBA != nullptr && bh.numBeamsCBA[slot] == nBM)
            || (earlyStopping != 1) && bh.batchDones[slot])
        {
            bh.batchDones[slot] = true;
            bh.finished[indexBatchBeam].setFinished();
        }
    }
}

#define BEAM_STAGE2_KERNEL(N_VOCAB_PART, IS_FAST)                                                                      \
    {                                                                                                                  \
        if (IS_FAST && nByteRuntimeSharedMemory > (48 << 10))                                                          \
        {                                                                                                              \
            CUDA_CHECK(cudaFuncSetAttribute(beamStage2Kernel<T, PBM, N_VOCAB_PART, IS_FAST>,                      \
                cudaFuncAttributeMaxDynamicSharedMemorySize, nByteRuntimeSharedMemory));                               \
        }                                                                                                              \
        beamStage2Kernel<T, PBM, N_VOCAB_PART, IS_FAST>                                                                \
            <<<dim3(nBS, nBM), N_VOCAB_PART, IS_FAST * nByteRuntimeSharedMemory, stream>>>(                            \
                pStage2Ids, pStage2LogProbs, pStage3, bh.cumLogProbs, bh.batchSlots, nV, nVPart);                      \
    }

template <typename T, int PBM, bool IS_V2>
void beamSearchKernelLauncher(
    T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream)
{


    size_t const nBS{bh.nBatchSize};
    size_t const nBM{bh.nBeamWidth};
    size_t const nV{bh.nVocabSize};
    size_t const nVPart{bh.nVPart};
    size_t const nByteMaxSharedMemoryPerBlock{bh.nByteMaxSharedMemoryPerBlock};
    int* pStage2Ids{nullptr};
    T* pStage2LogProbs{nullptr};
    float* pStage3{nullptr};

    if constexpr (IS_V2)
    {
        size_t const offsetStage1 = roundUp(nBS * nBM * nBM * 2, 4);
        size_t const offsetStage2 = roundUp(nBS * nBM * 2, 4);

        pStage2Ids = reinterpret_cast<int*>(workspace);
        int offset = sizeof(int) * offsetStage2;
        pStage2LogProbs = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + offset);
        offset += sizeof(T) * offsetStage2;
        int* pStage1Ids = reinterpret_cast<int*>(reinterpret_cast<char*>(workspace) + offset);
        pStage3 = reinterpret_cast<float*>(reinterpret_cast<char*>(workspace) + offset);
        offset += sizeof(int) * offsetStage1;
        T* pStage1LogProbs = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + offset);
        offset += sizeof(T) * offsetStage1;
        void* pTopK = reinterpret_cast<void*>(reinterpret_cast<char*>(workspace) + offset);

        invokeTopkLastDim<T>(nBS * nBM, nV, nBM * 2, true, logProbs, pStage1LogProbs, pStage1Ids, pTopK, stream);
        sync_check_cuda_error();

        int nThread = min(roundUp(nBM * nBM * 2, 32), 1024);
        addCumLogProbs<<<nBS, nThread, 0, stream>>>(
            pStage1LogProbs, bh.cumLogProbs, bh.finished, bh.endIds, bh.diversityRates, bh.batchSlots, nBS, nBM);
        sync_check_cuda_error();

        invokeTopkLastDim<T>(
            nBS, nBM * nBM * 2, nBM * 2, true, pStage1LogProbs, pStage2LogProbs, pStage2Ids, pTopK, stream);
        sync_check_cuda_error();

        nThread = min(roundUp(nBM * 2, 32), 1024);
        gatherId<<<nBS, nThread, 0, stream>>>(pStage1Ids, pStage2Ids, nBS, nBM, nV);
        sync_check_cuda_error();
    }
    else
    {
        int const offset = roundUp(nBS * nBM * nBM * 2, 4);
        pStage2Ids = reinterpret_cast<int*>(workspace);
        pStage2LogProbs = reinterpret_cast<T*>(pStage2Ids + offset);
        pStage3 = reinterpret_cast<float*>(pStage2LogProbs + offset);

        size_t constexpr nThreadStage1 = (PBM < 16) ? ((PBM < 8) ? nThreadForSmallBeamWidth : 128) : 64;
        dim3 grid(nBS, nBM, bh.nVPart);
        beamStage1Kernel<T, PBM, nThreadStage1><<<grid, nThreadStage1, bh.nByteSharedMemoryStage1, stream>>>(
            logProbs, bias, pStage3, bh.endIds, bh.finished, nV, bh.batchSlots);
        sync_check_cuda_error();

        size_t nByteRuntimeSharedMemory
            = sizeof(float) * nVPart * (PBM * 4) + sizeof(cub::KeyValuePair<int, T>) * PBM * 2;
        if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock && nVPart <= 32)
        {
            BEAM_STAGE2_KERNEL(32, true)
        }
        else if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock && nVPart <= 64)
        {
            BEAM_STAGE2_KERNEL(64, true)
        }
        else if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock)
        {
            BEAM_STAGE2_KERNEL(128, true)
        }
        else
        {
            LOG_TRACE("Use slow Beam Search stage 2 kernel due to large beam_width or vocab_size");
            BEAM_STAGE2_KERNEL(128, false)
        }
        sync_check_cuda_error();
    }

    size_t constexpr nThreadStage3 = (PBM + 31) / 32 * 32;
    size_t const nByteStaticSharedMemory = bh.nByteSharedMemoryStage3;
    size_t const nByteDynamicSharedMemory = (IS_V2) ? 0 : sizeof(T) * nBM * nBM * 2;
    size_t const nByteRuntimeSharedMemory = nByteStaticSharedMemory + nByteDynamicSharedMemory;
    if (nByteRuntimeSharedMemory <= nByteMaxSharedMemoryPerBlock)
    {
        if (nByteRuntimeSharedMemory > (48 << 10))
        {
            CUDA_CHECK(cudaFuncSetAttribute(beamStage3Kernel<T, PBM, nThreadStage3, true, IS_V2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, nByteRuntimeSharedMemory));
        }
        beamStage3Kernel<T, PBM, nThreadStage3, true, IS_V2>
            <<<nBS, nThreadStage3, nByteDynamicSharedMemory, stream>>>(pStage2Ids, pStage2LogProbs, pStage3, bh);
    }
    else
    {
        if (nByteStaticSharedMemory > (48 << 10))
        {
            CUDA_CHECK(cudaFuncSetAttribute(beamStage3Kernel<T, PBM, nThreadStage3, false, IS_V2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, nByteStaticSharedMemory));
        }
        beamStage3Kernel<T, PBM, nThreadStage3, false, IS_V2>
            <<<nBS, nThreadStage3, 0, stream>>>(pStage2Ids, pStage2LogProbs, pStage3, bh);
    }
    sync_check_cuda_error();
}

#undef BEAM_STAGE2_KERNEL

#define INSTANTIATE_BEAM_SEARCH(T, PBM, IS_V2)                                                                         \
    template void beamSearchKernelLauncher<T, PBM, IS_V2>(                                                             \
        T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

}
}
