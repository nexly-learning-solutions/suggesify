#include "../types.h"
#include "../bitonicSort.cuh"
#include "../gpuTypes.h"
#include "../src.cuh"
#include <limits>
#include "output.h"
#include <limits>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <nccl.h>
#include <mpi.h>

#define NEG_INF -1.0e30f

__device__ void bitonicSort(
    float* sKey, uint32_t* sValue, uint32_t bufferSize, uint32_t start, uint32_t end, bool ascending)
{
    uint32_t stride = end - start;
    for (uint32_t i = start; i < end; i += stride)
    {
        for (uint32_t j = i; j < i + stride / 2; j++)
        {
            uint32_t idx1 = j;
            uint32_t idx2 = j + stride / 2;
            bool compare = (ascending ? sKey[idx1] < sKey[idx2] : sKey[idx1] > sKey[idx2]);
            if (compare)
            {
                float tempKey = sKey[idx1];
                uint32_t tempValue = sValue[idx1];
                sKey[idx1] = sKey[idx2];
                sValue[idx1] = sValue[idx2];
                sKey[idx2] = tempKey;
                sValue[idx2] = tempValue;
            }
        }
    }
}

__global__ void invokeTopK_kernel(float* pOutputBuffer, float* pKeyBuffer, uint32_t* pValueBuffer, uint32_t batch,
    uint32_t width, uint32_t widthPadding, uint32_t k)
{
    __shared__ float sKey[128];
    __shared__ uint32_t sValue[128];

    uint32_t dataWidth = width - widthPadding;
    uint32_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = globalIdx >> 5; // Position in batch
    uint32_t threadIdxWithinWarp = threadIdx.x & 31;

    if (pos >= batch)
        return;

    float* pOutput = pOutputBuffer + pos * width;
    float localKeys[8] = {NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF};
    uint32_t localValues[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t localPos = threadIdxWithinWarp;
    for (int i = 0; i < 4; i++)
    {
        if (localPos < dataWidth)
        {
            localKeys[i] = pOutput[localPos];
            localValues[i] = localPos;
        }
        localPos += 32;
    }

    float minValue = NEG_INF;
    uint32_t bufferSize = 0;
    uint32_t readPos = 128;

    while (readPos < dataWidth)
    {
        float currentKey
            = (readPos + threadIdxWithinWarp < dataWidth) ? pOutput[readPos + threadIdxWithinWarp] : NEG_INF;
        uint32_t currentValue = (readPos + threadIdxWithinWarp < dataWidth) ? (readPos + threadIdxWithinWarp) : 0;
        uint32_t mask = __ballot_sync(0xFFFFFFFF, currentKey > minValue);
        uint32_t offset = __popc(mask & ((1 << threadIdxWithinWarp) - 1)) + bufferSize;
        if (currentKey > minValue)
        {
            sKey[offset] = currentKey;
            sValue[offset] = currentValue;
        }
        bufferSize += __popc(mask);

        __syncthreads();

        if (bufferSize >= 128)
        {
            bitonicSort(sKey, sValue, bufferSize, 0, 128, true);

            minValue = sKey[0];

            bufferSize -= 128;
            if (threadIdxWithinWarp < bufferSize)
            {
                sKey[threadIdxWithinWarp] = sKey[threadIdxWithinWarp + 128];
                sValue[threadIdxWithinWarp] = sValue[threadIdxWithinWarp + 128];
            }
        }

        __syncthreads();

        readPos += 32;
    }

    if (bufferSize > 0 || dataWidth <= 128)
    {
        float finalKeys[8] = {NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF};
        uint32_t finalValues[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        for (int i = 0; i < 4; i++)
        {
            if (threadIdxWithinWarp + i * 32 < bufferSize)
            {
                finalKeys[i] = sKey[threadIdxWithinWarp + i * 32];
                finalValues[i] = sValue[threadIdxWithinWarp + i * 32];
            }
        }

        bitonicSort(finalKeys, finalValues, bufferSize, 0, bufferSize, true);
    }

    float* pKey = pKeyBuffer + pos * k;
    uint32_t* pValue = pValueBuffer + pos * k;
    for (int i = 0; i < 4; i++)
    {
        uint32_t localWritePos = threadIdxWithinWarp + i * 32;
        if (localWritePos < k)
        {
            pKey[localWritePos] = localKeys[i];
            pValue[localWritePos] = localValues[i];
        }
    }
}

void invokeTopK(
    float* pOutput, float* pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k)
{
    uint32_t blocks = (batch + 3) / 4;
    invokeTopK_kernel<<<blocks, 128>>>(pOutput, pKey, pValue, batch, width, widthPadding, k);
    LAUNCHERROR("invokeTopK_kernel");
}