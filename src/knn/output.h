#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <stdint.h>
#include <stdio.h>

#define NNFloat float

#ifdef SYNCHRONOUS
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            exit(-1); \
        } \
    }
#endif

void invokeTopK(float* pOutput, float *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t widthPadding, uint32_t k);

#endif /* LIBKNN_TOPK_H_ */
