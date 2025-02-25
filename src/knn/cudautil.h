#ifndef CUDAUTIL_H_
#define CUDAUTIL_H_

#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

static void CHECK_ERR2(cudaError_t e, char const* fname, int line)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "FATAL ERROR: cuda failure(%d): %s in %s#%d\n", e, cudaGetErrorString(e), fname, line);
        exit(-1);
    }
}

static void STATUS_ERR2(cublasStatus_t e, char const* fname, int line)
{
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "FATAL ERROR: cublas failure %d in %s#%d\n", e, fname, line);
        exit(-1);
    }
}

static void LAUNCH_ERR2(char const* kernelName, char const* fname, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "FATAL ERROR: %s launching kernel: %s\n in %s#%d\n", cudaGetErrorString(e), kernelName, fname,
            line);
        exit(-1);
    }
}

#define CHECK_ERR(e)                                                                                                   \
    {                                                                                                                  \
        CHECK_ERR2(e, __FILE__, __LINE__);                                                                             \
    }

#define STATUS_ERR(e)                                                                                                  \
    {                                                                                                                  \
        STATUS_ERR2(e, __FILE__, __LINE__);                                                                            \
    }

#define LAUNCH_ERR(expression)                                                                                         \
    {                                                                                                                  \
        expression;                                                                                                    \
        LAUNCH_ERR2(#expression, __FILE__, __LINE__);                                                                  \
    }

namespace astdl
{
namespace cuda_util
{
void printMemInfo(char const* header = "");
void getDeviceMemoryInfoInMb(int device, size_t* total, size_t* free);
int getDeviceCount();
bool hasGpus();

} // namespace cuda_util
} // namespace astdl

#define REQUIRE_GPU                                                                                                    \
    if (!astdl::cuda_util::hasGpus())                                                                                  \
        return;
#define REQUIRE_GPUS(numGpus)                                                                                          \
    if (astdl::cuda_util::getDeviceCount() < numGpus)                                                                  \
        return;

#endif
