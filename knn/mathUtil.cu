#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace astdl
{
    namespace math
    {

    constexpr int THREADS_PER_BLOCK = 128;

    inline void checkCudaError(cudaError_t result)
    {
        if (result != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(result));
        }
    }

    __global__ void kFloatToHalf_kernel(float const* src, size_t length, half* dst)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            dst[idx] = __float2half(src[idx]);
        }
    }

    void kFloatToHalf(float const* hSource, size_t sourceSizeInBytes, half* dDest, size_t bufferSizeInBytes)
    {
        if (sourceSizeInBytes % sizeof(float) != 0)
        {
            throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(float)");
        }

        if (bufferSizeInBytes % sizeof(float) != 0)
        {
            throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
        }

        float* dBuffer;
        checkCudaError(cudaMalloc(&dBuffer, bufferSizeInBytes));

        auto launchKernel = [=](size_t length)
        {
            dim3 threads(THREADS_PER_BLOCK);
            dim3 blocks((length + (threads.x - 1)) / threads.x);
            kFloatToHalf_kernel<<<blocks, threads>>>(dBuffer, length, dDest);
            checkCudaError(cudaGetLastError());
        };

        size_t srcLeftBytes = sourceSizeInBytes;
        size_t offset = 0;

        while (srcLeftBytes > 0)
        {
            size_t cpyBytes = std::min(srcLeftBytes, bufferSizeInBytes);
            size_t cpyLength = cpyBytes / sizeof(float);

            checkCudaError(cudaMemcpy(dBuffer, hSource + offset, cpyBytes, cudaMemcpyHostToDevice));
            launchKernel(cpyLength);

            offset += cpyLength;
            srcLeftBytes -= cpyBytes;
        }

        checkCudaError(cudaFree(dBuffer));
    }

    __global__ void kHalfToFloat_kernel(half const* src, size_t length, float* dst)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length)
        {
            dst[idx] = __half2float(src[idx]);
        }
    }

    void kHalfToFloat(half const* dSource, size_t sourceSizeInBytes, float* hDest, size_t bufferSizeInBytes)
    {
        if (sourceSizeInBytes % sizeof(half) != 0)
        {
            throw std::invalid_argument("sourceSizeInBytes must be divisible by sizeof(half)");
        }

        if (bufferSizeInBytes % sizeof(float) != 0)
        {
            throw std::invalid_argument("bufferSizeInBytes must be divisible by sizeof(float)");
        }

        float* dBuffer;
        size_t bufferLen = bufferSizeInBytes / sizeof(float);
        checkCudaError(cudaMalloc(&dBuffer, bufferLen * sizeof(float)));

        auto launchKernel = [=](size_t length)
        {
            dim3 threads(THREADS_PER_BLOCK);
            dim3 blocks((length + (threads.x - 1)) / threads.x);
            kHalfToFloat_kernel<<<blocks, threads>>>(dSource, length, dBuffer);
            checkCudaError(cudaGetLastError());
        };

        size_t sourceLength = sourceSizeInBytes / sizeof(half);
        size_t srcLeftBytes = sourceLength * sizeof(float);
        size_t offset = 0;

        while (srcLeftBytes > 0)
        {
            size_t cpyBytes = std::min(srcLeftBytes, bufferSizeInBytes);
            size_t cpyLength = cpyBytes / sizeof(float);

            launchKernel(cpyLength);
            checkCudaError(cudaMemcpy(hDest + offset, dBuffer, cpyBytes, cudaMemcpyDeviceToHost));

            offset += cpyLength;
            srcLeftBytes -= cpyBytes;
        }

        checkCudaError(cudaFree(dBuffer));
    }

    }
}