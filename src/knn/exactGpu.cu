#include "cudautil.h"
#include "data.h"
#include "exactGpu.h"
#include "mathUtil.h"
#include "output.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace astdl
{
namespace knn
{

class GpuContext {
public:
    GpuContext(int device, int collectionRows, int collectionCols, int maxK, DataType dataType)
        : device(device), 
          collectionRows(collectionRows),
          collectionCols(collectionCols),
          maxK(maxK),
          dataType(dataType)
    {
        CHECK_ERR(cudaSetDevice(device));
        CHECK_ERR(cudaMallocManaged(&dInputBatchBuffer.data, dInputBatchBuffer.getSizeInBytes()));
        CHECK_ERR(cudaMallocManaged(&dProducts.data, dProducts.getSizeInBytes()));
        CHECK_ERR(cudaMallocManaged(&dResultScores.data, dResultScores.getSizeInBytes()));
        CHECK_ERR(cudaMallocManaged(&dResultIndexes.data, dResultIndexes.getSizeInBytes()));
        CHECK_ERR(cudaMallocPitch(&dCollectionPartition.data, &dCollectionPartition.pitch, 
                             dCollectionPartition.getSizeInBytes(), dCollectionPartition.numRows));

        CHECK_ERR(cudaStreamCreate(&stream));
        CHECK_ERR(cublasCreate(&handle));
        CHECK_ERR(cudaEventCreate(&startEvent));
        CHECK_ERR(cudaEventCreate(&stopEvent));

        CHECK_ERR(cudaMallocManaged(&dHeapScores, heapSize * sizeof(float)));
        CHECK_ERR(cudaMallocManaged(&dHeapIndexes, heapSize * sizeof(uint32_t)));

        elapsedTopK = 0.0f;
        elapsedSgemm = 0.0f;
    }

    ~GpuContext()
    {
        CHECK_ERR(cudaFree(dInputBatchBuffer.data));
        CHECK_ERR(cudaFree(dProducts.data));
        CHECK_ERR(cudaFree(dResultScores.data));
        CHECK_ERR(cudaFree(dResultIndexes.data));
        CHECK_ERR(cudaFree(dCollectionPartition.data));
        CHECK_ERR(cudaStreamDestroy(stream));
        CHECK_ERR(cublasDestroy(handle));
        CHECK_ERR(cudaEventDestroy(startEvent));
        CHECK_ERR(cudaEventDestroy(stopEvent));
        CHECK_ERR(cudaFree(dHeapScores));
        CHECK_ERR(cudaFree(dHeapIndexes));
    }

    void performMatrixMultiplication(int batchSize)
    {
        static constexpr cublasOperation_t transa = CUBLAS_OP_N;
        static constexpr cublasOperation_t transb = CUBLAS_OP_N;
        static constexpr float alpha = 1.0f;
        static constexpr float beta = 0.0f;

        cudaDataType aType, bType, cType = CUDA_R_32F;
        if (dataType == DataType::FP16)
        {
            aType = CUDA_R_16F;
            bType = CUDA_R_16F;
            CHECK_ERR(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dCollectionPartition.numRows, 
                              batchSize, dCollectionPartition.numColumns, &alpha, 
                              dCollectionPartition.data, aType, dCollectionPartition.numRows,
                              dInputBatchBuffer.data, bType, dInputBatchBuffer.numColumns, &beta,
                              dProducts.data, cType, dProducts.numRows,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP, 
                              cudaStream_t(stream))); 
        }
        else if (dataType == DataType::FP32)
        {
            aType = CUDA_R_32F;
            bType = CUDA_R_32F;
            CHECK_ERR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dCollectionPartition.numRows, 
                              batchSize, dCollectionPartition.numColumns, &alpha, 
                              dCollectionPartition.data, dCollectionPartition.numRows,
                              dInputBatchBuffer.data, dInputBatchBuffer.numColumns, &beta,
                              dProducts.data, dProducts.numRows, 
                              cudaStream_t(stream)));
        }

        CHECK_ERR(cudaEventRecord(startEvent, stream));
        CHECK_ERR(cudaEventRecord(stopEvent, stream));
        CHECK_ERR(cudaStreamSynchronize(stream));

        float elapsedTime;
        CHECK_ERR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
        elapsedSgemm = elapsedTime;
    }

    void performTopK(int batchSize)
    {
        const int blockDim = 256;
        const int numBlocks = (batchSize + blockDim - 1) / blockDim;

        CHECK_ERR(cudaEventRecord(startEvent, stream));
        topKShared<<<numBlocks, blockDim>>>(static_cast<float*>(dProducts.data), 
                                    dResultScores.data, dResultIndexes.data, 
                                    batchSize, dProducts.numColumns, 
                                    collectionRows, maxK);
        CHECK_ERR(cudaDeviceSynchronize());
        CHECK_ERR(cudaEventRecord(stopEvent, stream));
        CHECK_ERR(cudaStreamSynchronize(stream));

        float elapsedTime;
        CHECK_ERR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
        elapsedTopK = elapsedTime;
    }

    template <typename T>
    void copyInputToDevice(const T* inputs, size_t inputSize)
    {
        if (dataType == DataType::FP16)
        {
            astdl::math::kFloatToHalf(inputs, inputSize * sizeof(T),
                static_cast<half*>(dInputBatchBuffer.data),
                static_cast<float*>(dInputBatchBuffer.data),
                dInputBatchBuffer.getSizeInBytes());
        }
        else if (dataType == DataType::FP32)
        {
        }
        else
        {
            throw std::runtime_error("Unknown data type");
        }
    }

    void copyResultsToHost(float* hResultScores, uint32_t* hResultIndexes, int batchSize)
    {
        CHECK_ERR(cudaMemcpy(hResultScores, dResultScores.data, batchSize * maxK * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_ERR(cudaMemcpy(hResultIndexes, dResultIndexes.data, batchSize * maxK * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    float getElapsedSgemm() const { return elapsedSgemm; }
    float getElapsedTopK() const { return elapsedTopK; }

    int device; 
    int collectionRows;
    int collectionCols;
    int maxK;
    DataType dataType;

    cublasHandle_t handle;
    Matrix dCollectionPartition; 
    Matrix dInputBatchBuffer;
    Matrix dProducts;
    Matrix dResultScores;
    Matrix dResultIndexes;
    cudaStream_t stream; 
    cudaEvent_t startEvent, stopEvent;

    float* dHeapScores;
    uint32_t* dHeapIndexes;
    size_t heapSize = 256;

    float elapsedTopK;
    float elapsedSgemm;
};

class KnnExactGpu {
public:
    KnnExactGpu(KnnData* data) : data(data)
    {
        gpuContexts.resize(data->numGpus);
        for (int device = 0; device < data->numGpus; ++device)
        {
            gpuContexts[device] = std::make_unique<GpuContext>(
                device, data->dCollectionPartition.numRows / data->numGpus + (device < data->dCollectionPartition.numRows % data->numGpus ? 1 : 0),
                data->dCollectionPartition.numColumns, data->maxK, data->dataType);

            CHECK_ERR(cudaMemcpy(gpuContexts[device]->dCollectionPartition.data, data->hCollectionPartitions[0].data + device * (data->dCollectionPartition.numRows / data->numGpus) * data->dCollectionPartition.numColumns, 
                                 gpuContexts[device]->dCollectionPartition.getSizeInBytes(), cudaMemcpyHostToDevice));
        }

        if (usePinnedMemory)
        {
            CHECK_ERR(cudaHostAlloc(&pinnedInputData, data->dCollectionPartition.numColumns * data->batchSize * sizeof(float), cudaHostAllocDefault));
        }
    }

    ~KnnExactGpu() 
    {
        if (usePinnedMemory)
        {
            CHECK_ERR(cudaFreeHost(pinnedInputData));
        }
    }

    void search(int k, float const* inputs, int size, std::string* keys, float* scores)
    {
        if (k > data->maxK)
        {
            throw std::invalid_argument("k = " + std::to_string(k) + " is > maxK = " + std::to_string(data->maxK));
        }

        elapsedTopK.resize(data->numGpus);
        elapsedSgemm.resize(data->numGpus);

        int batchSize = data->batchSize; 

        for (int device = 0; device < data->numGpus; ++device)
        {
            gpuContexts[device]->copyInputToDevice(inputs, batchSize * data->dCollectionPartition.numColumns);
            gpuContexts[device]->performMatrixMultiplication(batchSize);
            gpuContexts[device]->performTopK(batchSize);

            elapsedSgemm[device] = gpuContexts[device]->getElapsedSgemm();
            elapsedTopK[device] = gpuContexts[device]->getElapsedTopK();
        }

        float avgSgemmTime = 0.0f;
        for (int device = 0; device < data->numGpus; ++device)
        {
            avgSgemmTime += elapsedSgemm[device];
        }
        avgSgemmTime /= data->numGpus; 

        if (avgSgemmTime > 0.01f)
        {
            batchSize = std::max(1, batchSize / 2);
        } 

        for (int i = 0; i < size; i += batchSize)
        {
            int currentBatchSize = std::min(batchSize, size - i);

            if (usePinnedMemory)
            {
                CHECK_ERR(cudaMemcpy(pinnedInputData, inputs + i * data->dCollectionPartition.numColumns,
                                     currentBatchSize * data->dCollectionPartition.numColumns * sizeof(float), cudaMemcpyHostToDevice));
            }

            for (int device = 0; device < data->numGpus; ++device)
            {
                if (usePinnedMemory)
                {
                    CHECK_ERR(cudaMemcpyAsync(gpuContexts[device]->dInputBatchBuffer.data, pinnedInputData,
                                         currentBatchSize * data->dCollectionPartition.numColumns * sizeof(float), cudaMemcpyDeviceToDevice, gpuContexts[device]->stream));
                }
                else
                {
                    gpuContexts[device]->copyInputToDevice(inputs + i * data->dCollectionPartition.numColumns,
                                                     currentBatchSize * data->dCollectionPartition.numColumns);
                }

                gpuContexts[device]->performMatrixMultiplication(currentBatchSize);
                gpuContexts[device]->performTopK(currentBatchSize);

                gpuContexts[device]->copyResultsToHost(data->hResultScores.data + i * k, data->hResultIndexes.data + i * k, currentBatchSize);
            }

            for (int device = 0; device < data->numGpus; ++device)
            {
                CHECK_ERR(cudaStreamSynchronize(gpuContexts[device]->stream));
            }

            mergeResults(data, currentBatchSize, k);

            CHECK_ERR(cudaMemcpy(scores + i * k, data->hResultScores.data, currentBatchSize * k * sizeof(float), cudaMemcpyHostToDevice));
        }

        for (int device = 0; device < data->numGpus; ++device)
        {
            std::cout << "GPU " << device << ": "
                      << "Sgemm Time: " << elapsedSgemm[device] << "ms"
                      << " Top-K Time: " << elapsedTopK[device] << "ms" << std::endl;
        }
    }

private:
    void mergeResults(KnnData* data, int batchSize, int maxK)
    {
        const size_t blockDim = 256;
        const size_t numBlocks = (data->numGpus + blockDim - 1) / blockDim;

        thrust::device_vector<float> dSharedScores(data->numGpus * maxK * batchSize);
        thrust::device_vector<uint32_t> dSharedIndexes(data->numGpus * maxK * batchSize);

        #pragma omp parallel for
        for (int device = 0; device < data->numGpus; ++device)
        {
            CHECK_ERR(cudaMemcpyAsync(thrust::raw_pointer_cast(dSharedScores.data()) + device * maxK * batchSize, 
                                   gpuContexts[device]->dResultScores.data, 
                                   maxK * batchSize * sizeof(float), cudaMemcpyDeviceToDevice, gpuContexts[device]->stream));
            CHECK_ERR(cudaMemcpyAsync(thrust::raw_pointer_cast(dSharedIndexes.data()) + device * maxK * batchSize, 
                                   gpuContexts[device]->dResultIndexes.data, 
                                   maxK * batchSize * sizeof(uint32_t), cudaMemcpyDeviceToDevice, gpuContexts[device]->stream));
            CHECK_ERR(cudaStreamSynchronize(gpuContexts[device]->stream));
        }

        for (int i = 0; i < batchSize; ++i)
        {
            thrust::device_ptr<float> dScoresPtr = thrust::raw_pointer_cast(dSharedScores.data()) + i * data->numGpus * maxK;
            thrust::device_ptr<uint32_t> dIndexesPtr = thrust::raw_pointer_cast(dSharedIndexes.data()) + i * data->numGpus * maxK;
            thrust::sort_by_key(dScoresPtr, dScoresPtr + data->numGpus * maxK, dIndexesPtr);
            thrust::copy(dScoresPtr, dScoresPtr + maxK, data->hResultScores.data + i * maxK);
            thrust::copy(dIndexesPtr, dIndexesPtr + maxK, data->hResultIndexes.data + i * maxK);
        }
    }

    KnnData* data;

    std::vector<std::unique_ptr<GpuContext>> gpuContexts;

    std::vector<float> elapsedTopK;
    std::vector<float> elapsedSgemm;


    float* pinnedInputData = nullptr; 
    bool usePinnedMemory = true;
};
}
}