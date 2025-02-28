#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "Types.h"

/// <summary>
/// Calculates the number of blocks needed to process a given size of data.
/// </summary>
/// <param name="size">The size of the data in bytes.</param>
/// <returns>The number of blocks needed to process the data.</returns>
uint32_t CalculateBlocks(uint64_t size);

/// <summary>
/// Returns the sign of a value.
/// </summary>
/// <typeparam name="T">The type of the value.</typeparam>
/// <param name="x">The value to get the sign of.</param>
/// <returns>1 if the value is positive, -1 if the value is negative, and 0 if the value is zero.</returns>
template<typename T> __device__ T sgn(T x) { return (x > 0) - (x < 0); }

/// <summary>
/// Sets the GPU data for the kernels.
/// </summary>
void SetKernelsGpuData();

/// <summary>
/// Gets the GPU data for the kernels.
/// </summary>
void GetKernelsGpuData();

/// <summary>
/// Sets the GPU data for the KLoss kernels.
/// </summary>
void SetKLossGpuData();

/// <summary>
/// Gets the GPU data for the KLoss kernels.
/// </summary>
void GetKLossGpuData();

/// <summary>
/// Sets the GPU data for the KActivation kernels.
/// </summary>
void SetKActivationGpuData();

/// <summary>
/// Gets the GPU data for the KActivation kernels.
/// </summary>
void GetKActivationGpuData();

/// <summary>
/// Sets the GPU data for the KDelta kernels.
/// </summary>
void SetKDeltaGpuData();

/// <summary>
/// Gets the GPU data for the KDelta kernels.
/// </summary>
void GetKDeltaGpuData();

/// <summary>
/// Scales and biases an array of floats.
/// </summary>
/// <param name="pData">The array of floats to scale and bias.</param>
/// <param name="size">The size of the array.</param>
/// <param name="scale">The scaling factor.</param>
/// <param name="bias">The bias value.</param>
void kScaleAndBias(float* pData, uint64_t size, float scale, float bias);

/// <summary>
/// Adds a bias to a unit array.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias">The bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kAddBias(float* pUnit, float* pBias, uint32_t stride, uint32_t batch);

/// <summary>
/// Adds two biases to a unit array.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias1">The first bias array.</param>
/// <param name="pBias2">The second bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kAddDualBias(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch);

/// <summary>
/// Adds three biases to a unit array.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias1">The first bias array.</param>
/// <param name="pBias2">The second bias array.</param>
/// <param name="pBias3">The third bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kAddTripleBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch);

/// <summary>
/// Adds four biases to a unit array.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias1">The first bias array.</param>
/// <param name="pBias2">The second bias array.</param>
/// <param name="pBias3">The third bias array.</param>
/// <param name="pBias4">The fourth bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kAddQuadBias(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit array and sets it to the value of the bias array.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias">The bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kClearUnit(float* pUnit, float* pBias, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit array and sets it to the value of the two bias arrays.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias1">The first bias array.</param>
/// <param name="pBias2">The second bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kClearDualSourceUnit(float* pUnit, float* pBias1, float* pBias2, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit array and sets it to the value of the three bias arrays.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias1">The first bias array.</param>
/// <param name="pBias2">The second bias array.</param>
/// <param name="pBias3">The third bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kClearTripleSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, uint32_t stride, uint32_t batch);

/// <summary>
/// Clears a unit array and sets it to the value of the four bias arrays.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="pBias1">The first bias array.</param>
/// <param name="pBias2">The second bias array.</param>
/// <param name="pBias3">The third bias array.</param>
/// <param name="pBias4">The fourth bias array.</param>
/// <param name="stride">The stride of the unit and bias arrays.</param>
/// <param name="batch">The batch size.</param>
void kClearQuadSourceUnit(float* pUnit, float* pBias1, float* pBias2, float* pBias3, float* pBias4, uint32_t stride, uint32_t batch);

/// <summary>
/// Updates the biases with the given delta values.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias array.</param>
/// <param name="pDelta">The delta values.</param>
/// <param name="pBias">The bias array.</param>
void kUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias);

/// <summary>
/// Invokes examples for a given key, value pair.
/// </summary>
/// <param name="pOutputKey">The output key array.</param>
/// <param name="pKey">The key array.</param>
/// <param name="pValue">The value array.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the key and value arrays.</param>
/// <param name="k">The number of examples to invoke.</param>
void invokeExamples(float* pOutputKey, float *pKey, uint32_t* pValue, uint32_t batch, uint32_t width, uint32_t k);

/// <summary>
/// Invokes examples for a given key, value pair.
/// </summary>
/// <param name="pOutputKey">The output key array.</param>
/// <param name="pOutputValue">The output value array.</param>
/// <param name="pKey">The key array.</param>
/// <param name="pValue">The value array.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the key and value arrays.</param>
/// <param name="k">The number of examples to invoke.</param>
void invokeExamples(float* pOutputKey, float* pOutputValue, float *pKey, float* pValue, uint32_t batch, uint32_t width, uint32_t k);

/// <summary>
/// Invokes examples for a given key, value pair.
/// </summary>
/// <param name="pOutputKey">The output key array.</param>
/// <param name="pOutputValue">The output value array.</param>
/// <param name="pKey">The key array.</param>
/// <param name="pValue">The value array.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the key and value arrays.</param>
/// <param name="k">The number of examples to invoke.</param>
void invokeExamples(float* pOutputKey, uint32_t* pOutputValue, float *pKey, uint32_t * pValue, uint32_t batch, uint32_t width, uint32_t k);

/// <summary>
/// Invokes k-sparse examples.
/// </summary>
/// <param name="pUnit">The unit array.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the unit array.</param>
/// <param name="kSparse">The number of sparse examples to invoke.</param>
void invokeKSparse(float* pUnit, uint32_t batch, uint32_t stride, uint32_t kSparse);

/// <summary>
/// Adds a scaled source buffer to a destination buffer.
/// </summary>
/// <param name="pDest">The destination buffer.</param>
/// <param name="pSrc">The source buffer.</param>
/// <param name="scale">The scaling factor.</param>
/// <param name="size">The size of the buffers.</param>
void kAddScaleBuffers(float* pDest, float* pSrc, float scale, uint64_t size);

/// <summary>
/// Adds a source buffer to a destination buffer.
/// </summary>
/// <param name="pDest">The destination buffer.</param>
/// <param name="pSrc">The source buffer.</param>
/// <param name="size">The size of the buffers.</param>
/// <param name="stream">The CUDA stream to use.</param>
void kAddBuffers(float* pDest, float* pSrc, uint64_t size, cudaStream_t stream = 0);

/// <summary>
/// Adds a source buffer to a destination buffer, handling 2D data.
/// </summary>
/// <param name="pDest">The destination buffer.</param>
/// <param name="dpitch">The pitch of the destination buffer.</param>
/// <param name="pSrc">The source buffer.</param>
/// <param name="spitch">The pitch of the source buffer.</param>
/// <param name="width">The width of the buffers.</param>
/// <param name="height">The height of the buffers.</param>
/// <param name="stream">The CUDA stream to use.</param>
void kAddBuffers2D(float* pDest, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);

/// <summary>
/// Copies data from a source buffer to a destination buffer, handling 2D data.
/// </summary>
/// <param name="pDest">The destination buffer.</param>
/// <param name="dpitch">The pitch of the destination buffer.</param>
/// <param name="pSrc">The source buffer.</param>
/// <param name="spitch">The pitch of the source buffer.</param>
/// <param name="width">The width of the buffers.</param>
/// <param name="height">The height of the buffers.</param>
/// <param name="stream">The CUDA stream to use.</param>
void kCopy2D(float* pDest, uint32_t dpitch, float* pSrc, uint32_t spitch, uint32_t width, uint32_t height, cudaStream_t stream = 0);
/// <summary>
/// Initializes the sorting process for a given number of items.
/// </summary>
/// <typeparam name="KeyType">The type of the key data.</typeparam>
/// <typeparam name="ValueType">The type of the value data.</typeparam>
/// <param name="items">The number of items to sort.</param>
/// <param name="pbKey">The GPU buffer containing the key data.</param>
/// <param name="pbValue">The GPU buffer containing the value data.</param>
/// <returns>The number of bytes allocated for temporary storage.</returns>
template<typename KeyType, typename ValueType> size_t kInitSort(uint32_t items, GpuBuffer<KeyType>* pbKey, GpuBuffer<ValueType>* pbValue);

/// <summary>
/// Sorts a given number of items using a radix sort algorithm.
/// </summary>
/// <typeparam name="KeyType">The type of the key data.</typeparam>
/// <typeparam name="ValueType">The type of the value data.</typeparam>
/// <param name="items">The number of items to sort.</param>
/// <param name="pKey0">The first key array.</param>
/// <param name="pKey1">The second key array.</param>
/// <param name="pValue0">The first value array.</param>
/// <param name="pValue1">The second value array.</param>
/// <param name="pTemp">The temporary storage buffer.</param>
/// <param name="tempBytes">The size of the temporary storage buffer.</param>
/// <returns>True if the sorting was successful, false otherwise.</returns>
template<typename KeyType, typename ValueType> bool kSort(uint32_t items, KeyType* pKey0, KeyType* pKey1, ValueType* pValue0, ValueType* pValue1, char* pTemp, size_t tempBytes);

/// <summary>
/// Loads an input unit from a dense data source.
/// </summary>
/// <typeparam name="T">The type of the data source.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pData">The data source.</param>
template<typename T> void kLoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData);

/// <summary>
/// Loads an input unit from a dense data source using an index array.
/// </summary>
/// <typeparam name="T">The type of the data source.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pData">The data source.</param>
template<typename T> void kLoadIndexedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData);

/// <summary>
/// Loads an input unit from a sparse data source.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
void kLoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Loads an input unit from a sparse data source using an index array.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
void kLoadIndexedSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Loads an input unit from a sparse data source with denoising.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pRandom">The random noise array.</param>
void kLoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom);

/// <summary>
/// Loads an input unit from a sparse data source with denoising using an index array.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pRandom">The random noise array.</param>
void kLoadIndexedSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom);

/// <summary>
/// Loads an input unit from a sparse data source with analog data.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
template<typename T> void kLoadSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Loads an input unit from a sparse data source with analog data using an index array.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
template<typename T> void kLoadIndexedSparseAnalogInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Loads an input unit from a sparse data source with analog data and denoising.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pRandom">The random noise array.</param>
template<typename T> void kLoadSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom);

/// <summary>
/// Loads an input unit from a sparse data source with analog data and denoising using an index array.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pRandom">The random noise array.</param>
template<typename T> void kLoadIndexedSparseAnalogDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom);

/// <summary>
/// Invokes a sparse Z function.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
void invokeSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta);

/// <summary>
/// Invokes a sparse Z function using an index array.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
void invokeIndexedSparseZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pUnit, float beta);
/// <summary>
/// Invokes a sparse Z function with analog data.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
template<typename T> void invokeSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta);

/// <summary>
/// Invokes a sparse Z function with analog data using an index array.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
template<typename T> void invokeIndexedSparseAnalogZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pUnit, float beta);

/// <summary>
/// Invokes a sparse Z function with denoising.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
void invokeSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Invokes a sparse Z function with denoising using an index array.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
void invokeIndexedSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Invokes a sparse Z function with analog data and denoising.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
template<typename T> void invokeSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Invokes a sparse Z function with analog data and denoising using an index array.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the input unit.</param>
/// <param name="pWeight">The weight array.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pUnit">The input unit array.</param>
/// <param name="beta">The beta value.</param>
template<typename T> void invokeIndexedSparseAnalogDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, float* pWeight, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, float* pUnit, float beta);

/// <summary>
/// Invokes a sparse transposed matrix operation.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
void invokeSparseTransposedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed matrix operation using an index array.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
void invokeIndexedSparseTransposedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed matrix operation with denoising.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
void invokeSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed matrix operation with denoising using an index array.
/// </summary>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
void invokeIndexedSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed weight gradient calculation.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="beta">The beta value.</param>
/// <param name="m">The number of rows in the matrix.</param>
/// <param name="n">The number of columns in the matrix.</param>
/// <param name="pSparseTransposedStart">The sparse transposed start array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pDelta">The delta array.</param>
/// <param name="pWeightGradient">The weight gradient array.</param>
void invokeSparseTransposedWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pDelta, float* pWeightGradient);

/// <summary>
/// Invokes a sparse transposed matrix operation with analog data.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
template<typename T> void invokeSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed matrix operation with analog data using an index array.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
template<typename T> void invokeIndexedSparseTransposedAnalogMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed matrix operation with analog data and denoising.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
template<typename T> void invokeSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed matrix operation with analog data and denoising using an index array.
/// </summary>
/// <typeparam name="T">The type of the analog data.</typeparam>
/// <param name="position">The position of the input unit.</param>
/// <param name="batch">The batch size.</param>
/// <param name="pIndex">The index array.</param>
/// <param name="pSparseStart">The sparse start array.</param>
/// <param name="pSparseEnd">The sparse end array.</param>
/// <param name="pSparseIndex">The sparse index array.</param>
/// <param name="pDataWeight">The sparse data weight array.</param>
/// <param name="pSparseData">The sparse analog data array.</param>
/// <param name="pRandom">The random noise array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
template<typename T> void invokeIndexedSparseTransposedAnalogDenoisedMatrix(uint32_t position, uint32_t batch, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, float* pRandom, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData);

/// <summary>
/// Invokes a sparse transposed weight gradient calculation with analog data.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="beta">The beta value.</param>
/// <param name="m">The number of rows in the matrix.</param>
/// <param name="n">The number of columns in the matrix.</param>
/// <param name="pSparseTransposedStart">The sparse transposed start array.</param>
/// <param name="pSparseTransposedEnd">The sparse transposed end array.</param>
/// <param name="pSparseTransposedIndex">The sparse transposed index array.</param>
/// <param name="pSparseTransposedData">The sparse transposed data array.</param>
/// <param name="pDelta">The delta array.</param>
/// <param name="pWeightGradient">The weight gradient array.</param>
void invokeSparseTransposedAnalogWeightGradient(float alpha, float beta, uint32_t m, uint32_t n, uint32_t* pSparseTransposedStart, uint32_t* pSparseTransposedEnd, uint32_t* pSparseTransposedIndex, float* pSparseTransposedData, float* pDelta, float* pWeightGradient);
/// <summary>
/// Calculates the L1 error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The L1 error.</returns>
template<typename T> float invokeL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the L1 error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The L1 error.</returns>
template<typename T> float invokeIndexedL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the L2 error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The L2 error.</returns>
template<typename T> float invokeL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the L2 error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The L2 error.</returns>
template<typename T> float invokeIndexedL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the L2 Hinge error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The L2 Hinge error.</returns>
template<typename T> float invokeL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the L2 Hinge error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The L2 Hinge error.</returns>
template<typename T> float invokeIndexedL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Cross Entropy error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Cross Entropy error.</returns>
template<typename T> float invokeCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Cross Entropy error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Cross Entropy error.</returns>
template<typename T> float invokeIndexedCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Scaled Marginal Cross Entropy error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Scaled Marginal Cross Entropy error.</returns>
template<typename T> float invokeScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Scaled Marginal Cross Entropy error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Scaled Marginal Cross Entropy error.</returns>
template<typename T> float invokeIndexedScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Multinomial Cross Entropy error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Multinomial Cross Entropy error.</returns>
template<typename T> float invokeMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Multinomial Cross Entropy error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Multinomial Cross Entropy error.</returns>
template<typename T> float invokeIndexedMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Multinomial Scaled Marginal Cross Entropy error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Multinomial Scaled Marginal Cross Entropy error.</returns>
template<typename T> float invokeMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Multinomial Scaled Marginal Cross Entropy error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Multinomial Scaled Marginal Cross Entropy error.</returns>
template<typename T> float invokeIndexedMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Hinge error between the given unit and data.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Hinge error.</returns>
template<typename T> float invokeHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, T* pData, float* pDataWeight);

/// <summary>
/// Calculates the Hinge error between the given unit and data, using a provided index.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="position">The position of the unit in the unit vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride of the data.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The index to use.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The Hinge error.</returns>
template<typename T> float invokeIndexedHingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, T* pData, float* pDataWeight);
/// <summary>
/// Invokes the sparse L1 error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse L1 error.</returns>
float invokeSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse L1 error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse L1 error.</returns>
float invokeIndexedSparseL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse L2 error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse L2 error.</returns>
float invokeSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse L2 error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse L2 error.</returns>
float invokeIndexedSparseL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse L2 hinge error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse L2 hinge error.</returns>
float invokeSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse L2 hinge error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse L2 hinge error.</returns>
float invokeIndexedSparseL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse cross-entropy error.</returns>
float invokeSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse cross-entropy error.</returns>
float invokeIndexedSparseCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse scaled marginal cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse scaled marginal cross-entropy error.</returns>
float invokeSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse scaled marginal cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse scaled marginal cross-entropy error.</returns>
float invokeIndexedSparseScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse multinomial cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The sparse multinomial cross-entropy error.</returns>
float invokeSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Invokes the indexed sparse multinomial cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The indexed sparse multinomial cross-entropy error.</returns>
float invokeIndexedSparseMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Invokes the sparse multinomial scaled marginal cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The sparse multinomial scaled marginal cross-entropy error.</returns>
float invokeSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Invokes the indexed sparse multinomial scaled marginal cross-entropy error calculation.
/// </summary>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <returns>The indexed sparse multinomial scaled marginal cross-entropy error.</returns>
float invokeIndexedSparseMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight);

/// <summary>
/// Invokes the sparse analog L1 error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse analog L1 error.</returns>
template<typename T> float invokeSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse analog L1 error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse analog L1 error.</returns>
template<typename T> float invokeIndexedSparseAnalogL1Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse analog L2 error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse analog L2 error.</returns>
template<typename T> float invokeSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse analog L2 error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse analog L2 error.</returns>
template<typename T> float invokeIndexedSparseAnalogL2Error(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse analog L2 hinge error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse analog L2 hinge error.</returns>
template<typename T> float invokeSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse analog L2 hinge error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse analog L2 hinge error.</returns>
template<typename T> float invokeIndexedSparseAnalogL2HingeError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse analog cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse analog cross-entropy error.</returns>
template<typename T> float invokeSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse analog cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse analog cross-entropy error.</returns>
template<typename T> float invokeIndexedSparseAnalogCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse analog scaled marginal cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse analog scaled marginal cross-entropy error.</returns>
template<typename T> float invokeSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse analog scaled marginal cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse analog scaled marginal cross-entropy error.</returns>
template<typename T> float invokeIndexedSparseAnalogScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse analog multinomial cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <returns>The sparse analog multinomial cross-entropy error.</returns>
template<typename T> float invokeSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Invokes the indexed sparse analog multinomial cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <returns>The indexed sparse analog multinomial cross-entropy error.</returns>
template<typename T> float invokeIndexedSparseAnalogMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Invokes the sparse analog multinomial scaled marginal cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <returns>The sparse analog multinomial scaled marginal cross-entropy error.</returns>
template<typename T> float invokeSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Invokes the indexed sparse analog multinomial scaled marginal cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <returns>The indexed sparse analog multinomial scaled marginal cross-entropy error.</returns>
template<typename T> float invokeIndexedSparseAnalogMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData);

/// <summary>
/// Invokes the sparse data scaled marginal cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The sparse data scaled marginal cross-entropy error.</returns>
template<typename T> float invokeSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse data scaled marginal cross-entropy error calculation.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <returns>The indexed sparse data scaled marginal cross-entropy error.</returns>
template<typename T> float invokeIndexedSparseDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);
/// <summary>
/// Invokes the regularization error calculation.
/// </summary>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="pWeight">The weight vector.</param>
/// <param name="size">The size of the weight vector.</param>
/// <returns>The regularization error.</returns>
float invokeRegularizationError(float lambda, float lambda1, float* pWeight, uint64_t size);

/// <summary>
/// Normalizes the weights.
/// </summary>
/// <param name="norm">The normalization value.</param>
/// <param name="outputStride">The output stride.</param>
/// <param name="inputStride">The input stride.</param>
/// <param name="pWeight">The weight vector.</param>
void kNormalizeWeights(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight);

/// <summary>
/// Calculates the magnitudes of the weights.
/// </summary>
/// <param name="outputStride">The output stride.</param>
/// <param name="inputStride">The input stride.</param>
/// <param name="pWeight">The weight vector.</param>
/// <param name="pMagnitude">The magnitude vector.</param>
void invokeWeightMagnitudes(uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude);

/// <summary>
/// Normalizes the magnitudes of the weights.
/// </summary>
/// <param name="norm">The normalization value.</param>
/// <param name="outputStride">The output stride.</param>
/// <param name="inputStride">The input stride.</param>
/// <param name="pWeight">The weight vector.</param>
/// <param name="pMagnitude">The magnitude vector.</param>
void kNormalizeWeightMagnitudes(float norm, uint32_t outputStride, uint32_t inputStride, float* pWeight, float* pMagnitude);

/// <summary>
/// Normalizes the deltas.
/// </summary>
/// <param name="norm">The normalization value.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pDelta">The delta vector.</param>
void kNormalizeDeltas(float norm, uint32_t batch, uint32_t stride, float* pDelta);

/// <summary>
/// Calculates the magnitudes of the deltas.
/// </summary>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pMagnitude">The magnitude vector.</param>
void invokeDeltaMagnitudes(uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude);

/// <summary>
/// Normalizes the magnitudes of the deltas.
/// </summary>
/// <param name="norm">The normalization value.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pMagnitude">The magnitude vector.</param>
void kNormalizeDeltaMagnitudes(float norm, uint32_t batch, uint32_t stride, float* pDelta, float* pMagnitude);

/// <summary>
/// Invokes the scaled biased dropout operation.
/// </summary>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pRandom">The random vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="p">The dropout probability.</param>
/// <param name="target">The target value.</param>
/// <param name="a">The scaling factor for the lower bound.</param>
/// <param name="b">The scaling factor for the upper bound.</param>
void invokeScaledBiasedDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target, float a, float b);

/// <summary>
/// Invokes the dropout operation.
/// </summary>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pRandom">The random vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="p">The dropout probability.</param>
/// <param name="target">The target value.</param>
void invokeDropout(float* pUnit, float* pRandom, uint32_t batch, uint32_t stride, float p, float target);
/// <summary>
/// Invokes the L1 output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed L1 output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the data.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeIndexedL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
template<typename T> void invokeCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);

/// <summary>
/// Invokes the indexed cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the data.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
template<typename T> void invokeIndexedCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Invokes the scaled marginal cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
template<typename T> void invokeScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);

/// <summary>
/// Invokes the indexed scaled marginal cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the data.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
template<typename T> void invokeIndexedScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);

/// <summary>
/// Invokes the output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the data.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeIndexedOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the L2 hinge output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed L2 hinge output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the data.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeIndexedL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the hinge output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
template<typename T> void invokeHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, T* pData, float* pDataWeight);

/// <summary>
/// Invokes the indexed hinge output delta function.
/// </summary>
/// <typeparam name="T">The data type of the data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the data.</param>
/// <param name="pData">The data.</param>
/// <param name="pDataWeight">The data weights.</param>
template<typename T> void invokeIndexedHingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, T* pData, float* pDataWeight);
/// <summary>
/// Invokes the sparse L1 output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed sparse L1 output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the sparse cross-entropy output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
void invokeSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse cross-entropy output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
void invokeIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse scaled marginal cross-entropy output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
void invokeSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse scaled marginal cross-entropy output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
void invokeIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed sparse output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the sparse L2 hinge output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed sparse L2 hinge output delta function.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeIndexedSparseL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
/// <summary>
/// Invokes the sparse L1 output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="scope">The scope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float scope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed sparse L1 output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="scope">The scope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeIndexedSparseL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float scope, float alpha, float lambda);

/// <summary>
/// Invokes the sparse cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
template<typename T> void invokeSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
template<typename T> void invokeIndexedSparseCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse scaled marginal cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
template<typename T> void invokeSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse scaled marginal cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
template<typename T> void invokeIndexedSparseScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed sparse output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeIndexedSparseOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t* pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the sparse data scaled marginal cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
template<typename T> void invokeSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the indexed sparse data scaled marginal cross-entropy output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
template<typename T> void invokeIndexedSparseDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit, float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, T* pSparseData, bool bSparseIgnoreZero);

/// <summary>
/// Invokes the sparse analog output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed sparse analog output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeIndexedSparseAnalogOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the sparse analog L2 hinge output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the indexed sparse analog L2 hinge output delta function.
/// </summary>
/// <typeparam name="T">The data type of the sparse data.</typeparam>
/// <param name="activation">The activation function.</param>
/// <param name="position">The position of the output.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pIndex">The indices of the sparse data.</param>
/// <param name="pSparseStart">The start indices of the sparse data.</param>
/// <param name="pSparseEnd">The end indices of the sparse data.</param>
/// <param name="pSparseIndex">The indices of the sparse data.</param>
/// <param name="pDataWeight">The data weights.</param>
/// <param name="pSparseData">The sparse data.</param>
/// <param name="bSparseIgnoreZero">Indicates whether to ignore zero values in the sparse data.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
template<typename T> void invokeIndexedSparseAnalogL2HingeOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, float* pUnit,  float* pDelta, uint32_t* pIndex, uint64_t* pSparseStart, uint64_t* pSparseEnd, uint32_t *pSparseIndex, float* pDataWeight, T* pSparseData, bool bSparseIgnoreZero, float slope, float alpha, float lambda);
/// <summary>
/// Invokes the sparseness penalty.
/// </summary>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="p">The p value.</param>
/// <param name="beta">The beta value.</param>
void invokeSparsenessPenalty(uint32_t batch,  uint32_t stride, float* pUnit, float* pDelta, float p, float beta);

/// <summary>
/// Invokes the Hadamard product.
/// </summary>
/// <param name="activation">The activation function.</param>
/// <param name="size">The size of the vector.</param>
/// <param name="scale">The scale factor.</param>
/// <param name="pUnit">The unit vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="slope">The slope value.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeHadamardProduct(Activation activation, uint64_t size, float scale, float* pUnit, float* pDelta, float slope, float alpha, float lambda);

/// <summary>
/// Invokes the sigmoid activation function.
/// </summary>
/// <param name="pData">The data vector.</param>
/// <param name="size">The size of the vector.</param>
void invokeSigmoidActivation(float* pData, uint64_t size);

/// <summary>
/// Invokes the tanh activation function.
/// </summary>
/// <param name="pData">The data vector.</param>
/// <param name="size">The size of the vector.</param>
void invokeTanhActivation(float* pData, uint64_t size);

/// <summary>
/// Invokes the ReLU activation function.
/// </summary>
/// <param name="pData">The data vector.</param>
/// <param name="size">The size of the vector.</param>
void invokeRELUActivation(float* pData, uint64_t size);

/// <summary>
/// Invokes the ELU activation function.
/// </summary>
/// <param name="pData">The data vector.</param>
/// <param name="size">The size of the vector.</param>
/// <param name="alpha">The alpha value.</param>
void invokeELUActivation(float* pData, uint64_t size, float alpha);

/// <summary>
/// Invokes the SELU activation function.
/// </summary>
/// <param name="pData">The data vector.</param>
/// <param name="size">The size of the vector.</param>
/// <param name="alpha">The alpha value.</param>
/// <param name="lambda">The lambda value.</param>
void invokeSELUActivation(float* pData, uint64_t size, float alpha, float lambda);

/// <summary>
/// Invokes the LReLU activation function.
/// </summary>
/// <param name="pData">The data vector.</param>
/// <param name="size">The size of the vector.</param>
/// <param name="slope">The slope value.</param>
void invokeLRELUActivation(float* pData, uint64_t size, float slope);

/// <summary>
/// Invokes the SoftMax activation function.
/// </summary>
/// <param name="pData">The data vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
void invokeSoftMaxActivation(float* pData, uint32_t batch, uint32_t stride);

/// <summary>
/// Updates the weights using the kSGD algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightGradient">The weight gradient vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kSGDUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightGradient, float* pWeight);

/// <summary>
/// Updates the biases using the kSGD algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pBias">The bias vector.</param>
void kSGDUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBias);

/// <summary>
/// Updates the weights using the kMomentum algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightVelocity">The weight velocity vector.</param>
/// <param name="pWeightGradient">The weight gradient vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kMomentumUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);

/// <summary>
/// Updates the biases using the kMomentum algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pBiasVelocity">The bias velocity vector.</param>
/// <param name="pBias">The bias vector.</param>
void kMomentumUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);

/// <summary>
/// Updates the weights using the kAdaGrad algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightVelocity">The weight velocity vector.</param>
/// <param name="pWeightGradient">The weight gradient vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kAdaGradUpdateWeights(float alpha, float lambda, float lambda1, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);

/// <summary>
/// Updates the biases using the kAdaGrad algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pBiasVelocity">The bias velocity vector.</param>
/// <param name="pBias">The bias vector.</param>
void kAdaGradUpdateBiases(float alpha, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);

/// <summary>
/// Shifts the weights using the kNesterov algorithm.
/// </summary>
/// <param name="mu">The momentum parameter.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightVelocity">The weight velocity vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kNesterovShiftWeights(float mu, uint64_t size, float* pWeightVelocity, float* pWeight);

/// <summary>
/// Shifts the biases using the kNesterov algorithm.
/// </summary>
/// <param name="mu">The momentum parameter.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pBiasVelocity">The bias velocity vector.</param>
/// <param name="pBias">The bias vector.</param>
void kNesterovShiftBiases(float mu, uint32_t width, float* pBiasVelocity, float* pBias);

/// <summary>
/// Updates the weights using the kNesterov algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightVelocity">The weight velocity vector.</param>
/// <param name="pWeightGradient">The weight gradient vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kNesterovUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);

/// <summary>
/// Updates the biases using the kNesterov algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pBiasVelocity">The bias velocity vector.</param>
/// <param name="pBias">The bias vector.</param>
void kNesterovUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);

/// <summary>
/// Updates the weights using the kRMSProp algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightVelocity">The weight velocity vector.</param>
/// <param name="pWeightGradient">The weight gradient vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kRMSPropUpdateWeights(float alpha, float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeight);

/// <summary>
/// Updates the biases using the kRMSProp algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pBiasVelocity">The bias velocity vector.</param>
/// <param name="pBias">The bias vector.</param>
void kRMSPropUpdateBiases(float alpha, float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBias);

/// <summary>
/// Updates the weights using the kAdaDelta algorithm.
/// </summary>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightVelocity">The weight velocity vector.</param>
/// <param name="pWeightGradient">The weight gradient vector.</param>
/// <param name="pWeightGradientVelocity">The weight gradient velocity vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kAdaDeltaUpdateWeights(float lambda, float lambda1, float mu, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight);

/// <summary>
/// Updates the biases using the kAdaDelta algorithm.
/// </summary>
/// <param name="mu">The momentum parameter.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pBiasVelocity">The bias velocity vector.</param>
/// <param name="pBiasGradientVelocity">The bias gradient velocity vector.</param>
/// <param name="pBias">The bias vector.</param>
void kAdaDeltaUpdateBiases(float mu, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias);

/// <summary>
/// Updates the weights using the kAdam algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="lambda">The L2 regularization parameter.</param>
/// <param name="lambda1">The L1 regularization parameter.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="mu1">The exponential decay rate for the moment estimates.</param>
/// <param name="t">The current timestep.</param>
/// <param name="size">The size of the weight vector.</param>
/// <param name="pWeightVelocity">The weight velocity vector.</param>
/// <param name="pWeightGradient">The weight gradient vector.</param>
/// <param name="pWeightGradientVelocity">The weight gradient velocity vector.</param>
/// <param name="pWeight">The weight vector.</param>
void kAdamUpdateWeights(float alpha, float lambda, float lambda1, float mu, float mu1, float t, uint64_t size, float* pWeightVelocity, float* pWeightGradient, float* pWeightGradientVelocity, float* pWeight);

/// <summary>
/// Updates the biases using the kAdam algorithm.
/// </summary>
/// <param name="alpha">The learning rate.</param>
/// <param name="mu">The momentum parameter.</param>
/// <param name="mu1">The exponential decay rate for the moment estimates.</param>
/// <param name="t">The current timestep.</param>
/// <param name="batch">The batch size.</param>
/// <param name="width">The width of the bias vector.</param>
/// <param name="pDelta">The delta vector.</param>
/// <param name="pBiasVelocity">The bias velocity vector.</param>
/// <param name="pBiasGradientVelocity">The bias gradient velocity vector.</param>
/// <param name="pBias">The bias vector.</param>
void kAdamUpdateBiases(float alpha, float mu, float mu1, float t, uint32_t batch, uint32_t width, float* pDelta, float* pBiasVelocity, float* pBiasGradientVelocity, float* pBias);

/// <summary>
/// Invokes the Maxout activation function.
/// </summary>
/// <param name="pSrc">The source vector.</param>
/// <param name="size">The size of the vector.</param>
/// <param name="pDst">The destination vector.</param>
void invokeMaxout(float* pSrc, size_t size, float* pDst);

/// <summary>
/// Invokes the cosine function.
/// </summary>
/// <param name="p0Vector">The first vector.</param>
/// <param name="pVector">The second vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pDPOut">The dot product output vector.</param>
/// <param name="pAOut">The A output vector.</param>
/// <param name="pBOut">The B output vector.</param>
/// <param name="outStride">The output stride.</param>
void invokeCosine(float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDPOut, float* pAOut, float* pBOut, uint32_t outStride);

/// <summary>
/// Invokes the dot product function.
/// </summary>
/// <param name="p0Vector">The first vector.</param>
/// <param name="pVector">The second vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pDPOut">The dot product output vector.</param>
/// <param name="outStride">The output stride.</param>
void invokeDotProduct(float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDPOut, uint32_t outStride);

/// <summary>
/// Invokes the Maxout delta function.
/// </summary>
/// <param name="pSrc">The source vector.</param>
/// <param name="pSrcDelta">The source delta vector.</param>
/// <param name="size">The size of the vector.</param>
/// <param name="beta">The beta value.</param>
/// <param name="pDst">The destination vector.</param>
/// <param name="pDstDelta">The destination delta vector.</param>
void invokeMaxoutDelta(float* pSrc, float* pSrcDelta, size_t size, float beta, float* pDst, float* pDstDelta);

/// <summary>
/// Invokes the dot product delta function.
/// </summary>
/// <param name="pDPDelta">The dot product delta vector.</param>
/// <param name="p0Vector">The first vector.</param>
/// <param name="pVector">The second vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pDelta0">The first delta vector.</param>
/// <param name="beta0">The first beta value.</param>
/// <param name="pDelta">The second delta vector.</param>
/// <param name="beta">The second beta value.</param>
/// <param name="inputStride">The input stride.</param>
void invokeDotProductDelta(float* pDPDelta, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride);

/// <summary>
/// Invokes the cosine delta function.
/// </summary>
/// <param name="pDPDelta">The dot product delta vector.</param>
/// <param name="pDP">The dot product vector.</param>
/// <param name="pA">The A vector.</param>
/// <param name="pB">The B vector.</param>
/// <param name="p0Vector">The first vector.</param>
/// <param name="pVector">The second vector.</param>
/// <param name="batch">The batch size.</param>
/// <param name="stride">The stride.</param>
/// <param name="pDelta0">The first delta vector.</param>
/// <param name="beta0">The first beta value.</param>
/// <param name="pDelta">The second delta vector.</param>
/// <param name="beta">The second beta value.</param>
/// <param name="inputStride">The input stride.</param>
void invokeCosineDelta(float* pDPDelta, float* pDP, float* pA, float* pB, float* p0Vector, float* pVector, uint32_t batch, uint32_t stride, float* pDelta0, float beta0, float* pDelta, float beta, uint32_t inputStride);

template <typename T>
/// <summary>
/// Invokes the ban repeat ngram operation.
/// </summary>
/// <typeparam name="T">The data type of the logits.</typeparam>
/// <param name="logits">The logits tensor.</param>
/// <param name="output_ids_buf">The output ids buffer.</param>
/// <param name="finished_buf">The finished buffer.</param>
/// <param name="parent_ids_buf">The parent ids buffer.</param>
/// <param name="batch_size">The batch size.</param>
/// <param name="local_batch_size">The local batch size.</param>
/// <param name="beam_width">The beam width.</param>
/// <param name="no_repeat_ngram_size_buf">The no repeat ngram size buffer.</param>
/// <param name="id_offset">The id offset.</param>
/// <param name="vocab_size_padded">The padded vocab size.</param>
/// <param name="step">The current step.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const bool* finished_buf, const int* parent_ids_buf,
    int batch_size, int local_batch_size, int beam_width, const int* no_repeat_ngram_size_buf, int id_offset,
    int vocab_size_padded, size_t step, cudaStream_t stream);

template <typename T, typename Idx>
/// <summary>
/// Invokes the look-up operation.
/// </summary>
/// <typeparam name="T">The data type of the output and weights.</typeparam>
/// <typeparam name="Idx">The data type of the input and batch size.</typeparam>
/// <param name="out">The output tensor.</param>
/// <param name="input">The input tensor.</param>
/// <param name="weight">The weight tensor.</param>
/// <param name="batch_size">The batch size.</param>
/// <param name="offset">The offset.</param>
/// <param name="size">The size.</param>
/// <param name="n_embed">The number of embeddings.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeLookUp(T* out, const Idx* input, const T* weight, const Idx batch_size, const Idx offset, const Idx size,
    const int n_embed, cudaStream_t stream = 0);

struct gatherTreeParam
{
    /// <summary>
    /// The beams tensor.
    /// </summary>
    int* beams = nullptr;
    /// <summary>
    /// The sequence lengths tensor.
    /// </summary>
    int* sequence_lengths = nullptr;
    /// <summary>
    /// The maximum sequence length for the final step.
    /// </summary>
    int max_sequence_length_final_step = 0;
    /// <summary>
    /// The input lengths tensor.
    /// </summary>
    const int* input_lengths = nullptr;
    /// <summary>
    /// The response input lengths tensor.
    /// </summary>
    int* response_input_lengths = nullptr;
    /// <summary>
    /// The maximum sequence length.
    /// </summary>
    int max_seq_len = 0;
    /// <summary>
    /// The batch size.
    /// </summary>
    int batch_size = 0;
    /// <summary>
    /// The beam width.
    /// </summary>
    int beam_width = 0;
    /// <summary>
    /// The step ids tensor.
    /// </summary>
    const int* step_ids = nullptr;
    /// <summary>
    /// The parent ids tensor.
    /// </summary>
    const int* parent_ids = nullptr;
    /// <summary>
    /// The end tokens tensor.
    /// </summary>
    const int* end_tokens = nullptr;
    /// <summary>
    /// The output ids tensor.
    /// </summary>
    int* output_ids = nullptr;
    /// <summary>
    /// The CUDA stream.
    /// </summary>
    cudaStream_t stream;
    /// <summary>
    /// The cumulative log probabilities tensor.
    /// </summary>
    float* cum_log_probs = nullptr;
    /// <summary>
    /// The length penalty.
    /// </summary>
    float length_penalty = 1.0f;
};

/// <summary>
/// Invokes the gather tree operation.
/// </summary>
/// <param name="param">The gather tree parameters.</param>
void invokeGatherTree(gatherTreeParam param);

/// <summary>
/// Invokes the finalize operation.
/// </summary>
/// <param name="output_ids">The output ids tensor.</param>
/// <param name="sequence_lengths">The sequence lengths tensor.</param>
/// <param name="cum_log_probs">The cumulative log probabilities tensor.</param>
/// <param name="output_log_probs">The output log probabilities tensor.</param>
/// <param name="topk_output_ids">The top-k output ids tensor.</param>
/// <param name="topk_sequence_lengths">The top-k sequence lengths tensor.</param>
/// <param name="scores">The scores tensor.</param>
/// <param name="topk_cum_log_probs">The top-k cumulative log probabilities tensor.</param>
/// <param name="topk_log_probs">The top-k log probabilities tensor.</param>
/// <param name="num_beams">The number of beams tensor.</param>
/// <param name="input_lengths">The input lengths tensor.</param>
/// <param name="beam_width">The beam width.</param>
/// <param name="max_seq_len">The maximum sequence length.</param>
/// <param name="batch_size">The batch size.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeFinalize(int* output_ids, int* sequence_lengths, float* cum_log_probs, float* output_log_probs,
    const int* topk_output_ids, const int* topk_sequence_lengths, const float* scores, const float* topk_cum_log_probs,
    const float* topk_log_probs, const int* num_beams, const int* input_lengths, const int beam_width,
    const int max_seq_len, const int batch_size, cudaStream_t stream);

/// <summary>
/// Invokes the initialize output operation.
/// </summary>
/// <param name="output_ids">The output ids tensor.</param>
/// <param name="end_ids">The end ids tensor.</param>
/// <param name="batch_beam">The batch beam size.</param>
/// <param name="max_seq_len">The maximum sequence length.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeInitializeOutput(int* output_ids, const int* end_ids, int batch_beam, int max_seq_len, cudaStream_t stream);

/// <summary>
/// Invokes the copy next step ids operation.
/// </summary>
/// <param name="next_step_ids">The next step ids tensor.</param>
/// <param name="output_ids_ptr">The output ids pointer.</param>
/// <param name="sequence_lengths">The sequence lengths tensor.</param>
/// <param name="batch_size">The batch size.</param>
/// <param name="beam_width">The beam width.</param>
/// <param name="max_seq_len">The maximum sequence length.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeCopyNextStepIds(int* next_step_ids, int** output_ids_ptr, const int* sequence_lengths, int batch_size,
    int beam_width, int max_seq_len, cudaStream_t stream);

template <typename T>
/// <summary>
/// Invokes the general layer normalization operation.
/// </summary>
/// <typeparam name="T">The data type of the input, output, gamma, and beta.</typeparam>
/// <param name="out">The output tensor.</param>
/// <param name="input">The input tensor.</param>
/// <param name="gamma">The gamma tensor.</param>
/// <param name="beta">The beta tensor.</param>
/// <param name="eps">The epsilon value.</param>
/// <param name="tokens">The number of tokens.</param>
/// <param name="hidden_dim">The hidden dimension.</param>
/// <param name="stream">The CUDA stream.</param>
/// <param name="use_diff_of_squares">Indicates whether to use the difference of squares.</param>
/// <param name="scale">The scale tensor.</param>
/// <param name="dynamic_scale">The dynamic scale tensor.</param>
/// <param name="out_quant">The quantized output tensor.</param>
void invokeGeneralLayerNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream = 0, bool use_diff_of_squares = true, const float* scale = nullptr,
    float* dynamic_scale = nullptr, int8_t* out_quant = nullptr);

template <typename T>
/// <summary>
/// Applies per-channel scaling to the input tensor.
/// </summary>
/// <typeparam name="T">The data type of the input and output tensors.</typeparam>
/// <param name="smoothed_act">The output tensor.</param>
/// <param name="act">The input tensor.</param>
/// <param name="per_channel_scale">The per-channel scaling factors tensor.</param>
/// <param name="rows">The number of rows in the tensor.</param>
/// <param name="cols">The number of columns in the tensor.</param>
/// <param name="stream">The CUDA stream.</param>
void apply_per_channel_scale_kernel_launcher(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0);

template <typename T>
/// <summary>
/// Invokes the quantization operation.
/// </summary>
/// <typeparam name="T">The data type of the source tensor.</typeparam>
/// <param name="dst">The destination quantized tensor.</param>
/// <param name="src">The source tensor.</param>
/// <param name="size">The size of the tensor.</param>
/// <param name="scalePtr">The scaling factor pointer.</param>
/// <param name="stream">The CUDA stream.</param>
/// <param name="maxGirdSize">The maximum grid size.</param>
void invokeQuantization(
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream = 0, int maxGirdSize = 0);

template <typename T>
/// <summary>
/// Invokes the per-token quantization operation.
/// </summary>
/// <typeparam name="T">The data type of the source tensor.</typeparam>
/// <param name="dst">The destination quantized tensor.</param>
/// <param name="src">The source tensor.</param>
/// <param name="numRows">The number of rows in the tensor.</param>
/// <param name="numCols">The number of columns in the tensor.</param>
/// <param name="scalePtr">The scaling factor pointer.</param>
/// <param name="stream">The CUDA stream.</param>
void invokePerTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, cudaStream_t stream = 0);

template <typename T>
/// <summary>
/// Invokes the general RMS normalization operation.
/// </summary>
/// <typeparam name="T">The data type of the input, output, gamma, and beta.</typeparam>
/// <param name="out">The output tensor.</param>
/// <param name="input">The input tensor.</param>
/// <param name="gamma">The gamma tensor.</param>
/// <param name="beta">The beta tensor.</param>
/// <param name="eps">The epsilon value.</param>
/// <param name="tokens">The number of tokens.</param>
/// <param name="hidden_dim">The hidden dimension.</param>
/// <param name="stream">The CUDA stream.</param>
/// <param name="scale">The scale tensor.</param>
/// <param name="dynamic_scale">The dynamic scale tensor.</param>
/// <param name="out_quant">The quantized output tensor.</param>
void invokeGeneralRmsNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps, const int tokens,
    const int hidden_dim, cudaStream_t stream = 0, const float* scale = nullptr, float* dynamic_scale = nullptr,
    int8_t* out_quant = nullptr);

/// <summary>
/// Invokes the stop words criterion operation.
/// </summary>
/// <param name="output_ids">The output ids buffer.</param>
/// <param name="parent_ids">The parent ids buffer.</param>
/// <param name="stop_words">The stop words tensor.</param>
/// <param name="finished">The finished buffer.</param>
/// <param name="sequence_lengths">The sequence lengths tensor.</param>
/// <param name="id_offset">The id offset.</param>
/// <param name="stop_words_len">The length of the stop words tensor.</param>
/// <param name="batch_size">The batch size.</param>
/// <param name="beam_width">The beam width.</param>
/// <param name="max_seq_len">The maximum sequence length.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeStopWordsCriterion(const int** output_ids, const int** parent_ids, const int* stop_words, bool* finished,
    const int* sequence_lengths, size_t id_offset, size_t stop_words_len, int batch_size, int beam_width,
    int max_seq_len, cudaStream_t stream);

/// <summary>
/// Invokes the length criterion operation.
/// </summary>
/// <param name="finished">The finished buffer.</param>
/// <param name="finished_sum">The finished sum tensor.</param>
/// <param name="sequence_limit_length">The sequence limit length tensor.</param>
/// <param name="sequence_lengths">The sequence lengths tensor.</param>
/// <param name="batch_size">The batch size.</param>
/// <param name="beam_width">The beam width.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeLengthCriterion(bool* finished, int* finished_sum, const uint32_t* sequence_limit_length,
    const int* sequence_lengths, int batch_size, int beam_width, cudaStream_t stream);

template <typename T>
/// <summary>
/// Invokes the ban bad words operation.
/// </summary>
/// <typeparam name="T">The data type of the logits.</typeparam>
/// <param name="logits">The logits tensor.</param>
/// <param name="output_ids_ptr">The output ids pointer.</param>
/// <param name="parent_ids_ptr">The parent ids pointer.</param>
/// <param name="batch_size">The batch size.</param>
/// <param name="local_batch_size">The local batch size.</param>
/// <param name="beam_width">The beam width.</param>
/// <param name="bad_words">The bad words tensor.</param>
/// <param name="share_words">Indicates whether to share words across batches.</param>
/// <param name="bad_words_len">The length of the bad words tensor.</param>
/// <param name="vocab_size_padded">The padded vocab size.</param>
/// <param name="sequence_lengths">The sequence lengths tensor.</param>
/// <param name="max_seq_len">The maximum sequence length.</param>
/// <param name="stream">The CUDA stream.</param>
void invokeBanBadWords(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
    int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
    int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream);


/// <summary>
/// Converts an integer to a 64-bit unsigned integer.
/// </summary>
/// <param name="l">The integer to convert.</param>
/// <returns>The converted 64-bit unsigned integer.</returns>
__device__ inline uint64_t llitoulli(int64_t l)
{
    uint64_t u;
    asm("mov.b64    %0, %1;" : "=l"(u) : "l"(l));
    return u;
}

/// <summary>
/// Converts a 64-bit unsigned integer to an integer.
/// </summary>
/// <param name="u">The 64-bit unsigned integer to convert.</param>
/// <returns>The converted integer.</returns>
__device__ inline int64_t ullitolli(uint64_t u)
{
    int64_t l;
    asm("mov.b64    %0, %1;" : "=l"(l) : "l"(u));
    return l;
}

#if (CUDA_VERSION >= 9000)
/// <summary>
/// Performs a shuffle operation on the given value.
/// </summary>
/// <param name="x">The value to shuffle.</param>
/// <param name="lane">The lane to shuffle from.</param>
/// <returns>The shuffled value.</returns>
#define SHFL(x, lane) __shfl_sync(0xffffffff, (x), (lane))
/// <summary>
/// Performs a ballot operation on the given predicate.
/// </summary>
/// <param name="predicate">The predicate to evaluate.</param>
/// <returns>The ballot result.</returns>
#define BALLOT(predicate) __ballot_sync(0xffffffff, (predicate))
/// <summary>
/// Checks if any thread in the warp satisfies the given predicate.
/// </summary>
/// <param name="predicate">The predicate to evaluate.</param>
/// <returns>True if any thread satisfies the predicate, false otherwise.</returns>
#define ANY(predicate) __any_sync(0xffffffff, (predicate))
#else
/// <summary>
/// Performs a shuffle operation on the given value.
/// </summary>
/// <param name="x">The value to shuffle.</param>
/// <param name="lane">The lane to shuffle from.</param>
/// <returns>The shuffled value.</returns>
#define SHFL(x, lane) __shfl((x), (lane))
/// <summary>
/// Performs a ballot operation on the given predicate.
/// </summary>
/// <param name="predicate">The predicate to evaluate.</param>
/// <returns>The ballot result.</returns>
#define BALLOT(predicate) __ballot(predicate)
/// <summary>
/// Checks if any thread in the warp satisfies the given predicate.
/// </summary>
/// <param name="predicate">The predicate to evaluate.</param>
/// <returns>True if any thread satisfies the predicate, false otherwise.</returns>
#define ANY(predicate) __any(predicate)
#endif


/// <summary>
/// Reduces the given error value across the warp.
/// </summary>
/// <param name="error">The error value to reduce.</param>
#define REDUCEERROR(error) \
    if (ANY(error != (float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        error                  += SHFL(error, tgx ^ 1); \
        error                  += SHFL(error, tgx ^ 2); \
        error                  += SHFL(error, tgx ^ 4); \
        error                  += SHFL(error, tgx ^ 8); \
        error                  += SHFL(error, tgx ^ 16); \
        if (tgx == 0) \
        { \
            atomicAdd(cData._pAccumulator, llitoulli(llrintf(ERRORSCALEF * error))); \
        } \
    }


/// <summary>
/// Reduces the given value across the warp.
/// </summary>
/// <param name="a">The value to reduce.</param>
#define REDUCE(a) \
    if (ANY((a) != (float)0.0)) \
    { \
        uint32_t tgx            = threadIdx.x & cData._warpMask; \
        a                      += SHFL((a), tgx ^ 1); \
        a                      += SHFL((a), tgx ^ 2); \
        a                      += SHFL((a), tgx ^ 4); \
        a                      += SHFL((a), tgx ^ 8); \
        a                      += SHFL((a), tgx ^ 16); \
    }


#endif
