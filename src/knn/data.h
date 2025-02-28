#ifndef DATA_H_
#define DATA_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "DataReader.h"

namespace astdl
{
namespace knn
{

enum class DataType
{
  FP32 = 0, FP16 = 1
};

std::string getDataTypeString(DataType dataType);

DataType getDataTypeFromString(const std::string &dataTypeLiteral);

struct Matrix
{
    void *data;
    uint32_t numRows;
    int numColumns;
    size_t elementSize;
    cudaMemoryType memoryType;

    Matrix();

    Matrix(void* data, uint32_t numRows, int numColumns, size_t elementSize, cudaMemoryType memoryType);

    size_t getSizeInBytes();

    size_t getLength();

};

Matrix loadDataOnHost(DataReader *dataReader);

struct KnnData
{
    const int numGpus;
    const int batchSize;
    const int maxK;

    std::vector<cublasHandle_t> cublasHandles;
    std::vector<Matrix> dCollectionPartitions;
    std::vector<Matrix> dInputBatches;
    std::vector<Matrix> dProducts;
    std::vector<Matrix> dResultScores;
    std::vector<Matrix> dResultIndexes;
    std::vector<Matrix> hResultScores;
    std::vector<Matrix> hResultIndexes;
    std::vector<std::vector<std::string>> hKeys;
    std::vector<Matrix> dInputBatchTmpBuffers;
    std::vector<uint32_t> collectionRowsPadded;
    std::vector<float> elapsedSgemm;
    std::vector<float> elapsedTopK;


    const DataType dataType;

    KnnData(int numGpus, int batchSize, int maxK, DataType dataType);

    void load(int device, DataReader *dataReader);

    void load(const std::map<int, DataReader*> &deviceToData);

    void load(const std::map<int, std::string> &deviceToFile, char keyValDelim, char vecDelim);

    int getFeatureSize() const;

    ~KnnData();
};

Matrix allocateMatrixOnHost(uint32_t numRows, int numColumns, size_t elementSize);

Matrix allocateMatrixOnDevice(uint32_t numRows, int numColumns, size_t elementSize);

void freeMatrix(const Matrix &matrix);

}
}

#endif
