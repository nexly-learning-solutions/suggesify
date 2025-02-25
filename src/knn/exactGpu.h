#ifndef EXACT_GPU_H_
#define EXACT_GPU_H_

#include "data.h"

namespace astdl
{
namespace knn
{
class Knn
{
public:
    virtual void search(int k, float const* inputs, std::string* keys, float* scores) = 0;

    virtual ~Knn() {}

protected:
    KnnData* data;

    Knn(KnnData* data);
};

class KnnExactGpu : public Knn
{
public:
    KnnExactGpu(KnnData* data);

    void search(int k, float const* inputs, int size, std::string* keys, float* scores);

    void search(int k, float const* inputs, std::string* keys, float* scores)
    {
        search(k, inputs, data->batchSize, keys, scores);
    }
};

void mergeKnn(int k, int batchSize, int width, int numGpus, std::vector<float*> const& allScores,
    std::vector<uint32_t*> const& allIndexes, std::vector<std::vector<std::string>> const& allKeys, float* scores,
    std::string* keys);
}
}

#endif