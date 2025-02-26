#pragma once

#include "../gpuTypes.h"
#include "../layer.h"
#include "../types.h"
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class runtime
{
private:
    static int const ALL = -1;

    std::string const networkFilename;
    uint32_t const batchSize;
    uint32_t const maxK;

    std::map<std::string, std::unique_ptr<GpuBuffer<float>>> dOutputScores;
    std::map<std::string, std::unique_ptr<GpuBuffer<uint32_t>>> dOutputIndexes;

public:
    runtime(std::string const& networkFilename, uint32_t batchSize, int maxK = ALL);
    ~runtime();

    Network* getNetwork() const;

    void initInputLayerDataSets(std::vector<DataSetDescriptor> const& datasetDescriptors);

    GpuBuffer<float>* getOutputScoresBuffer(std::string const& layerName);
    GpuBuffer<uint32_t>* getOutputIndexesBuffer(std::string const& layerName);

    static runtime* fromPtr(long ptr)
    {
        runtime* dc = reinterpret_cast<runtime*>(ptr);
        if (dc == nullptr)
        {
            throw std::runtime_error("Cannot convert nullptr to runtime");
        }
        return dc;
    }
};
