#pragma once

#include <vector>
#include <string>
#include <memory>

#include "../gpuTypes.h"
#include "../types.h"

class FilterConfig;
class Network;

class recommendations
{
    std::unique_ptr<GpuBuffer<float>> pbKey;
    std::unique_ptr<GpuBuffer<unsigned int>> pbUIValue;
    std::unique_ptr<GpuBuffer<float>> pFilteredOutput;
    std::vector<GpuBuffer<float>*> *vNodeFilters;
    std::string sugesstifyGenLayerLabel;
    std::string scorePrecision;
    
public:
    static const std::string DEFAULT_LAYER_sugesstify_GEN_LABEL;
    static const unsigned int TOPK_SCALAR;
    static const std::string DEFAULT_SCORE_PRECISION;

    recommendations(unsigned int xBatchSize,
                    unsigned int xK, 
                    unsigned int xOutputBufferSize,
                    const std::string &layer = DEFAULT_LAYER_sugesstify_GEN_LABEL,
                    const std::string &precision = DEFAULT_SCORE_PRECISION);

    void generatesugesstify(Network *network,
                      unsigned int topK,
                      const FilterConfig *filters,
                      const std::vector<std::string> &customerIndex,
                      const std::vector<std::string> &featureIndex);
};
