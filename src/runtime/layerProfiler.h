
#pragma once

#include "common.h"
#include <vector>

#include <NvInfer.h>

namespace suggestify::runtime
{
struct LayerProfile
{
    std::string name;
    std::vector<float> timeMs;
};

class LayerProfiler : public nvinfer1::IProfiler
{

public:
    void reportLayerTime(char const* layerName, float timeMs) noexcept override;

    std::string getLayerProfile() noexcept;

private:
    [[nodiscard]] float getTotalTime() const noexcept;

    std::vector<LayerProfile> mLayers;
    std::vector<LayerProfile>::iterator mIterator{mLayers.begin()};
    int32_t mUpdatesCount{0};
};
}
