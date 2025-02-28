
#pragma once

#include "../executor/executor.h"
#include "../runtime/common.h"
#include <optional>

namespace sugesstify::batch_manager
{

using runtime::SizeType32;

struct PeftCacheManagerConfig
{

    static float constexpr kDefaultDeviceCachePercent = 0.02;
    static size_t constexpr kDefaultHostCacheSize = 1024 * 1024 * 1024;

    explicit PeftCacheManagerConfig(SizeType32 numHostModuleLayer = 0, SizeType32 numDeviceModuleLayer = 0,
        SizeType32 optimalAdapterSize = 8, SizeType32 maxAdapterSize = 64, SizeType32 numPutWorkers = 1,
        SizeType32 numEnsureWorkers = 1, SizeType32 numCopyStreams = 1, SizeType32 maxPagesPerBlockHost = 24,
        SizeType32 maxPagesPerBlockDevice = 8, std::optional<float> deviceCachePercent = std::nullopt,
        std::optional<size_t> hostCacheSize = std::nullopt)
        : numHostModuleLayer(numHostModuleLayer)
        , numDeviceModuleLayer(numDeviceModuleLayer)
        , optimalAdapterSize(optimalAdapterSize)
        , maxAdapterSize(maxAdapterSize)
        , numPutWorkers(numPutWorkers)
        , numEnsureWorkers(numEnsureWorkers)
        , numCopyStreams(numCopyStreams)
        , maxPagesPerBlockHost(maxPagesPerBlockHost)
        , maxPagesPerBlockDevice(maxPagesPerBlockDevice)
        , deviceCachePercent(deviceCachePercent)
        , hostCacheSize(hostCacheSize)
    {
    }

    explicit PeftCacheManagerConfig(executor::PeftCacheConfig cfg)
        : numHostModuleLayer(cfg.getNumHostModuleLayer())
        , numDeviceModuleLayer(cfg.getNumDeviceModuleLayer())
        , optimalAdapterSize(cfg.getOptimalAdapterSize())
        , maxAdapterSize(cfg.getMaxAdapterSize())
        , numPutWorkers(cfg.getNumPutWorkers())
        , numEnsureWorkers(cfg.getNumEnsureWorkers())
        , numCopyStreams(cfg.getNumCopyStreams())
        , maxPagesPerBlockHost(cfg.getMaxPagesPerBlockHost())
        , maxPagesPerBlockDevice(cfg.getMaxPagesPerBlockDevice())
        , deviceCachePercent(cfg.getDeviceCachePercent())
        , hostCacheSize(cfg.getHostCacheSize())
    {
    }

    SizeType32 numHostModuleLayer;
    SizeType32 numDeviceModuleLayer;
    SizeType32 optimalAdapterSize;
    SizeType32 maxAdapterSize;
    SizeType32 numPutWorkers;
    SizeType32 numEnsureWorkers;
    SizeType32 numCopyStreams;
    SizeType32 maxPagesPerBlockHost;
    SizeType32 maxPagesPerBlockDevice;
    std::optional<float> deviceCachePercent;
    std::optional<size_t> hostCacheSize;
};
}
