#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cudautil.h"

namespace astdl
{

namespace
{

constexpr int kBytesInMb = 1024 * 1024;
constexpr auto kMemoryUpdateInterval = std::chrono::milliseconds(1000);
constexpr auto kMemoryInfoTimeout = std::chrono::seconds(5);

struct MemoryInfo
{
    int device;
    size_t free;
    size_t total;
    std::chrono::steady_clock::time_point lastUpdate;

    MemoryInfo()
        : device(0)
        , free(0)
        , total(0)
        , lastUpdate(std::chrono::steady_clock::now())
    {
        CHECK_ERR(cudaGetDevice(&device));
        CHECK_ERR(cudaMemGetInfo(&free, &total));
    }

    MemoryInfo(int device)
        : device(device)
        , free(0)
        , total(0)
        , lastUpdate(std::chrono::steady_clock::now())
    {
        CHECK_ERR(cudaSetDevice(device));
        CHECK_ERR(cudaMemGetInfo(&free, &total));
    }

    [[nodiscard]] std::string getFormattedInfo() const
    {
        auto freeMb = static_cast<long>(free / kBytesInMb);
        auto usedMb = static_cast<long>((total - free) / kBytesInMb);
        auto totalMb = static_cast<long>(total / kBytesInMb);
        return "GPU [" + std::to_string(device) + "] Mem Used: " + std::to_string(usedMb)
            + " MB. Free: " + std::to_string(freeMb) + " MB. Total: " + std::to_string(totalMb) + " MB";
    }

    void update()
    {
        CHECK_ERR(cudaSetDevice(device));
        CHECK_ERR(cudaMemGetInfo(&free, &total));
        lastUpdate = std::chrono::steady_clock::now();
    }
};

std::mutex memoryInfoMutex;
std::condition_variable memoryInfoCv;
std::map<int, MemoryInfo> deviceInfos;
std::atomic<bool> memoryInfoThreadRunning{false};

void updateMemoryInfoThread()
{
    memoryInfoThreadRunning = true;
    while (memoryInfoThreadRunning)
    {
        std::unique_lock<std::mutex> lock(memoryInfoMutex);
        int const deviceCount = cuda_util::getDeviceCount();
        for (int i = 0; i < deviceCount; ++i)
        {
            deviceInfos[i].update();
        }
        memoryInfoCv.notify_all();
        lock.unlock();
        std::this_thread::sleep_for(kMemoryUpdateInterval);
    }
}

auto getMemoryInfoForAllDevices() -> std::map<int, MemoryInfo>
{
    std::unique_lock<std::mutex> lock(memoryInfoMutex);
    if (memoryInfoCv.wait_for(lock, kMemoryInfoTimeout, [] { return !deviceInfos.empty(); }))
    {
        return deviceInfos;
    }
    else
    {
        std::cerr << "Timeout waiting for memory info update.\n";
        return {};
    }
}

void printMemInfoInternal(char const* header, std::map<int, MemoryInfo> const& infos)
{
    std::cout << "--" << header << std::endl;
    for (auto const& [device, info] : infos)
    {
        std::cout << "  " << info.getFormattedInfo() << std::endl;
    }
}

void checkMemoryThreshold(std::map<int, MemoryInfo> const& infos, size_t thresholdInMb)
{
    for (auto const& [device, info] : infos)
    {
        if (info.free / kBytesInMb < thresholdInMb)
        {
            std::cerr << "Warning: GPU [" << device << "] has less than " << thresholdInMb
                      << " MB of free memory. Consider freeing up some memory." << std::endl;
        }
    }
}

}

namespace cuda_util
{

void printMemInfo(char const* header)
{
    auto deviceInfos = getMemoryInfoForAllDevices();
    printMemInfoInternal(header, deviceInfos);
}

void printMemInfoForDevice(char const* header, int device)
{
    std::unique_lock<std::mutex> lock(memoryInfoMutex);
    if (memoryInfoCv.wait_for(lock, kMemoryInfoTimeout, [&device] { return deviceInfos.count(device) > 0; }))
    {
        std::cout << "--" << header << " " << deviceInfos[device].getFormattedInfo() << std::endl;
    }
    else
    {
        std::cerr << "Timeout waiting for memory info for device " << device << ".\n";
    }
}

void getDeviceMemoryInfoInMb(int device, size_t* total, size_t* free)
{
    CHECK_ERR(cudaGetDevice(&device));

    size_t freeInBytes, totalInBytes;
    CHECK_ERR(cudaMemGetInfo(&freeInBytes, &totalInBytes));

    *total = totalInBytes / kBytesInMb;
    *free = freeInBytes / kBytesInMb;
}

[[nodiscard]] int getDeviceCount()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "** ERROR (" << err << " - " << cudaGetErrorString(err) << ") calling cudaGetDeviceCount()."
                  << " The host probably does not have any GPUs."
                  << " Returning -1\n";
        return -1;
    }
    return deviceCount;
}

[[nodiscard]] bool hasGpus()
{
    return getDeviceCount() > 0;
}

void printMemInfoAndCheckThreshold(char const* header, size_t thresholdInMb)
{
    auto deviceInfos = getMemoryInfoForAllDevices();
    printMemInfoInternal(header, deviceInfos);
    checkMemoryThreshold(deviceInfos, thresholdInMb);
}

}

}