
#include "cudaMemPool.h"
#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"
#include <array>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>

namespace suggestify::runtime
{

CudaMemPool::CudaMemPool(cudaMemPool_t pool, int device)
    : mDevice{device}
{
    CHECK_WITH_INFO(pool != nullptr, "Pointer to cudaMemPool cannot be nullptr.");
    mPool = PoolPtr{pool, Deleter{}};
}

std::size_t CudaMemPool::memoryPoolReserved() const
{
    std::size_t reserved = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(mPool.get(), cudaMemPoolAttrReservedMemCurrent, &reserved));
    return reserved;
}

std::size_t CudaMemPool::memoryPoolUsed() const
{
    std::size_t used = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(mPool.get(), cudaMemPoolAttrUsedMemCurrent, &used));
    return used;
}

std::size_t CudaMemPool::memoryPoolFree() const
{
    return memoryPoolReserved() - memoryPoolUsed();
}

void CudaMemPool::memoryPoolTrimTo(std::size_t size)
{
    CUDA_CHECK(::cudaMemPoolTrimTo(mPool.get(), size));
}

cudaMemPool_t CudaMemPool::getPool() const
{
    return mPool.get();
}

bool CudaMemPool::supportsMemoryPool(int deviceId)
{
    int32_t value{};
    CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, deviceId));
    return value != 0;
}

void CudaMemPool::Deleter::operator()(cudaMemPool_t pool) const
{
    CUDA_CHECK_FREE_RESOURCE(::cudaMemPoolDestroy(pool));
    LOG_TRACE("Destroyed pool %p", pool);
}

namespace
{

std::shared_ptr<CudaMemPool> createPrimaryDevicePool(int deviceId)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    ::cudaMemPool_t memPool = nullptr;
    ::cudaMemPoolProps poolProps{};
    poolProps.allocType = ::cudaMemAllocationTypePinned;
    poolProps.handleTypes = ::cudaMemHandleTypeNone;
    poolProps.location.type = ::cudaMemLocationTypeDevice;
    poolProps.location.id = deviceId;
    CUDA_CHECK(::cudaMemPoolCreate(&memPool, &poolProps));
    auto maxThreshold = std::numeric_limits<std::uint64_t>::max();
    CUDA_CHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &maxThreshold));
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return std::make_shared<CudaMemPool>(memPool, deviceId);
}

constexpr size_t maxDevicePerNode = 64;

std::mutex primaryDevicePoolsMutex{};

std::array<std::shared_ptr<CudaMemPool>, maxDevicePerNode> primaryDevicePools{};

std::array<bool, maxDevicePerNode> primaryDevicePoolInitAttempted{};

}

std::shared_ptr<CudaMemPool> CudaMemPool::getPrimaryPoolForDevice(int deviceId)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (primaryDevicePoolInitAttempted.at(deviceId))
    {
        LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return primaryDevicePools.at(deviceId);
    }

    {
        std::lock_guard lockGuard{primaryDevicePoolsMutex};

        if (primaryDevicePoolInitAttempted.at(deviceId))
        {
            LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return primaryDevicePools.at(deviceId);
        }

        if (!CudaMemPool::supportsMemoryPool(deviceId))
        {
            primaryDevicePoolInitAttempted.at(deviceId) = true;
            LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return {};
        }

        try
        {
            primaryDevicePools.at(deviceId) = createPrimaryDevicePool(deviceId);
            primaryDevicePoolInitAttempted.at(deviceId) = true;
            LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return primaryDevicePools.at(deviceId);
        }
        catch (std::exception const& exception)
        {
            LOG_ERROR("Failed to initialized memory pool for device %i.", deviceId);
            LOG_EXCEPTION(exception);
            primaryDevicePoolInitAttempted.at(deviceId) = true;
            LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
            return {};
        }
    }
}

}
