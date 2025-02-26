
#pragma once

#include <memory>

struct CUmemPoolHandle_st;
using cudaMemPool_t = CUmemPoolHandle_st*;

namespace suggestify::runtime
{

class CudaMemPool
{
public:
    explicit CudaMemPool(cudaMemPool_t pool, int device);

    [[nodiscard]] std::size_t memoryPoolReserved() const;

    [[nodiscard]] std::size_t memoryPoolUsed() const;

    [[nodiscard]] std::size_t memoryPoolFree() const;

    void memoryPoolTrimTo(std::size_t size);

    [[nodiscard]] cudaMemPool_t getPool() const;

    static std::shared_ptr<suggestify::runtime::CudaMemPool> getPrimaryPoolForDevice(int deviceId);

    static bool supportsMemoryPool(int deviceId);

private:
    class Deleter
    {
    public:
        void operator()(cudaMemPool_t pool) const;
    };

    using PoolPtr = std::unique_ptr<std::remove_pointer_t<cudaMemPool_t>, Deleter>;

    PoolPtr mPool;
    int mDevice{-1};
};

}
