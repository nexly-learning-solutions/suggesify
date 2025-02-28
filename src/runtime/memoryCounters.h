
#pragma once

#include "../common/assert.h"
#include "iBuffer.h"

#include <atomic>
#include <cstddef>
#include <string>

namespace suggestify::runtime
{

class MemoryCounters
{
public:
    using SizeType32 = std::size_t;
    using DiffType = std::ptrdiff_t;

    MemoryCounters() = default;

    [[nodiscard]] SizeType32 getGpu() const
    {
        return mGpu;
    }

    [[nodiscard]] SizeType32 getCpu() const
    {
        return mCpu;
    }

    [[nodiscard]] SizeType32 getPinned() const
    {
        return mPinned;
    }

    [[nodiscard]] SizeType32 getUVM() const
    {
        return mUVM;
    }

    [[nodiscard]] SizeType32 getPinnedPool() const
    {
        return mPinnedPool;
    }

    [[nodiscard]] DiffType getGpuDiff() const
    {
        return mGpuDiff;
    }

    [[nodiscard]] DiffType getCpuDiff() const
    {
        return mCpuDiff;
    }

    [[nodiscard]] DiffType getPinnedDiff() const
    {
        return mPinnedDiff;
    }

    [[nodiscard]] DiffType getUVMDiff() const
    {
        return mUVMDiff;
    }

    [[nodiscard]] DiffType getPinnedPoolDiff() const
    {
        return mPinnedPoolDiff;
    }

    template <MemoryType T>
    void allocate(SizeType32 size)
    {
        auto const sizeDiff = static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU)
        {
            mGpu += size;
            mGpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kCPU)
        {
            mCpu += size;
            mCpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNED)
        {
            mPinned += size;
            mPinnedDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kUVM)
        {
            mUVM += size;
            mUVMDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNEDPOOL)
        {
            mPinnedPool += size;
            mPinnedPoolDiff = sizeDiff;
        }
        else
        {
            THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    void allocate(MemoryType memoryType, SizeType32 size);

    template <MemoryType T>
    void deallocate(SizeType32 size)
    {
        auto const sizeDiff = -static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU)
        {
            mGpu -= size;
            mGpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kCPU)
        {
            mCpu -= size;
            mCpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNED)
        {
            mPinned -= size;
            mPinnedDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kUVM)
        {
            mUVM -= size;
            mUVMDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNEDPOOL)
        {
            mPinnedPool -= size;
            mPinnedPoolDiff = sizeDiff;
        }
        else
        {
            THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    void deallocate(MemoryType memoryType, SizeType32 size);

    static MemoryCounters& getInstance();

    static std::string bytesToString(SizeType32 bytes, int precision = 2);

    static std::string bytesToString(DiffType bytes, int precision = 2);

    [[nodiscard]] std::string toString() const;

private:
    std::atomic<SizeType32> mGpu{}, mCpu{}, mPinned{}, mUVM{}, mPinnedPool{};
    std::atomic<DiffType> mGpuDiff{}, mCpuDiff{}, mPinnedDiff{}, mUVMDiff{}, mPinnedPoolDiff{};
};

}
