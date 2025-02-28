
#pragma once

#include "../common/assert.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace suggestify::kernels
{

class KVCacheIndex
{
public:
    using UnderlyingType = std::int32_t;

    static constexpr UnderlyingType kSecondaryPoolFlag = static_cast<UnderlyingType>(1)
        << (8 * sizeof(UnderlyingType) - 1);

    explicit KVCacheIndex(UnderlyingType value, bool isSecondary = false)
        : value{isSecondary ? value | kSecondaryPoolFlag : value}
    {
        CHECK_DEBUG(value >= 0);
    }

    __host__ __device__ [[nodiscard]] UnderlyingType get() const
    {
        return value & (~kSecondaryPoolFlag);
    }

    __host__ __device__ [[nodiscard]] bool isPrimary() const
    {
        return (value & kSecondaryPoolFlag) == 0;
    }

private:
    UnderlyingType value;
};

}
