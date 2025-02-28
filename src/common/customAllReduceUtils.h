
#pragma once

#include <cstddef>

namespace suggestify::utils::customAllReduceUtils
{

constexpr size_t NUM_POINTERS_PER_RANK = 7;

inline size_t getMaxRequiredWorkspaceSize(int worldSize) noexcept
{
    if (worldSize <= 2)
    {
        return 16 * 1000 * 1000;
    }
    return 8 * 1000 * 1000;
}

}
