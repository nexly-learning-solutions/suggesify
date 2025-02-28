
#pragma once

#include "../runtime/common.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace suggestify
{
namespace layers
{

class DefaultDecodingParams
{
public:
    [[nodiscard]] __host__ __device__ static constexpr float getTemperature()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getRepetitionPenalty()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getPresencePenalty()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getFrequencyPenalty()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getMinLength()
    {
        return 1;
    }

    [[nodiscard]] __host__ __device__ static constexpr uint64_t getSeed()
    {
        return 0;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getTopK()
    {
        return 0;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopP()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopPDecay()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopPMin()
    {
        return 1.0e-6f;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::TokenIdType getTopPResetId()
    {
        return -1;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getBeamSearchDiversity()
    {
        return 0.f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getLengthPenalty()
    {
        return 0.f;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getEarlyStopping()
    {
        return 1;
    }

    [[nodiscard]] __host__ __device__ static constexpr bool getNormalizeLogProbs()
    {
        return false;
    }

    [[nodiscard]] static std::vector<runtime::SizeType32> getTopKMedusaHeads()
    {
        return {};
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getNoRepeatNgramSize()
    {
        return 1 << 30;
    }
};
}
}
