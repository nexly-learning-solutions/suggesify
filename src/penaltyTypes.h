
#pragma once

#include "../common/assert.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace suggestify
{
namespace kernels
{

enum class DecodingPenaltyType
{
    Temperature,
    Repetition,
    Presence,
    Frequency,
    MinLength,
};

inline std::pair<float, float> getLimitsPenalty(DecodingPenaltyType penaltyType)
{
    auto constexpr fltMax = std::numeric_limits<float>::max();
    auto constexpr fltMin = std::numeric_limits<float>::lowest();
    auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();

    switch (penaltyType)
    {
    case DecodingPenaltyType::Temperature: return std::make_pair(0.f, fltMax);
    case DecodingPenaltyType::Repetition: return std::make_pair(0.f, fltMax);
    case DecodingPenaltyType::Presence: return std::make_pair(fltMin, fltMax);
    case DecodingPenaltyType::Frequency: return std::make_pair(fltMin, fltMax);
    case DecodingPenaltyType::MinLength: return std::make_pair(-fltEpsilon, fltMax);
    }
    CHECK_WITH_INFO(false, "Unknown penalty type %d", static_cast<int32_t>(penaltyType));
    return std::make_pair(fltMin, fltMax);
}
}
}
