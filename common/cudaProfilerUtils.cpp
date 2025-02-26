
#include "cudaProfilerUtils.h"
#include "logger.h"
#include "stringUtils.h"
#include <cstdint>
#include <optional>

namespace
{

std::tuple<std::unordered_set<int32_t>, std::unordered_set<int32_t>> populateIterationIndexesImpl(
    std::string const& envVarName)
{
    auto envVarVal = std::getenv(envVarName.c_str());
    auto envVarValStr = std::string{envVarVal != nullptr ? envVarVal : ""};
    auto values = suggestify::common::str2set(envVarValStr, ',');
    std::unordered_set<int32_t> startSet;
    std::unordered_set<int32_t> endSet;
    for (std::string const& value : values)
    {
        size_t dashIdx = value.find("-");
        if (dashIdx != std::string::npos)
        {
            int32_t start = std::stoi(value.substr(0, dashIdx));
            startSet.insert(start);
            int32_t end = std::stoi(value.substr(dashIdx + 1));
            endSet.insert(end);
        }
        else
        {
            int32_t start_end = std::stoi(value);
            startSet.insert(start_end);
            endSet.insert(start_end);
        }
    }

    return std::make_pair(startSet, endSet);
}

}

namespace suggestify::common
{

std::pair<std::unordered_set<int32_t>, std::unordered_set<int32_t>> populateIterationIndexes(
    std::string const& envVarName, std::optional<std::string> const& legacyEnvVarName)
{
    auto [profileIterIdxs, stopIterIdxs] = populateIterationIndexesImpl(envVarName);

    if (legacyEnvVarName && profileIterIdxs.empty() && stopIterIdxs.empty())
    {
        std::tie(profileIterIdxs, stopIterIdxs) = populateIterationIndexesImpl(legacyEnvVarName.value());

        if (!profileIterIdxs.empty() || !stopIterIdxs.empty())
        {
            LOG_WARNING(
                "Using deprecated environment variable %s to specify cudaProfiler start and stop iterations. "
                "Please "
                "use %s "
                "instead.",
                legacyEnvVarName.value().c_str(), envVarName.c_str());
        }
    }

    return std::make_pair(profileIterIdxs, stopIterIdxs);
}

}
