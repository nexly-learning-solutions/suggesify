
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

namespace suggestify::common
{

std::pair<std::unordered_set<int32_t>, std::unordered_set<int32_t>> populateIterationIndexes(
    std::string const& envVarName, std::optional<std::string> const& legacyEnvVarName = std::nullopt);

}
