
#pragma once
#include <cstdint>
#include <optional>

namespace suggestify
{
namespace kernels
{
inline constexpr int kMinHistoryTokensPerBlock = 128;

inline constexpr float kEnableMinBlockFactor = 4.0;
inline constexpr int kTargetWaveFactor = 8;

// For multi-block mode. We reserve workspace for this amount of sub-sequences.
// This should be enough. Huge batch size may result in larger value, but for large batch size,
// multi-block mode is not useful. For llama v2 70b, 6000 results in ~12MB multi-block
// workspace, and is enough for > 10 waves.
inline constexpr int kXQA_MAX_NUM_SUB_SEQ = 6000;

} // namespace kernels
} // namespace suggestify
