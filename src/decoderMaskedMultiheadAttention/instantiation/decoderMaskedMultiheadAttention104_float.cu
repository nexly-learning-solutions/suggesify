

#include "../decoderMaskedMultiheadAttentionLaunch.h"

namespace suggestify
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 104;
} // namespace

namespace mmha
{

#ifndef FAST_BUILD // skip mmha_104 for fast build
INSTANTIATE_MMHA_LAUNCHERS(float, kSizePerHead)
#endif             // FAST_BUILD

} // namespace mmha

} // namespace kernels
} // namespace suggestify
