

#include "../decoderMaskedMultiheadAttentionLaunch.h"

namespace suggestify
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 224;
} // namespace

namespace mmha
{

#ifndef FAST_BUILD // skip mmha_224 for fast build
INSTANTIATE_MMHA_LAUNCHERS(uint16_t, kSizePerHead)
#endif             // FAST_BUILD

} // namespace mmha

} // namespace kernels
} // namespace suggestify
