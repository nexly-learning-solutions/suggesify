

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
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_LAUNCHERS(__nv_bfloat16, kSizePerHead)
#endif // ENABLE_BF16
#endif // FAST_BUILD

} // namespace mmha

} // namespace kernels
} // namespace suggestify
