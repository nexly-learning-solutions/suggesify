

#include "../decoderMaskedMultiheadAttentionLaunch.h"

namespace suggestify
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 128;
} // namespace

namespace mmha
{

INSTANTIATE_MMHA_LAUNCHERS_WITH_IMPLICIT_REL_ATTN_BIAS(uint16_t, kSizePerHead)

} // namespace mmha

} // namespace kernels
} // namespace suggestify
