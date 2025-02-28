

#include "../decoderMaskedMultiheadAttentionLaunch.h"

namespace suggestify
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 64;
} // namespace

namespace mmha
{

#ifdef ENABLE_BF16
INSTANTIATE_MMHA_LAUNCHERS_WITH_IMPLICIT_REL_ATTN_BIAS(__nv_bfloat16, kSizePerHead)
#endif

} // namespace mmha

} // namespace kernels
} // namespace suggestify
