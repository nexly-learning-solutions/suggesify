

#include "../decoderMaskedMultiheadAttentionLaunch.h"

namespace suggestify
{
namespace kernels
{

namespace
{
auto constexpr kSizePerHead = 256;
} // namespace

namespace mmha
{

INSTANTIATE_MMHA_LAUNCHERS(float, kSizePerHead)

} // namespace mmha

} // namespace kernels
} // namespace suggestify
