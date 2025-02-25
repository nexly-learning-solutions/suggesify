
#pragma once
#include "suggestify/kernels/decoderMaskedMultiheadAttention/xqaParams.h"

namespace suggestify
{
namespace kernels
{
namespace jit
{

bool supportConfigQGMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin);
bool supportConfigHMMA(XQAParams const& xqaParams, int SM, bool forConfigurePlugin);

} // namespace jit
} // namespace kernels
} // namespace suggestify
