

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{
#ifndef FAST_BUILD
INSTANTIATE_BEAM_SEARCH(float, 16, true);
INSTANTIATE_BEAM_SEARCH(half, 16, true);
#endif
}
}
