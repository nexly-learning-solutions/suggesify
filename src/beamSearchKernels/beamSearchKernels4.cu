

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{
INSTANTIATE_BEAM_SEARCH(float, 4, false);
INSTANTIATE_BEAM_SEARCH(float, 4, true);
INSTANTIATE_BEAM_SEARCH(half, 4, false);
INSTANTIATE_BEAM_SEARCH(half, 4, true);
}
}
