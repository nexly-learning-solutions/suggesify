#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{
INSTANTIATE_BEAM_SEARCH(float, 8, false);
INSTANTIATE_BEAM_SEARCH(float, 8, true);
INSTANTIATE_BEAM_SEARCH(half, 8, false);
INSTANTIATE_BEAM_SEARCH(half, 8, true);
}
}
