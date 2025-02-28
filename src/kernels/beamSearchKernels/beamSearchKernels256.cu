

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{

#ifndef FAST_BUILD 
INSTANTIATE_BEAM_SEARCH(float, 256, true);
INSTANTIATE_BEAM_SEARCH(half, 256, true);
#endif

}
}
