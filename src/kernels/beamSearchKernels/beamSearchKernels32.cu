

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{

#ifndef FAST_BUILD 
INSTANTIATE_BEAM_SEARCH(float, 32, true);
INSTANTIATE_BEAM_SEARCH(half, 32, true);
#endif

}
}
