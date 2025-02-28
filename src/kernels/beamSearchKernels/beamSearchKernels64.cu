

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{

#ifndef FAST_BUILD 
INSTANTIATE_BEAM_SEARCH(float, 64, true);
INSTANTIATE_BEAM_SEARCH(half, 64, true);
#endif

}
}
