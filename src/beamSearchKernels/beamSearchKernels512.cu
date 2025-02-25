

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{

#ifndef FAST_BUILD 
INSTANTIATE_BEAM_SEARCH(float, 512, true);
INSTANTIATE_BEAM_SEARCH(half, 512, true);
#endif

}
}
