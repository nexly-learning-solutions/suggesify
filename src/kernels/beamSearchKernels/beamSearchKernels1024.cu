

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{

#ifndef FAST_BUILD 
INSTANTIATE_BEAM_SEARCH(float, 1024, true);
INSTANTIATE_BEAM_SEARCH(half, 1024, true);
#endif

}
}
