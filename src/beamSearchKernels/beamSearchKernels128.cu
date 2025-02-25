

#include "beamSearchKernelsTemplate.h"

namespace suggestify
{
namespace kernels
{

#ifndef FAST_BUILD 
INSTANTIATE_BEAM_SEARCH(float, 128, true);
INSTANTIATE_BEAM_SEARCH(half, 128, true);
#endif

}
}
