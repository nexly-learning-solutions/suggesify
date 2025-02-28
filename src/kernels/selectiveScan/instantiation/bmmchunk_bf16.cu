

#include "../bmmchunk.h"

namespace suggestify
{
namespace kernels
{

GetBmmChunkKernelFunc getBmmChunkKernel_bf16 = getBmmChunkKernel<bf16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
