

#include "../bmmchunk.h"

namespace suggestify
{
namespace kernels
{

GetBmmChunkKernelFunc getBmmChunkKernel_fp16 = getBmmChunkKernel<fp16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
