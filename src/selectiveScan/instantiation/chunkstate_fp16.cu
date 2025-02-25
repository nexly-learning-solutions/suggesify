

#include "../chunkstate.h"

namespace suggestify
{
namespace kernels
{

GetChunkStateKernelFunc getChunkStateKernel_fp16 = getChunkStateKernel<fp16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
