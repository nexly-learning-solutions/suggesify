

#include "../chunkstate.h"

namespace suggestify
{
namespace kernels
{

GetChunkStateKernelFunc getChunkStateKernel_bf16 = getChunkStateKernel<bf16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
