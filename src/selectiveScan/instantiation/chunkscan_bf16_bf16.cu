#include "../chunkscan.h"

namespace suggestify
{
namespace kernels
{

GetChunkScanKernelFunc getChunkScanKernel_bf16_bf16 = getChunkScanKernel<bf16_t, bf16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
