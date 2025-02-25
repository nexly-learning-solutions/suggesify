

#include "../chunkscan.h"

namespace suggestify
{
namespace kernels
{

GetChunkScanKernelFunc getChunkScanKernel_fp16_fp16 = getChunkScanKernel<fp16_t, fp16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
