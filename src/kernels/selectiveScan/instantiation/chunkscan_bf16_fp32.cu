

#include "../chunkscan.h"

namespace suggestify
{
namespace kernels
{

GetChunkScanKernelFunc getChunkScanKernel_bf16_fp32 = getChunkScanKernel<bf16_t, fp32_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
