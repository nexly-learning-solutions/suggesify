

#include "../chunkcumsum.h"

namespace suggestify
{
namespace kernels
{

GetChunkCumsumKernelFunc getChunkCumsumKernel_bf16_fp32 = getChunkCumsumKernel<bf16_t, fp32_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
