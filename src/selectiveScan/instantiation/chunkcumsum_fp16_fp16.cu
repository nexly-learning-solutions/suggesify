

#include "../chunkcumsum.h"

namespace suggestify
{
namespace kernels
{

GetChunkCumsumKernelFunc getChunkCumsumKernel_fp16_fp16 = getChunkCumsumKernel<fp16_t, fp16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
