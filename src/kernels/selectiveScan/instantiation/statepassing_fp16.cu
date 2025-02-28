

#include "../statepassing.h"

namespace suggestify
{
namespace kernels
{

GetStatePassingKernelFunc getStatePassingKernel_fp16 = getStatePassingKernel<fp16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
