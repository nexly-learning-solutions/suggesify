

#include "../statepassing.h"

namespace suggestify
{
namespace kernels
{

GetStatePassingKernelFunc getStatePassingKernel_bf16 = getStatePassingKernel<bf16_t>;

} // namespace kernels
} // namespace suggestify

// vim: ts=2 sw=2 sts=2 et sta
