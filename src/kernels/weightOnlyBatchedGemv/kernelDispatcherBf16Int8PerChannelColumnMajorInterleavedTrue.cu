

#include "../src/weightOnlyBatchedGemv/kernelDispatcher.h"

namespace sugesstify
{
namespace kernels
{
namespace weight_only
{
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true, 64);
} // namespace weight_only
} // namespace kernels
} // namespace sugesstify
