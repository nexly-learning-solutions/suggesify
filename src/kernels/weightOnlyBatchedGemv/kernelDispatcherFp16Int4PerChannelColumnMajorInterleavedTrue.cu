

#include "../src/weightOnlyBatchedGemv/kernelDispatcher.h"

namespace sugesstify
{
namespace kernels
{
namespace weight_only
{
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int4PerChannel, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true, 64);
} // namespace weight_only
} // namespace kernels
} // namespace sugesstify
