

#include "../src/weightOnlyBatchedGemv/kernelDispatcher.h"

namespace sugesstify
{
namespace kernels
{
namespace weight_only
{
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajor, false, 64);
} // namespace weight_only
} // namespace kernels
} // namespace sugesstify
