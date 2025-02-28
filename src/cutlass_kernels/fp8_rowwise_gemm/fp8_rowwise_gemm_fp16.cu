

#include "fp8_rowwise_gemm_template.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

template class CutlassFp8RowwiseGemmRunner<half>;

} // namespace cutlass_kernels
} // namespace kernels
} // namespace suggestify
