

#include "fp8_rowwise_gemm_template.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_BF16
template class CutlassFp8RowwiseGemmRunner<__nv_bfloat16>;
#endif
} // namespace cutlass_kernels
} // namespace kernels
} // namespace suggestify
