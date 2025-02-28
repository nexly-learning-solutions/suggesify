

#include "../src/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_BF16
template class CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>;
#endif
} // namespace cutlass_kernels
} // namespace kernels
} // namespace suggestify
