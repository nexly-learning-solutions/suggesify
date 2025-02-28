

#include "../src/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h"

namespace suggestify
{
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16>;
#endif
} // namespace suggestify
