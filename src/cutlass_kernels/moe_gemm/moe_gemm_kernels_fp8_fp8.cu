

#include "../src/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h"

namespace suggestify
{
#ifdef ENABLE_FP8
template class MoeGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
#endif
// template class MoeGemmRunner<__nv_fp8_e5m2, __nv_fp8_e5m2>;
#endif
} // namespace suggestify
