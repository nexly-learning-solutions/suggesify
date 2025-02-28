

#include "../src/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{
#ifdef ENABLE_FP8
template class CutlassFpAIntBGemmRunner<__nv_fp8_e4m3,        /*Activation Type*/
    cutlass::uint4b_t,                                        /*Weight Type*/
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, /*Scale and Zero Type*/
    __nv_bfloat16,                                            /*Bias type Type*/
    __nv_bfloat16                                             /*Output type Type*/
    >;
#endif
} // namespace cutlass_kernels
} // namespace kernels
} // namespace suggestify
