

#include "fused_gated_gemm_template.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{
template class CutlassFusedGatedGemmRunner<__nv_fp8_e4m3>;
} // namespace cutlass_kernels
} // namespace kernels
} // namespace suggestify
