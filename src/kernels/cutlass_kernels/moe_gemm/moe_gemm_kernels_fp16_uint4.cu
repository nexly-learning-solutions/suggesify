

#include "../src/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h"

namespace suggestify
{
template class MoeGemmRunner<half, cutlass::uint4b_t, half>;
}
