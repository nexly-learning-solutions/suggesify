
#include "../src/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{
template class CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>;
} // namespace cutlass_kernels
} // namespace kernels
} // namespace suggestify
