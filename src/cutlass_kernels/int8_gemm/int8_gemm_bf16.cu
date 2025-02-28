
#include "../src/cutlass_kernels/int8_gemm/int8_gemm_template.h"

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

#ifdef ENABLE_BF16
template class CutlassInt8GemmRunner<__nv_bfloat16>;
#endif

} // namespace cutlass_kernels
} // namespace kernels
} // namespace suggestify
