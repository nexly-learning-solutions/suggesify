

#include "unfusedAttentionKernels_2_template.h"

namespace suggestify
{
namespace kernels
{

INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(float, __nv_fp8_e4m3, KVBlockArray);
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(float, __nv_fp8_e4m3, KVLinearBuffer);

} // namespace kernels
} // namespace suggestify
