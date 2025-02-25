

#include "unfusedAttentionKernels_2_template.h"

namespace suggestify
{
namespace kernels
{

#ifdef ENABLE_BF16
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(__nv_bfloat16, int8_t, KVBlockArray);
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(__nv_bfloat16, int8_t, KVLinearBuffer);
#endif

} // namespace kernels
} // namespace suggestify
