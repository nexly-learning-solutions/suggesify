

#include "unfusedAttentionKernels_2_template.h"

namespace suggestify
{
namespace kernels
{

INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(half, int8_t, KVBlockArray);
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(half, int8_t, KVLinearBuffer);

} // namespace kernels
} // namespace suggestify
