

#include "unfusedAttentionKernels_2_template.h"

namespace suggestify
{
namespace kernels
{

INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(half, half, KVBlockArray);
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(half, half, KVLinearBuffer);

} // namespace kernels
} // namespace suggestify
