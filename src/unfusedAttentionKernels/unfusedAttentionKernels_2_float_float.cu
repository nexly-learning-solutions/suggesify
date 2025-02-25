

#include "unfusedAttentionKernels_2_template.h"

namespace suggestify
{
namespace kernels
{

INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(float, float, KVBlockArray);
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(float, float, KVLinearBuffer);

} // namespace kernels
} // namespace suggestify
