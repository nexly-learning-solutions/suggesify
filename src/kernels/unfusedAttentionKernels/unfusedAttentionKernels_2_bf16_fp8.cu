#include "unfusedAttentionKernels_2_template.h"

namespace suggestify
{
namespace kernels
{

#ifdef ENABLE_BF16
/// <summary>
/// Instantiates the attention input/output processing templates for bfloat16 data type with fp8_e4m3.
/// This specific instantiation uses the KVBlockArray storage format.
/// </summary>
/// <typeparam name="TData">The data type of the input and output (e.g., __nv_bfloat16).</typeparam>
/// <typeparam name="TWeight">The data type of the weights (e.g., __nv_fp8_e4m3).</typeparam>
/// <typeparam name="TKVBlockArray">The storage format used for the key-value blocks.</typeparam>
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(__nv_bfloat16, __nv_fp8_e4m3, KVBlockArray);

/// <summary>
/// Instantiates the attention input/output processing templates for bfloat16 data type with fp8_e4m3.
/// This specific instantiation uses the KVLinearBuffer storage format.
/// </summary>
/// <typeparam name="TData">The data type of the input and output (e.g., __nv_bfloat16).</typeparam>
/// <typeparam name="TWeight">The data type of the weights (e.g., __nv_fp8_e4m3).</typeparam>
/// <typeparam name="TKVLinearBuffer">The storage format used for the key-value linear buffer.</typeparam>
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(__nv_bfloat16, __nv_fp8_e4m3, KVLinearBuffer);
#endif

} // namespace kernels
} // namespace suggestify