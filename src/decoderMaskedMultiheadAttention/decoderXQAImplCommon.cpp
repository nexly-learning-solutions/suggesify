
#include "../src/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"

namespace suggestify
{
namespace kernels
{

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQAParams const& xqaParams)
{
    unsigned int head_size = xqaParams.head_size;
    unsigned int num_q_heads = xqaParams.num_q_heads;
    unsigned int num_kv_heads = xqaParams.num_kv_heads;
    CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
    unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
    unsigned int beam_width = xqaParams.beam_width;

    // Use mTileSize = 16 kernels when qSeqLen <= 16.
    unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
    unsigned int mTileSize = qSeqLen <= 16 ? 16 : 32;
    // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
    unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
    // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
    unsigned int kernel_m_tilesize = xqaParams.multi_query_tokens ? mTileSize : num_q_heads_over_kv;

    return {xqaParams.kv_cache_data_type, head_size, beam_width, kernel_num_q_heads_over_kv, kernel_m_tilesize,
        xqaParams.paged_kv_cache ? static_cast<unsigned int>(xqaParams.tokens_per_block) : 0, xqaParams.paged_kv_cache,
        xqaParams.multi_query_tokens};
}

} // namespace kernels
} // namespace suggestify
