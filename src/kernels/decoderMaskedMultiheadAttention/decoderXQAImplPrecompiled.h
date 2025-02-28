
#pragma once
#include "../src/decoderMaskedMultiheadAttention/decoderXQAImpl.h"

namespace suggestify
{
namespace kernels
{

class DecoderXQAImplPrecompiled : public DecoderXQAImpl
{
public:
    DecoderXQAImplPrecompiled(DecoderXQARunner* runner)
        : DecoderXQAImpl(runner)
    {
    }

    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin) override;
    void prepare(XQAParams const& xqa_params) override;

protected:
    void runWithKVLinearBuffer(
        XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream) override;
    void runWithKVBlockArray(
        XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream) override;

private:
    template <typename KVCacheBuffer>
    void runDispatchBuffer(
        XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);
};

} // namespace kernels
} // namespace suggestify
