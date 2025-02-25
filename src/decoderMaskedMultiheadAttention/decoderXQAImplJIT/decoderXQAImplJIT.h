
#pragma once
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImpl.h"

#include "compileEngine.h"
#include "cubinObjRegistry.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include <unordered_set>

namespace suggestify
{
namespace kernels
{

class DecoderXQAImplJIT : public DecoderXQAImpl
{
public:
    DecoderXQAImplJIT(DecoderXQARunner* runner);

    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin) override;
    void prepare(XQAParams const& xqaParams) override;

protected:
    void runWithKVLinearBuffer(
        XQAParams const& xqaParams, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream) override;
    void runWithKVBlockArray(
        XQAParams const& xqaParams, KVBlockArray const& kv_block_array, cudaStream_t const& stream) override;

private:
    std::shared_ptr<suggestify::common::CUDADriverWrapper> mDriver;

    //! Whether DecoderXQAImplJIT supports xqaParams.
    bool supportConfig(XQAParams const& xqaParams, bool forConfigurePlugin) const;
    //! Whether DecoderXQAImplJIT has perf gain over the default (non-XQA-optimized) implementation.
    bool mayHavePerfGain(XQAParams const& xqaParams) const;

    void prepareForActualXQAParams(XQAParams const& xqaParams);

    template <typename T, typename KVCacheBuffer>
    void runImpl(XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer, int multiprocessor_count,
        cudaStream_t const& stream);

    template <typename KVCacheBuffer>
    void runDispatchKVCacheBuffer(
        XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    bool mForceXQA;
    int mSM;

    jit::CubinObjKey getCubinObjKeyFromXQAParams(XQAParams const& xqaParams) const;
};

} // namespace kernels
} // namespace suggestify
