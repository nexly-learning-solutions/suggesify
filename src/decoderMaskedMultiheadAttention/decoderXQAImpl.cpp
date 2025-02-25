
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImpl.h"

#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.h"

#include <cassert>
#include <functional>
#include <memory>

namespace suggestify
{
namespace kernels
{

template <>
void DecoderXQAImpl::run(
    XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream)
{
    runWithKVLinearBuffer(xqa_params, kv_linear_buffer, stream);
}

template <>
void DecoderXQAImpl::run(XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream)
{
    runWithKVBlockArray(xqa_params, kv_block_array, stream);
}

std::unique_ptr<DecoderXQAImpl> DecoderXQAImpl::create(DecoderXQARunner* runner, ImplType implType)
{
    switch (implType)
    {
    case ImplType::kPrecompiled: return std::unique_ptr<DecoderXQAImpl>(new DecoderXQAImplPrecompiled(runner));
    case ImplType::kJIT: return std::unique_ptr<DecoderXQAImpl>(new DecoderXQAImplJIT(runner));
    }
    // Shouldn't reach here.
    THROW("Unknown DecoderXQAImpl::ImplType");
}

} // namespace kernels
} // namespace suggestify
