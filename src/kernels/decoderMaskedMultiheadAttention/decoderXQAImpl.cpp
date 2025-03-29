#include "../src/decoderMaskedMultiheadAttention/decoderXQAImpl.h"
#include "../src/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.h"
#include "../src/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.h"

#include <cassert>
#include <functional>
#include <memory>

namespace suggestify
{
namespace kernels
{

/// <summary>
/// Runs the DecoderXQAImpl using a KVLinearBuffer for key-value storage.
/// </summary>
/// <typeparam name="">No template parameters are explicitly used, but the implementation is specialized for this specific KVLinearBuffer use case.</typeparam>
/// <param name="xqa_params">Parameters for the XQA (decoder) operation.</param>
/// <param name="kv_linear_buffer">The key-value linear buffer containing the key and value data.</param>
/// <param name="stream">The CUDA stream to use for asynchronous execution.</param>
template <>
void DecoderXQAImpl::run(
    XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream)
{
    runWithKVLinearBuffer(xqa_params, kv_linear_buffer, stream);
}

/// <summary>
/// Runs the DecoderXQAImpl using a KVBlockArray for key-value storage.
/// </summary>
/// <typeparam name="">No template parameters are explicitly used, but the implementation is specialized for this specific KVBlockArray use case.</typeparam>
/// <param name="xqa_params">Parameters for the XQA (decoder) operation.</param>
/// <param name="kv_block_array">The key-value block array containing the key and value data.</param>
/// <param name="stream">The CUDA stream to use for asynchronous execution.</param>
template <>
void DecoderXQAImpl::run(XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream)
{
    runWithKVBlockArray(xqa_params, kv_block_array, stream);
}

/// <summary>
/// Creates a DecoderXQAImpl object of the specified implementation type.
/// </summary>
/// <param name="runner">Pointer to the decoder runner, providing context for the XQA operation.</param>
/// <param name="implType">The implementation type to create (Precompiled or JIT).</param>
/// <returns>A unique pointer to the created DecoderXQAImpl object.</returns>
std::unique_ptr<DecoderXQAImpl> DecoderXQAImpl::create(DecoderXQARunner* runner, ImplType implType)
{
    switch (implType)
    {
    case ImplType::kPrecompiled: return std::unique_ptr<DecoderXQAImpl>(new DecoderXQAImplPrecompiled(runner));
    case ImplType::kJIT: return std::unique_ptr<DecoderXQAImpl>(new DecoderXQAImplJIT(runner));
    }

    THROW("Unknown DecoderXQAImpl::ImplType");
}

} // namespace kernels
} // namespace suggestify