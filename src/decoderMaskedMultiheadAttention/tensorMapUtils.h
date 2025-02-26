#pragma once
#include "cudaDriverWrapper.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/xqaParams.h"

namespace suggestify
{
namespace kernels
{
    /// <summary>
    /// Creates a tensor map for the given key-value cache buffer using CUDA driver.
    /// </summary>
    /// <typeparam name="KVCacheBuffer">The type of the key-value cache buffer.</typeparam>
    /// <param name="driver">The shared pointer to the CUDA driver wrapper responsible for managing the CUDA context.</param>
    /// <param name="xqaParams">The XQA parameters to configure the tensor map creation.</param>
    /// <param name="kv_cache_buffer">The key-value cache buffer used in the tensor map creation.</param>
    /// <returns>A tensor map for the provided key-value cache buffer.</returns>
    template <typename KVCacheBuffer>
    CUtensorMap makeTensorMapForKVCache(
        std::shared_ptr<suggestify::common::CUDADriverWrapper> const& driver,
        XQAParams const& xqaParams, 
        KVCacheBuffer const& kv_cache_buffer);

} // namespace kernels
} // namespace suggestify
