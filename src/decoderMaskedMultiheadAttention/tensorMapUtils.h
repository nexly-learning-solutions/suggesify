
#pragma once
#include "suggestify/common/cudaDriverWrapper.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/xqaParams.h"

namespace suggestify
{
namespace kernels
{

template <typename KVCacheBuffer>
CUtensorMap makeTensorMapForKVCache(std::shared_ptr<suggestify::common::CUDADriverWrapper> const& driver,
    XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer);

} // namespace kernels
} // namespace suggestify
