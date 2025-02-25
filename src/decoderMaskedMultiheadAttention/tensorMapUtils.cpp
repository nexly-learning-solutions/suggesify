#include "suggestify/kernels/decoderMaskedMultiheadAttention/tensorMapUtils.h"
#include "suggestify/kernels/kvCacheUtils.h"

#include <cstdint>
#include <type_traits>

namespace suggestify::kernels
{

namespace
{

using suggestify::common::CUDADriverWrapper;

/// <summary>
/// Determines the number of bytes per element based on the specified data type.
/// </summary>
/// <param name="dataType">The data type for which to determine the byte size.</param>
/// <returns>The number of bytes per element for the given data type.</returns>
/// <exception cref="std::runtime_error">Thrown if the data type is unsupported.</exception>
uint32_t getElemBytes(CUtensorMapDataType_enum dataType)
{
    switch (dataType)
    {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8: return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_INT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_INT64: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return 4;
    }
    throw std::runtime_error("unsupported data type");
}

/// <summary>
/// Creates a tensor map for a paged key-value cache.
/// </summary>
/// <param name="driver">The shared pointer to the CUDA driver wrapper responsible for managing the CUDA context.</param>
/// <param name="addr">The address of the key-value cache data.</param>
/// <param name="dataType">The data type of the cache.</param>
/// <param name="headElems">The number of head elements in the cache.</param>
/// <param name="nbKHeads">The number of key heads in the cache.</param>
/// <param name="tokensPerPage">The number of tokens per page in the cache.</param>
/// <param name="nbTokensPerTile">The number of tokens per tile (optional, defaults to 64).</param>
/// <returns>A tensor map for the paged key-value cache.</returns>
CUtensorMap makeTensorMapForPagedKVCache(std::shared_ptr<CUDADriverWrapper> const& driver, void const* addr,
    CUtensorMapDataType_enum dataType, uint32_t headElems, uint32_t nbKHeads, uint32_t tokensPerPage,
    uint32_t nbTokensPerTile = 64)
{
    CUtensorMap tensorMap{};
    uint32_t elemBytes = getElemBytes(dataType);
    uint64_t const globalDims[] = {headElems, tokensPerPage, nbKHeads, 1U << 31};
    uint32_t const headBytes = elemBytes * headElems;
    uint64_t const globalStrides[] = {headBytes, headBytes * tokensPerPage, headBytes * tokensPerPage * nbKHeads};
    CHECK(headElems <= 256);
    uint32_t const paddedHeadElems = headElems <= 64 ? 64 : (headElems <= 128 ? 128 : 256);
    uint32_t const partElems = std::min(elemBytes * paddedHeadElems, 128U) / elemBytes;
    uint32_t const boxDims[] = {partElems, std::min(tokensPerPage, nbTokensPerTile), 1, 1};
    uint32_t const elemStrides[] = {1, 1, 1, 1};

    auto const swizzle = [&]
    {
        switch (partElems)
        {
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        default: THROW("unsupported cache head size");
        }
    }();

    CU_CHECK(driver->cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims,
        globalStrides, boxDims, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensorMap;
}

/// <summary>
/// Creates a tensor map for a contiguous key-value cache.
/// </summary>
/// <param name="driver">The shared pointer to the CUDA driver wrapper responsible for managing the CUDA context.</param>
/// <param name="addr">The address of the key-value cache data.</param>
/// <param name="dataType">The data type of the cache.</param>
/// <param name="headElems">The number of head elements in the cache.</param>
/// <param name="nbKHeads">The number of key heads in the cache.</param>
/// <param name="maxCacheLen">The maximum length of the cache.</param>
/// <param name="beamWidth">The beam width for the cache.</param>
/// <param name="batchSize">The batch size for the cache.</param>
/// <param name="nbTokensPerTile">The number of tokens per tile (optional, defaults to 64).</param>
/// <returns>A tensor map for the contiguous key-value cache.</returns>
CUtensorMap makeTensorMapForContiguousKVCache(std::shared_ptr<CUDADriverWrapper> const& driver, void const* addr,
    CUtensorMapDataType_enum dataType, uint32_t headElems, uint32_t nbKHeads, uint32_t maxCacheLen, uint32_t beamWidth,
    uint32_t batchSize, uint32_t nbTokensPerTile = 64)
{
    CUtensorMap tensorMap{};
    uint64_t const globalDims[] = {headElems, maxCacheLen, nbKHeads, 2 * beamWidth * batchSize};
    uint32_t elemBytes = getElemBytes(dataType);
    uint32_t const headBytes = elemBytes * headElems;
    uint64_t const globalStrides[] = {headBytes, headBytes * maxCacheLen, headBytes * maxCacheLen * nbKHeads};
    CHECK(headElems <= 256);
    uint32_t const paddedHeadElems = headElems <= 64 ? 64 : (headElems <= 128 ? 128 : 256);
    uint32_t const partElems = std::min(elemBytes * paddedHeadElems, 128U) / elemBytes;
    uint32_t const boxDims[] = {partElems, nbTokensPerTile, 1, 1};
    uint32_t const elemStrides[] = {1, 1, 1, 1};

    auto const swizzle = [&]
    {
        switch (partElems)
        {
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        default: THROW("unsupported cache head size");
        }
    }();

    CU_CHECK(driver->cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims,
        globalStrides, boxDims, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensorMap;
}

/// <summary>
/// Template function to create a tensor map for a given key-value cache buffer.
/// Based on the type of the key-value cache buffer (either KVBlockArray or KVLinearBuffer), it selects the appropriate map creation function.
/// </summary>
/// <typeparam name="KVCacheBuffer">The type of the key-value cache buffer (either KVBlockArray or KVLinearBuffer).</typeparam>
/// <param name="driver">The shared pointer to the CUDA driver wrapper responsible for managing the CUDA context.</param>
/// <param name="xqaParams">The XQA parameters that define the cache structure.</param>
/// <param name="kv_cache_buffer">The key-value cache buffer to create a tensor map for.</param>
/// <returns>A tensor map corresponding to the provided key-value cache buffer.</returns>
template <typename KVCacheBuffer>
CUtensorMap makeTensorMapForKVCache(
    std::shared_ptr<CUDADriverWrapper> const& driver, XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer)
{
    if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
    {
        return makeTensorMapForPagedKVCache(driver, kv_cache_buffer.mPrimaryPoolPtr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            xqaParams.head_size, xqaParams.num_kv_heads, xqaParams.tokens_per_block);
    }
    else
    {
        static_assert(std::is_same_v<KVCacheBuffer, KVLinearBuffer>);
        return makeTensorMapForContiguousKVCache(driver, kv_cache_buffer.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            xqaParams.head_size, xqaParams.num_kv_heads, xqaParams.max_attention_window_size, xqaParams.beam_width,
            xqaParams.batch_size);
    }
}

// Explicit template instantiations for different KVCacheBuffer types
template CUtensorMap makeTensorMapForKVCache(
    std::shared_ptr<CUDADriverWrapper> const&, XQAParams const&, KVBlockArray const&);
template CUtensorMap makeTensorMapForKVCache(
    std::shared_ptr<CUDADriverWrapper> const&, XQAParams const&, KVLinearBuffer const&);

} // namespace suggestify::kernels
