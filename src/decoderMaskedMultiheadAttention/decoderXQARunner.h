

#pragma once

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include "suggestify/common/assert.h"
#include "suggestify/common/cudaUtils.h"
#include "suggestify/common/quantization.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/cubinObjRegistry.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/xqaParams.h"
#include "suggestify/kernels/gptKernels.h"
#include "suggestify/kernels/kvCacheUtils.h"
#include "suggestify/kernels/multiHeadAttentionCommon.h"

using namespace suggestify::common;

namespace suggestify
{
namespace kernels
{

template <typename T, typename KVCacheBuffer>
struct XQADispatchHelper
{
    static constexpr bool CanSupport = false;
};

template <>
struct XQADispatchHelper<__half, KVLinearBuffer>
{
    static constexpr bool CanSupport = true;
};

template <>
struct XQADispatchHelper<__half, KVBlockArray>
{
    static constexpr bool CanSupport = true;
};

#ifdef ENABLE_BF16
template <>
struct XQADispatchHelper<__nv_bfloat16, KVLinearBuffer>
{
    static constexpr bool CanSupport = true;
};

template <>
struct XQADispatchHelper<__nv_bfloat16, KVBlockArray>
{
    static constexpr bool CanSupport = true;
};
#endif

class DecoderXQARunner
{
public:
    DecoderXQARunner(
        const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode);
    ~DecoderXQARunner();

    /**
     * \param[in] xqaParams the xqaParams to be tested against.
     * \param[in] forConfigurePlugin indicates whether this method is called in configurePlugin, or in
     * enqueueGeneration.
     */
    bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin);

    size_t getWorkspaceSize(int max_num_tokens);

    void prepare(XQAParams const& xqa_params)
    {
        this->prepareForRun(xqa_params);
    }

    template <typename KVCacheBuffer>
    void dispatch(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
    {
        sync_check_cuda_error();
        this->run(xqa_params, kv_cache_buffer, stream);
    }

    class Resource;
    static Resource* getResourceGlobal();

private:
    void prepareForRun(XQAParams const& xqa_params);

    template <typename KVCacheBuffer>
    void run(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    static constexpr int kMaxBeamWidth = 4;

    XQADataType mDataType;
    int mNumHeads;
    int mNumKVHeads;
    int mHeadSize;
    bool mMultiBlockMode;
    int mMultiProcessorCount;

    std::unique_ptr<DecoderXQAImpl> mJITImpl, mPrecompiledImpl;
    DecoderXQAImpl* getImplFromXQAParams(XQAParams const& params, bool for_configure_plugin);

    friend DecoderXQAImplPrecompiled;
    friend DecoderXQAImplJIT;
};

class DecoderXQARunner::Resource
{
public:
    Resource();
    Resource(Resource const& other);
    Resource& operator=(Resource const& other);
    Resource(Resource&& other) = default;
    Resource& operator=(Resource&& other) = default;
    // Construct from a serialized buffer.
    Resource(void const* buffer, size_t buffer_size);
    ~Resource() = default;

    // When initialize is true, initialize cubins.
    void merge(Resource const& other, bool initialize)
    {
        getCubinObjRegistry()->merge(*other.getCubinObjRegistry(), initialize);
    }

    jit::CubinObjRegistry* getCubinObjRegistry()
    {
        return mCubinObjRegistry.get();
    }

    jit::CubinObjRegistry const* getCubinObjRegistry() const
    {
        return mCubinObjRegistry.get();
    }

    size_t getSerializationSize() const noexcept;
    void serialize(void* buffer, size_t buffer_size) const noexcept;

private:
    std::unique_ptr<jit::CubinObjRegistry> mCubinObjRegistry;
};

} // namespace kernels
} // namespace suggestify
