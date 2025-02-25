
#pragma once
#include <memory>

#include "suggestify/kernels/decoderMaskedMultiheadAttention/xqaParams.h"
#include "suggestify/kernels/kvCacheUtils.h"

namespace suggestify
{
namespace kernels
{

// Forward declaration to avoid cyclic dependency.
class DecoderXQARunner;

/**
 * The underlying XQA implementation called from DecoderXQARunner.
 *
 * We need this layer of abstraction for abstracting out implementation details. Two possible implementations:
 *   1. Precompiled, i.e. kernels are compiled and saved as cubins in advance.
 *   2. JIT, i.e. kernels are compiled on the fly via NVRTC.
 */
class DecoderXQAImpl
{
public:
    // TODO(minwei): shouldUse()/prepare() should be templated with KVCacheBuffer.
    // Whether it is beneficial to use this XQA codepath.
    //
    // forConfigurePlugin: whether this method is called in configure plugin phase.
    virtual bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin) = 0;
    // Prepares for the kernel running. Must be called before calling run.
    virtual void prepare(XQAParams const& xqa_params) = 0;
    // Run XQA kernel with KVCacheBuffer.
    //
    // Sub-classes should implement runWithKVLinearBuffer and runWithKVBlockArray.
    template <typename KVCacheBuffer>
    void run(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

    enum class ImplType
    {
        kPrecompiled = 0,
        kJIT = 1,
    };
    // Needs runner pointer for accessing resources in DecoderXQARunner class.
    static std::unique_ptr<DecoderXQAImpl> create(DecoderXQARunner* runner, ImplType implType);

protected:
    DecoderXQAImpl(DecoderXQARunner* runner)
        : mRunner(runner)
    {
    }

    virtual void runWithKVLinearBuffer(
        XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream)
        = 0;
    virtual void runWithKVBlockArray(
        XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream)
        = 0;

    DecoderXQARunner* mRunner;
};

enum class XQAKernelType : int32_t
{
    kAMPERE_WARP_SPECIALIZED = 0,
    kHOPPER_WARP_SPECIALIZED = 1
};

} // namespace kernels
} // namespace suggestify
