#pragma once

#include <memory>

#include "../src/decoderMaskedMultiheadAttention/xqaParams.h"
#include "../src/kvCacheUtils.h"

namespace suggestify
{
namespace kernels
{
    /// <summary>
    /// A class that implements the XQA kernel logic in a decoder model.
    /// </summary>
    class DecoderXQARunner;

    /// <summary>
    /// Provides the base class for DecoderXQA implementation. 
    /// It defines methods for determining when XQA should be used, preparing parameters, and running the kernel.
    /// </summary>
    class DecoderXQAImpl
    {
    public:
        /// <summary>
        /// Determines whether XQA should be used based on the provided parameters.
        /// </summary>
        /// <param name="xqaParams">The parameters to determine if XQA should be used.</param>
        /// <param name="forConfigurePlugin">A flag indicating whether the check is for plugin configuration.</param>
        /// <returns>True if XQA should be used, false otherwise.</returns>
        virtual bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin) = 0;

        /// <summary>
        /// Prepares the XQA parameters for the kernel execution.
        /// </summary>
        /// <param name="xqa_params">The XQA parameters to prepare.</param>
        virtual void prepare(XQAParams const& xqa_params) = 0;

        /// <summary>
        /// Executes the XQA kernel using a given cache buffer and CUDA stream.
        /// </summary>
        /// <typeparam name="KVCacheBuffer">The type of the key-value cache buffer.</typeparam>
        /// <param name="xqa_params">The parameters for the XQA execution.</param>
        /// <param name="kv_cache_buffer">The key-value cache buffer used in the execution.</param>
        /// <param name="stream">The CUDA stream for kernel execution.</param>
        template <typename KVCacheBuffer>
        void run(XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream);

        /// <summary>
        /// The type of the XQA implementation.
        /// </summary>
        enum class ImplType
        {
            /// <summary>
            /// Precompiled implementation of the XQA kernel.
            /// </summary>
            kPrecompiled = 0,

            /// <summary>
            /// Just-in-time (JIT) compiled implementation of the XQA kernel.
            /// </summary>
            kJIT = 1,
        };

        /// <summary>
        /// Creates an instance of the XQA implementation.
        /// </summary>
        /// <param name="runner">The runner object responsible for running the XQA kernel.</param>
        /// <param name="implType">The type of implementation (precompiled or JIT).</param>
        /// <returns>A unique pointer to the created XQA implementation.</returns>
        static std::unique_ptr<DecoderXQAImpl> create(DecoderXQARunner* runner, ImplType implType);

    protected:
        /// <summary>
        /// Protected constructor for the DecoderXQAImpl class.
        /// </summary>
        /// <param name="runner">The runner responsible for running the XQA kernel.</param>
        DecoderXQAImpl(DecoderXQARunner* runner)
            : mRunner(runner)
        {
        }

        /// <summary>
        /// Runs the XQA kernel with a key-value linear buffer.
        /// </summary>
        /// <param name="xqa_params">The XQA parameters.</param>
        /// <param name="kv_linear_buffer">The key-value linear buffer used in execution.</param>
        /// <param name="stream">The CUDA stream for execution.</param>
        virtual void runWithKVLinearBuffer(
            XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream)
            = 0;

        /// <summary>
        /// Runs the XQA kernel with a key-value block array.
        /// </summary>
        /// <param name="xqa_params">The XQA parameters.</param>
        /// <param name="kv_block_array">The key-value block array used in execution.</param>
        /// <param name="stream">The CUDA stream for execution.</param>
        virtual void runWithKVBlockArray(
            XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream)
            = 0;

        /// <summary>
        /// The runner responsible for executing the XQA kernel.
        /// </summary>
        DecoderXQARunner* mRunner;
    };

    /// <summary>
    /// Specifies the different types of XQA kernels available.
    /// </summary>
    enum class XQAKernelType : int32_t
    {
        /// <summary>
        /// AMPERE specialized warp kernel type.
        /// </summary>
        kAMPERE_WARP_SPECIALIZED = 0,

        /// <summary>
        /// HOPPER specialized warp kernel type.
        /// </summary>
        kHOPPER_WARP_SPECIALIZED = 1
    };

}
}
