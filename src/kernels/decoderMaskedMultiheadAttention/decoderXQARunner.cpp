#include "decoderXQARunner.h"

#include <assert.h>
#include <string.h>

#include <mutex>
#include <unordered_map>

#include "cudaUtils.h"
#include "envUtils.h"
#include "workspace.h"
#include "../src/decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.h"
#include "../src/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "../src/decoderMaskedMultiheadAttention/decoderXQAImpl.h"
#include "../src/kvCacheUtils.h"
#include "../src/unfusedAttentionKernels.h"

namespace suggestify
{
namespace kernels
{
    /// <summary>
    /// Constructor for the DecoderXQARunner class. Initializes the data type, number of heads, and other parameters.
    /// </summary>
    /// <param name="data_type">The data type to be used (e.g., fp16 or bf16).</param>
    /// <param name="num_heads">The number of attention heads.</param>
    /// <param name="num_kv_heads">The number of key-value heads.</param>
    /// <param name="head_size">The size of each attention head.</param>
    /// <param name="multi_block_mode">Flag to enable multi-block mode.</param>
    DecoderXQARunner::DecoderXQARunner(
        const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode)
        : mDataType(data_type)
        , mNumHeads(num_heads)
        , mNumKVHeads(num_kv_heads)
        , mHeadSize(head_size)
        , mMultiBlockMode(multi_block_mode)
    {
        mMultiProcessorCount = suggestify::common::getMultiProcessorCount();
        mJITImpl = DecoderXQAImpl::create(this, DecoderXQAImpl::ImplType::kJIT);
        mPrecompiledImpl = DecoderXQAImpl::create(this, DecoderXQAImpl::ImplType::kPrecompiled);
    }

    /// <summary>
    /// Destructor for the DecoderXQARunner class.
    /// </summary>
    DecoderXQARunner::~DecoderXQARunner() = default;

    namespace
    {
        /// <summary>
        /// Calculates the ceiling of division of two values.
        /// </summary>
        /// <typeparam name="T">The type of the values.</typeparam>
        /// <param name="a">The numerator.</param>
        /// <param name="b">The denominator.</param>
        /// <returns>The ceiling of the division result.</returns>
        template <typename T>
        constexpr inline T divUp(T a, T b)
        {
            return (a + b - 1) / b;
        }

        /// <summary>
        /// Rounds up a value to the next multiple of another value.
        /// </summary>
        /// <typeparam name="T">The type of the values.</typeparam>
        /// <param name="a">The value to round up.</param>
        /// <param name="b">The value to round to.</param>
        /// <returns>The rounded value.</returns>
        template <typename T>
        constexpr inline T roundUp(T a, T b)
        {
            return divUp(a, b) * b;
        }
    } // namespace

    /// <summary>
    /// Calculates the required workspace size for a given maximum number of tokens.
    /// </summary>
    /// <param name="max_num_tokens">The maximum number of tokens to process.</param>
    /// <returns>The required workspace size in bytes.</returns>
    size_t DecoderXQARunner::getWorkspaceSize(int max_num_tokens)
    {
        constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
        size_t workspace_size = roundUp<size_t>(kXQA_OUT_ELEM_SIZE * mHeadSize * mNumHeads * max_num_tokens, 128); // rope
        workspace_size = roundUp<size_t>(workspace_size + kXQA_OUT_ELEM_SIZE * mHeadSize * mNumHeads * max_num_tokens, 128);
        
        if (mMultiBlockMode)
        {
            int workspaces[4];
            uint32_t const nbSubSeq = kXQA_MAX_NUM_SUB_SEQ;
            uint32_t const nbSeq = nbSubSeq / 2;
            int group_size = mNumHeads / mNumKVHeads;
            workspaces[0] = sizeof(uint32_t) * nbSeq;                           // semaphores
            workspaces[1] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowMax
            workspaces[2] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowSum
            int32_t const multi_block_workspace_alignment
                = roundUp<int32_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * mHeadSize, 128);
            workspaces[3] = multi_block_workspace_alignment * nbSubSeq;
            workspace_size = roundUp<size_t>(workspace_size, multi_block_workspace_alignment)
                + roundUp(workspaces[0], multi_block_workspace_alignment)
                + roundUp(workspaces[1], multi_block_workspace_alignment)
                + roundUp(workspaces[2], multi_block_workspace_alignment)
                + roundUp(workspaces[3], multi_block_workspace_alignment)
                + multi_block_workspace_alignment; // extra space reserved for alignment
        }
        
        return workspace_size;
    }

    /// <summary>
    /// Selects the appropriate implementation based on the given XQA parameters.
    /// </summary>
    /// <param name="xqaParams">The XQA parameters.</param>
    /// <param name="for_configure_plugin">Flag indicating if it's for plugin configuration.</param>
    /// <returns>The selected implementation.</returns>
    DecoderXQAImpl* DecoderXQARunner::getImplFromXQAParams(XQAParams const& xqaParams, bool for_configure_plugin)
    {
        if (xqaParams.multi_query_tokens)
        {
            return mPrecompiledImpl.get();
        }

        std::optional<bool> envEnableXQAJIT = suggestify::common::getEnvEnableXQAJIT();

        if (envEnableXQAJIT.has_value())
        {
            return envEnableXQAJIT.value() ? mJITImpl.get() : mPrecompiledImpl.get();
        }
        else
        {
            return mJITImpl.get();
        }
    }

    /// <summary>
    /// Determines whether the current implementation should be used based on the XQA parameters.
    /// </summary>
    /// <param name="xqa_params">The XQA parameters.</param>
    /// <param name="for_configure_plugin">Flag indicating if it's for plugin configuration.</param>
    /// <returns>True if the implementation should be used, false otherwise.</returns>
    bool DecoderXQARunner::shouldUse(XQAParams const& xqa_params, bool for_configure_plugin)
    {
        return getImplFromXQAParams(xqa_params, for_configure_plugin)->shouldUse(xqa_params, for_configure_plugin);
    }

    /// <summary>
    /// Prepares for running the implementation with the given XQA parameters.
    /// </summary>
    /// <param name="xqa_params">The XQA parameters.</param>
    void DecoderXQARunner::prepareForRun(XQAParams const& xqa_params)
    {
        return getImplFromXQAParams(xqa_params, true)->prepare(xqa_params);
    }

    /// <summary>
    /// Executes the implementation with the given XQA parameters and KV cache buffer.
    /// </summary>
    /// <typeparam name="KVCacheBuffer">The type of the KV cache buffer.</typeparam>
    /// <param name="xqa_params">The XQA parameters.</param>
    /// <param name="kv_cache_buffer">The KV cache buffer.</param>
    /// <param name="stream">The CUDA stream for execution.</param>
    template <typename KVCacheBuffer>
    void DecoderXQARunner::run(
        XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
    {
        return getImplFromXQAParams(xqa_params, false)->run(xqa_params, kv_cache_buffer, stream);
    }

    /// <summary>
    /// Gets the global resource object for the DecoderXQARunner.
    /// </summary>
    /// <returns>The global resource object.</returns>
    DecoderXQARunner::Resource* DecoderXQARunner::getResourceGlobal()
    {
        static DecoderXQARunner::Resource sResource;
        return &sResource;
    }

    // Template instantiation for different KV cache buffer types
    template void DecoderXQARunner::run(
        XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream);
    template void DecoderXQARunner::run(
        XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream);

    /// <summary>
    /// Constructor for the Resource class in DecoderXQARunner.
    /// </summary>
    DecoderXQARunner::Resource::Resource()
        : mCubinObjRegistry(std::make_unique<jit::CubinObjRegistry>())
    {
    }

    /// <summary>
    /// Copy constructor for the Resource class in DecoderXQARunner.
    /// </summary>
    DecoderXQARunner::Resource::Resource(DecoderXQARunner::Resource const& other)
        : mCubinObjRegistry(other.mCubinObjRegistry->clone())
    {
    }

    /// <summary>
    /// Assignment operator for the Resource class in DecoderXQARunner.
    /// </summary>
    DecoderXQARunner::Resource& DecoderXQARunner::Resource::operator=(DecoderXQARunner::Resource const& other)
    {
        if (this == &other)
        {
            return *this;
        }
        mCubinObjRegistry = other.mCubinObjRegistry->clone();
        return *this;
    }

    /// <summary>
    /// Constructor for the Resource class with a buffer and its size.
    /// </summary>
    /// <param name="buffer">The buffer containing serialized data.</param>
    /// <param name="buffer_size">The size of the buffer.</param>
    DecoderXQARunner::Resource::Resource(void const* buffer, size_t buffer_size)
        : mCubinObjRegistry(std::make_unique<jit::CubinObjRegistry>(buffer, buffer_size))
    {
    }

    /// <summary>
    /// Gets the serialization size for the Resource object.
    /// </summary>
    /// <returns>The serialization size in bytes.</returns>
    size_t DecoderXQARunner::Resource::getSerializationSize() const noexcept
    {
        return mCubinObjRegistry->getSerializationSize();
    }

    /// <summary>
    /// Serializes the Resource object into the given buffer.
    /// </summary>
    /// <param name="buffer">The buffer to store the serialized data.</param>
    /// <param name="buffer_size">The size of the buffer.</param>
    void DecoderXQARunner::Resource::serialize(void* buffer, size_t buffer_size) const noexcept
    {
        mCubinObjRegistry->serialize(buffer, buffer_size);
    }

} // namespace kernels

} // namespace suggestify
