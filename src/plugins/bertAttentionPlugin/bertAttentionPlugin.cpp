#include "bertAttentionPlugin.h"
#include "../src/decoderMaskedMultiheadAttention.h"
#include "../src/gptKernels.h"
#include "../src/unfusedAttentionKernels.h"
#include "../runtime/iBuffer.h"

using namespace nvinfer1;
using namespace suggestify::kernels;
namespace tc = suggestify::common;

using suggestify::plugins::BertAttentionPluginCreator;
using suggestify::plugins::BertAttentionPlugin;

static char const* BERT_ATTENTION_PLUGIN_VERSION{"1"};
static char const* BERT_ATTENTION_PLUGIN_NAME{"BertAttention"};
PluginFieldCollection BertAttentionPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> BertAttentionPluginCreator::mPluginAttributes;

BertAttentionPlugin::BertAttentionPlugin(int num_heads, int head_size, float q_scaling,
    ContextFMHAType context_fmha_type, nvinfer1::DataType type, bool do_relative_attention, int max_distance,
    bool remove_padding)
    : mNumHeads(num_heads)
    , mHeadSize(head_size)
    , mQScaling(q_scaling)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::DISABLED)
    , mFMHAForceFP32Acc(context_fmha_type == ContextFMHAType::ENABLED_WITH_FP32_ACC)
    , mType(type)
    , mRelativeAttention(do_relative_attention)
    , mMaxDistance(max_distance)
    , mRemovePadding(remove_padding)
{
    if (mEnableContextFMHA)
    {
        mEnableContextFMHA = false;
        if (!(mType == DataType::kHALF || mType == DataType::kBF16))
        {
            LOG_WARNING("Fall back to unfused MHA because of unsupported data type.");
        }
        else if (mRelativeAttention)
        {
            LOG_WARNING("Fall back to unfused MHA because of relative position embedding.");
        }
        else
        {
            mEnableContextFMHA = true;
        }
    }
}

BertAttentionPlugin::BertAttentionPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mNumHeads);
    read(d, mHeadSize);
    read(d, mQScaling);
    read(d, mQKHalfAccum);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mType);
    read(d, mRelativeAttention);
    read(d, mMaxDistance);
    read(d, mRemovePadding);
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different nexly version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* BertAttentionPlugin::clone() const noexcept
{
    auto* plugin = new BertAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs BertAttentionPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    CHECK(outputIndex == 0);
    auto ret = inputs[0];
    ret.d[mRemovePadding ? 1 : 2] = exprBuilder.constant(ret.d[mRemovePadding ? 1 : 2]->getConstantValue() / 3);
    return ret;
}

bool BertAttentionPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (nbInputs == 2)
    {
        if (pos == 1)
        {
            return inOut[pos].type == nvinfer1::DataType::kINT32;
        }
        else
        {
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    else if (nbInputs > 2)
    {
        if (pos == 1 || pos == 2)
        {
            return inOut[pos].type == nvinfer1::DataType::kINT32;
        }
        else
        {
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    else
    {
        return false;
    }
}

void BertAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t BertAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    int const batch_size = mRemovePadding ? inputs[1].dims.d[0] : inputs[0].dims.d[0];
    int const input_seq_len = mRemovePadding ? inputs[2].dims.d[0] : inputs[0].dims.d[1];
    int const local_hidden_units_ = inputs[0].dims.d[mRemovePadding ? 1 : 2] / 3;

    auto const size = suggestify::runtime::BufferDataType(inputs[0].type).getSize();

    const size_t attention_mask_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * input_seq_len;
    const size_t cu_seqlens_size = sizeof(int) * (batch_size + 1);
    const size_t q_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t k_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t v_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_size = mEnableContextFMHA ? 0 : size * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t qkv_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * batch_size * input_seq_len;
    const size_t fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;

    int const NUM_BUFFERS = 11;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = CUBLAS_WORKSPACE_SIZE;
    workspaces[1] = attention_mask_size;
    workspaces[2] = cu_seqlens_size;
    workspaces[3] = q_buf_2_size;
    workspaces[4] = k_buf_2_size;
    workspaces[5] = v_buf_2_size;
    workspaces[6] = qk_buf_size;
    workspaces[7] = qkv_buf_2_size;
    workspaces[8] = qk_buf_float_size;
    workspaces[9] = padding_offset_size;
    workspaces[10] = fmha_scheduler_counter;

    return tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
}

template <typename T>
int BertAttentionPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{


    int const batch_size = mRemovePadding ? inputDesc[1].dims.d[0] : inputDesc[0].dims.d[0];
    int const input_seq_len = mRemovePadding ? inputDesc[2].dims.d[0] : inputDesc[0].dims.d[1];
    int const num_tokens = mRemovePadding ? inputDesc[0].dims.d[0] : batch_size * input_seq_len;
    int const request_batch_size = batch_size;
    int const request_seq_len = input_seq_len;
    int const local_hidden_units_ = inputDesc[0].dims.d[mRemovePadding ? 1 : 2] / 3;
    float const q_scaling = mQScaling;

    T const* attention_input = reinterpret_cast<T const*>(inputs[0]);
    int const* input_lengths = reinterpret_cast<int const*>(inputs[1]);
    T const* relative_attn_table = mRelativeAttention ? reinterpret_cast<T const*>(inputs[3]) : nullptr;
    T* context_buf_ = (T*) (outputs[0]);

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(workspace);
    if (inputDesc[0].type == DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if (inputDesc[0].type == DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    const size_t attention_mask_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * input_seq_len;
    const size_t cu_seqlens_size = sizeof(int) * (batch_size + 1);
    const size_t q_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t k_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t v_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_size
        = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t qkv_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * batch_size * input_seq_len;
    const size_t fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;

    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    T* attention_mask = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, attention_mask_size));
    int* cu_seqlens = reinterpret_cast<int*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    T* q_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, q_buf_2_size));
    T* k_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, k_buf_2_size));
    T* v_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, v_buf_2_size));
    T* qk_buf_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_size));
    T* qkv_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, qkv_buf_2_size));
    float* qk_buf_float_
        = reinterpret_cast<float*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_float_size));
    int* padding_offset = reinterpret_cast<int*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, padding_offset_size));
    uint32_t* fmha_tile_counter_ptr
        = reinterpret_cast<uint32_t*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, fmha_scheduler_counter));

    BuildDecoderInfoParams<T> params;
    memset(&params, 0, sizeof(params));
    params.seqQOffsets = cu_seqlens;
    params.paddingOffsets = padding_offset;
    params.attentionMask = attention_mask;
    params.seqQLengths = input_lengths;
    params.batchSize = batch_size;
    params.maxQSeqLength = input_seq_len;
    params.numTokens = num_tokens;
    params.attentionMaskType = AttentionMaskType::PADDING;
    params.fmhaTileCounter = fmha_tile_counter_ptr;
    invokeBuildDecoderInfo(params, stream);
    sync_check_cuda_error();

    auto const gemm_data_type = tc::CudaDataType<T>::value;
    int const attention_seq_len_1 = request_seq_len;
    int const attention_seq_len_2 = request_seq_len;

    float const qk_scale
        = 1.0f / (sqrtf(mHeadSize * 1.0f) * q_scaling);
    float const qk_scale_gemm = mRelativeAttention ? qk_scale : 1.0f;
    const T qk_scale_softmax = static_cast<T>(mRelativeAttention ? 1.0f : qk_scale);

    T* linear_bias_slopes = nullptr;

    if (mEnableContextFMHA)
    {
        MHARunnerParams fmhaParams{};
        fmhaParams.b = request_batch_size;
        fmhaParams.qSeqLen = request_seq_len;
        fmhaParams.kvSeqLen = request_seq_len;
        fmhaParams.totalQSeqLen = request_batch_size * request_seq_len;
        fmhaParams.qkvPtr = attention_input;
        fmhaParams.outputPtr = context_buf_;
        fmhaParams.cuQSeqLenPtr = cu_seqlens;
        fmhaParams.cuKvSeqLenPtr = cu_seqlens;
        fmhaParams.tileCounterPtr = fmha_tile_counter_ptr;
        fmhaParams.stream = stream;

        mFMHARunner->run(fmhaParams);
    }
    else
    {
        cudaMemsetAsync(k_buf_2_, 0,
            reinterpret_cast<int8_t*>(v_buf_2_) - reinterpret_cast<int8_t*>(k_buf_2_) + v_buf_2_size, stream);

        invokeAddFusedQKVBiasTranspose(q_buf_2_, k_buf_2_, v_buf_2_, const_cast<T*>(attention_input), input_lengths,
            mRemovePadding ? padding_offset : nullptr, batch_size, input_seq_len, num_tokens, mNumHeads, mNumHeads,
            mHeadSize, 0, 0.0f, RotaryScalingType::kNONE, 0.0f, 0, PositionEmbeddingType::kLEARNED_ABSOLUTE,
            (float*) nullptr, 0, stream);

        if (!mQKHalfAccum && gemm_data_type != CUDA_R_32F)
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,
                attention_seq_len_1,
                mHeadSize,
                qk_scale_gemm, k_buf_2_, gemm_data_type,
                mHeadSize,
                attention_seq_len_2 * mHeadSize,
                q_buf_2_, gemm_data_type,
                mHeadSize,
                attention_seq_len_1 * mHeadSize,
                0.0f, qk_buf_float_, CUDA_R_32F,
                attention_seq_len_2,
                attention_seq_len_2 * attention_seq_len_1,
                request_batch_size * mNumHeads,
                CUDA_R_32F);

            if (mRelativeAttention)
            {
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_float_, relative_attn_table, request_batch_size,
                    mNumHeads, attention_seq_len_1, attention_seq_len_2, stream, mMaxDistance > 0,
                    inputDesc[3].dims.d[1], mMaxDistance, true);
            }

            MaskedSoftmaxParam<T, float> param;
            param.attention_score = qk_buf_;
            param.qk = qk_buf_float_;
            param.attention_mask = attention_mask;
            param.batch_size = request_batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);
            invokeMaskedSoftmax(param, stream);
        }
        else
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N, attention_seq_len_2, attention_seq_len_1,
                mHeadSize, k_buf_2_, mHeadSize, attention_seq_len_2 * mHeadSize, q_buf_2_, mHeadSize,
                attention_seq_len_1 * mHeadSize, qk_buf_, attention_seq_len_2,
                attention_seq_len_2 * attention_seq_len_1, request_batch_size * mNumHeads, qk_scale_gemm,
                0.0f);

            if (mRelativeAttention)
            {
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_, relative_attn_table, request_batch_size, mNumHeads,
                    attention_seq_len_1, attention_seq_len_2, stream, mMaxDistance > 0, inputDesc[3].dims.d[1],
                    mMaxDistance, true);
            }

            MaskedSoftmaxParam<T, T> param;
            param.attention_score = qk_buf_;
            param.qk = qk_buf_;
            param.attention_mask = attention_mask;
            param.batch_size = request_batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);
            invokeMaskedSoftmax(param, stream);
        }

        mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, attention_seq_len_1,
            attention_seq_len_2, v_buf_2_, mHeadSize, attention_seq_len_2 * mHeadSize, qk_buf_, attention_seq_len_2,
            attention_seq_len_1 * attention_seq_len_2, qkv_buf_2_, mHeadSize, attention_seq_len_1 * mHeadSize,
            request_batch_size * mNumHeads);

        if (!mRemovePadding)
        {
            invokeTransposeQKV(context_buf_, qkv_buf_2_, request_batch_size, attention_seq_len_1, mNumHeads, mHeadSize,
                (float*) nullptr, 0, stream);
        }
        else
        {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_, context_buf_, num_tokens, request_batch_size,
                request_seq_len, mNumHeads, mHeadSize, padding_offset, (float*) nullptr, 0, stream);
        }
    }
    sync_check_cuda_error();
    return 0;
}

template int BertAttentionPlugin::enqueueImpl<half>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

template int BertAttentionPlugin::enqueueImpl<float>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

#ifdef ENABLE_BF16
template int BertAttentionPlugin::enqueueImpl<__nv_bfloat16>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);
#endif

int BertAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#endif
    return 0;
}

nvinfer1::DataType BertAttentionPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index == 0);
    return inputTypes[0];
}


char const* BertAttentionPlugin::getPluginType() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

char const* BertAttentionPlugin::getPluginVersion() const noexcept
{
    return BERT_ATTENTION_PLUGIN_VERSION;
}

int BertAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int BertAttentionPlugin::initialize() noexcept
{
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    mCublasWrapper.reset(new tc::CublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, nullptr));
    if (mEnableContextFMHA)
    {
        Data_type data_type;
        if (mType == DataType::kHALF)
        {
            data_type = DATA_TYPE_FP16;
        }
        else if (mType == DataType::kBF16)
        {
            data_type = DATA_TYPE_BF16;
        }
        else
        {
            CHECK_WITH_INFO(false, "GPTAttentionPlugin received wrong data type.");
        }

        MHARunnerFixedParams fmhaParams{};
        fmhaParams.dataType = data_type;
        fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;
        fmhaParams.attentionMaskType = ContextAttentionMaskType::PADDING;
        fmhaParams.isSPadded = !mRemovePadding;
        fmhaParams.numQHeads = mNumHeads;
        fmhaParams.numKvHeads = mNumHeads;
        fmhaParams.headSize = mHeadSize;
        fmhaParams.qScaling = mQScaling;

        mFMHARunner.reset(new FusedMHARunnerV2(fmhaParams));

        mEnableContextFMHA = mFMHARunner->isFmhaSupported();
    }

    return 0;
}

void BertAttentionPlugin::destroy() noexcept
{
    delete this;
}

size_t BertAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mQScaling) + sizeof(mQKHalfAccum) + sizeof(mEnableContextFMHA)
        + sizeof(mFMHAForceFP32Acc) + sizeof(mType) + sizeof(mRelativeAttention) + sizeof(mMaxDistance)
        + sizeof(mRemovePadding);
}

void BertAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumHeads);
    write(d, mHeadSize);
    write(d, mQScaling);
    write(d, mQKHalfAccum);
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mType);
    write(d, mRelativeAttention);
    write(d, mMaxDistance);
    write(d, mRemovePadding);
    assert(d == a + getSerializationSize());
}

void BertAttentionPlugin::terminate() noexcept {}


BertAttentionPluginCreator::BertAttentionPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("do_relative_attention", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("max_distance", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("remove_padding", nullptr, PluginFieldType::kINT8, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* BertAttentionPluginCreator::getPluginName() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

char const* BertAttentionPluginCreator::getPluginVersion() const noexcept
{
    return BERT_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* BertAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* BertAttentionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int num_heads, head_size;
    ContextFMHAType context_fmha_type;
    float q_scaling;
    nvinfer1::DataType type;
    bool do_relative_attention;
    int max_distance;
    bool remove_padding;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            num_heads = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_size"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            head_size = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "q_scaling"))
        {
            CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            q_scaling = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "context_fmha_type"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT8);
            context_fmha_type = static_cast<ContextFMHAType>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "do_relative_attention"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT8);
            do_relative_attention = static_cast<bool>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "max_distance"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            max_distance = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_padding"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT8);
            remove_padding = static_cast<bool>(*(static_cast<int8_t const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new BertAttentionPlugin(num_heads, head_size, q_scaling, context_fmha_type, type,
            do_relative_attention, max_distance, remove_padding);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* BertAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new BertAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
