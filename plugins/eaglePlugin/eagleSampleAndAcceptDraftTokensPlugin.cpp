#include "eagleSampleAndAcceptDraftTokensPlugin.h"

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/dataType.h"
#include "../common/memoryUtils.h"
#include "../src/samplingTopKKernels.h"
#include "../src/speculativeDecoding/common.h"
#include "../src/speculativeDecoding/eagleDecodingKernels.h"
#include "../src/speculativeDecoding/medusaDecodingKernels.h"
#include "../runtime/common.h"
#include "../runtime/iTensor.h"

using namespace nvinfer1;
using suggestify::plugins::EagleSampleAndAcceptDraftTokensPluginCreator;
using suggestify::plugins::EagleSampleAndAcceptDraftTokensPlugin;
using namespace suggestify::kernels;
using namespace suggestify::kernels::speculative_decoding;
using namespace suggestify::runtime;
namespace tc = suggestify::common;

static char const* EAGLE_SAMPLE_AND_ACCEPT_DRAFT_TOKENS_PLUGIN_VERSION{"1"};
static char const* EAGLE_SAMPLE_AND_ACCEPT_DRAFT_TOKENS_PLUGIN_NAME{"EagleSampleAndAcceptDraftTokens"};
PluginFieldCollection EagleSampleAndAcceptDraftTokensPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EagleSampleAndAcceptDraftTokensPluginCreator::mPluginAttributes;

EagleSampleAndAcceptDraftTokensPlugin::EagleSampleAndAcceptDraftTokensPlugin(nvinfer1::DataType type)
    : mDtype(type)
{
}

EagleSampleAndAcceptDraftTokensPlugin::EagleSampleAndAcceptDraftTokensPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDtype);
    CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

nvinfer1::IPluginV2DynamicExt* EagleSampleAndAcceptDraftTokensPlugin::clone() const noexcept
{
    auto* plugin = new EagleSampleAndAcceptDraftTokensPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs EagleSampleAndAcceptDraftTokensPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    CHECK(nbInputs == 9);
    CHECK(outputIndex < 7);
    auto const batchSizeExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[0];
    auto const maxDecodingDraftTokensExpr = inputs[getIdx(InputIdxEntry::DRAFT_TOKEN_IDS)].d[1];
    auto const maxDecodingTokensExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[1];
    auto const maxPathLenExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[2];

    nvinfer1::DimsExprs ret;
    if (outputIndex == getIdx(OutputIdxEntry::ACCEPTED_TOKENS))
    {
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxPathLenExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::ACCEPTED_LENS))
    {
        ret.nbDims = 1;
        ret.d[0] = batchSizeExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::BEST_ACCEPTED_PATHS))
    {
        ret.nbDims = 1;
        ret.d[0] = batchSizeExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::NEXT_DRAFT_TOKEN_IDS))
    {
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::NEXT_DRAFT_LENS))
    {
        ret.nbDims = 1;
        ret.d[0] = batchSizeExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::NEXT_DRAFT_PATHS))
    {
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingTokensExpr;
        ret.d[2] = maxPathLenExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::HIDDEN_SIZE_BATCH_LEVEL_STARTS))
    {
        ret.nbDims = 1;
        ret.d[0] = exprBuilder.operation(DimensionOperation::kSUM, *exprBuilder.constant(1),
            *exprBuilder.operation(DimensionOperation::kPROD,
                *exprBuilder.operation(DimensionOperation::kSUB, *maxPathLenExpr, *exprBuilder.constant(1)),
                *batchSizeExpr));
    }
    return ret;
}

bool EagleSampleAndAcceptDraftTokensPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getIdx(InputIdxEntry::LOGITS))
    {
        return (inOut[pos].type == mDtype) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == getIdx(InputIdxEntry::TEMPERATURE) || pos == getIdx(InputIdxEntry::RAND_VALIDATION)
        || pos == getIdx(InputIdxEntry::POSTERIOR_ALPHA)
        || pos == getIdx(InputIdxEntry::POSTERIOR_THRESHOLD))
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void EagleSampleAndAcceptDraftTokensPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

template <typename T>
size_t EagleSampleAndAcceptDraftTokensPlugin::getWorkspaceSizeType(nvinfer1::PluginTensorDesc const* inputs,
    int nbInputs, nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    size_t workspaceSize{0};

    auto const vocabSizePadded = inputs[getIdx(InputIdxEntry::LOGITS)].dims.d[1];
    auto const batchSize = inputs[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const maxDecodingTokens = inputs[getIdx(InputIdxEntry::PATHS)].dims.d[1];

    auto const greedySamplingWorkspaceSize
        = getTopKWorkspaceSize<T>(batchSize, maxDecodingTokens, 1, vocabSizePadded);

    auto const typicalSamplingWorkspaceSize
        = getTypicalAcceptanceWorkspaceSize<T>(batchSize, maxDecodingTokens, vocabSizePadded);

    auto const primarySamplingWorkspaceSize = std::max(greedySamplingWorkspaceSize, typicalSamplingWorkspaceSize);

    auto const targetOutputIdsSize = batchSize * maxDecodingTokens * sizeof(TokenIdType);
    auto const logitsPtrsSize = batchSize * maxDecodingTokens * sizeof(T*);
    SizeType32 constexpr NUM_BUFFERS{4};
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = targetOutputIdsSize;
    workspaces[1] = primarySamplingWorkspaceSize;
    workspaces[2] = logitsPtrsSize;
    workspaces[3] = batchSize * sizeof(SizeType32);
    workspaceSize = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);

    return workspaceSize;
}

size_t EagleSampleAndAcceptDraftTokensPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    auto const logitsType = inputs[getIdx(InputIdxEntry::LOGITS)].type;
    if (logitsType == nvinfer1::DataType::kFLOAT)
    {
        return getWorkspaceSizeType<float>(inputs, nbInputs, outputs, nbOutputs);
    }
    else if (logitsType == nvinfer1::DataType::kHALF)
    {
        return getWorkspaceSizeType<__half>(inputs, nbInputs, outputs, nbOutputs);
    }
    else
    {
        CHECK_WITH_INFO(false, "Unsupported logits type");
    }
    return 0;
}

template <typename T>
void EagleSampleAndAcceptDraftTokensPlugin::samplePrimeHeadTokens(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const maxNumTokens = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const vocabSizePadded = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[1];
    auto const batchSize = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[1];

    auto logits = static_cast<T const*>(inputs[getIdx(InputIdxEntry::LOGITS)]);
    auto prevDraftLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::DRAFT_LENS)]);

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace);
    size_t offset{0};

    auto const samplingWorkspaceSize
        = getTopKWorkspaceSize<T>(batchSize, maxDecodingTokens, 1, vocabSizePadded);

    TokenIdType* outputIds = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(TokenIdType)));
    void* workspaceSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, samplingWorkspaceSize));
    T const** logitsPtrs = reinterpret_cast<T const**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(T*)));
    SizeType32* decodingTokens
        = reinterpret_cast<SizeType32*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(SizeType32)));

    invokeAssembleTargetLogitsOffsets(
        logitsPtrs, decodingTokens, logits, prevDraftLens, batchSize, maxDecodingTokens, vocabSizePadded, stream);

    sync_check_cuda_error();

    TopKSamplingKernelParams<T> params;
    params.logProbsPtrs = logitsPtrs;
    params.outputIds = outputIds;
    params.workspace = workspaceSampling;
    params.maxTopK = 1;
    params.batchSize = batchSize;
    params.maxBatchSize = batchSize;
    params.tokensPerStep = decodingTokens;
    params.maxTokensPerStep = maxDecodingTokens;
    params.maxSeqLen = maxDecodingTokens;
    params.vocabSizePadded = vocabSizePadded;

    invokeBatchTopKSampling(params, stream);

    sync_check_cuda_error();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleSampleAndAcceptDraftTokensPlugin::doTypicalAcceptance(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const maxNumTokens = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const vocabSizePadded = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[1];

    auto const batchSize = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[1];
    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[2];
    auto const maxDraftPathLen = maxPathLen - 1;

    auto logits = static_cast<T const*>(inputs[getIdx(InputIdxEntry::LOGITS)]);
    auto prevDraftLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::DRAFT_LENS)]);

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace);
    size_t offset{0};

    auto const primarySamplingWorkspaceSize
        = getTypicalAcceptanceWorkspaceSize<T>(batchSize, maxDecodingTokens, vocabSizePadded);

    TokenIdType* outputIds = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(TokenIdType)));
    void* workspaceSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, primarySamplingWorkspaceSize));
    T** logitsPtrs = reinterpret_cast<T**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(T*)));
    SizeType32* decodingTokens
        = reinterpret_cast<SizeType32*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(SizeType32)));

    invokeAssembleTargetLogitsOffsets(const_cast<T const**>(logitsPtrs), decodingTokens, logits, prevDraftLens,
        batchSize, maxDecodingTokens, vocabSizePadded, stream);

    sync_check_cuda_error();

    TypicalAcceptanceSampling<T> params;
    params.logitsPtrs = logitsPtrs;
    params.generationLengths = decodingTokens;
    params.temperatures = reinterpret_cast<float const*>(inputs[getIdx(InputIdxEntry::TEMPERATURE)]);
    params.posteriorThresholds = reinterpret_cast<float const*>(inputs[getIdx(InputIdxEntry::POSTERIOR_THRESHOLD)]);
    params.posteriorAlphas = reinterpret_cast<float const*>(inputs[getIdx(InputIdxEntry::POSTERIOR_ALPHA)]);
    params.outputIds = outputIds;
    params.workspace = reinterpret_cast<int8_t*>(workspaceSampling);
    params.randomVals = reinterpret_cast<float const*>(inputs[getIdx(InputIdxEntry::RAND_VALIDATION)]);

    params.batchSize = batchSize;
    params.maxBatchSize = batchSize;
    params.maxDecodingTokens = maxDecodingTokens;
    params.vocabSize = vocabSizePadded;

    if (mSmCnt <= 0)
    {
        auto const deviceId = suggestify::common::getDevice();
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
        mSmCnt = prop.multiProcessorCount;
    }
    params.smCnt = mSmCnt;

    params.checkParams();

    typicalAcceptanceSampling(params, stream);

    sync_check_cuda_error();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleSampleAndAcceptDraftTokensPlugin::acceptDraftTokens(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const maxNumTokens = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const vocabSizePadded = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[1];

    auto const batchSize = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[1];
    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[2];
    auto const maxDraftPathLen = maxPathLen - 1;

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace);
    size_t offset{0};

    auto const samplingWorkspaceSize
        = getTopKWorkspaceSize<T>(batchSize, maxDecodingTokens, 1, vocabSizePadded);

    TokenIdType* outputIds = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(TokenIdType)));

    AcceptDraftTokensByIdsWithPathsParams<T> params;
    params.outputIds = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::ACCEPTED_TOKENS)]);
    params.draftIds = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::DRAFT_TOKEN_IDS)]);
    params.targetIds = outputIds;
    params.acceptedLengths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::ACCEPTED_LENS)]);
    params.paths = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PATHS)]);
    params.bestPathIds = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::BEST_ACCEPTED_PATHS)]);
    params.batchSize = batchSize;
    params.maxBatchSize = batchSize;
    params.vocabSize = vocabSizePadded;
    params.maxSeqLen = maxPathLen;
    params.maxDraftPathLen = maxDraftPathLen;
    params.maxDecodingTokens = maxDecodingTokens;
    params.stream = stream;

    params.checkParams();

    acceptDraftTokensByIdsWithPaths(params);

    cudaMemcpyAsync(outputs[getIdx(OutputIdxEntry::NEXT_DRAFT_PATHS)], inputs[getIdx(InputIdxEntry::PATHS)],
        batchSize * maxDecodingTokens * maxPathLen * sizeof(SizeType32), cudaMemcpyDeviceToDevice, stream);

    sync_check_cuda_error();

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleSampleAndAcceptDraftTokensPlugin::enqueueType(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const greedySampling = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::GREEDY_SAMPLING)])[0];
    if (greedySampling)
    {
        samplePrimeHeadTokens<T>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        doTypicalAcceptance<T>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    acceptDraftTokens<T>(inputDesc, outputDesc, inputs, outputs, workspace, stream);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

int EagleSampleAndAcceptDraftTokensPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    auto const logitsType = inputDesc[getIdx(InputIdxEntry::LOGITS)].type;
    if (logitsType == nvinfer1::DataType::kFLOAT)
    {
        enqueueType<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (logitsType == nvinfer1::DataType::kHALF)
    {
        enqueueType<__half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        CHECK_WITH_INFO(false, "Unsupported logits type");
    }

    return 0;
}

nvinfer1::DataType EagleSampleAndAcceptDraftTokensPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    CHECK(index < 7);
    return inputTypes[getIdx(InputIdxEntry::DRAFT_TOKEN_IDS)];
}


char const* EagleSampleAndAcceptDraftTokensPlugin::getPluginType() const noexcept
{
    return EAGLE_SAMPLE_AND_ACCEPT_DRAFT_TOKENS_PLUGIN_NAME;
}

char const* EagleSampleAndAcceptDraftTokensPlugin::getPluginVersion() const noexcept
{
    return EAGLE_SAMPLE_AND_ACCEPT_DRAFT_TOKENS_PLUGIN_VERSION;
}

int EagleSampleAndAcceptDraftTokensPlugin::getNbOutputs() const noexcept
{
    return 7;
}

int EagleSampleAndAcceptDraftTokensPlugin::initialize() noexcept
{
    return 0;
}

void EagleSampleAndAcceptDraftTokensPlugin::terminate() noexcept {}

size_t EagleSampleAndAcceptDraftTokensPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDtype);
}

void EagleSampleAndAcceptDraftTokensPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDtype);
    assert(d == a + getSerializationSize());
}

void EagleSampleAndAcceptDraftTokensPlugin::destroy() noexcept
{
    delete this;
}


EagleSampleAndAcceptDraftTokensPluginCreator::EagleSampleAndAcceptDraftTokensPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EagleSampleAndAcceptDraftTokensPluginCreator::getPluginName() const noexcept
{
    return EAGLE_SAMPLE_AND_ACCEPT_DRAFT_TOKENS_PLUGIN_NAME;
}

char const* EagleSampleAndAcceptDraftTokensPluginCreator::getPluginVersion() const noexcept
{
    return EAGLE_SAMPLE_AND_ACCEPT_DRAFT_TOKENS_PLUGIN_VERSION;
}

PluginFieldCollection const* EagleSampleAndAcceptDraftTokensPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* EagleSampleAndAcceptDraftTokensPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {
        auto* obj = new EagleSampleAndAcceptDraftTokensPlugin(type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EagleSampleAndAcceptDraftTokensPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new EagleSampleAndAcceptDraftTokensPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
