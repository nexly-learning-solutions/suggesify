#include "eagleDecodeDraftTokensPlugin.h"

#include "../common/assert.h"
#include "../common/dataType.h"
#include "../common/memoryUtils.h"
#include "../src/samplingTopKKernels.h"
#include "../src/speculativeDecoding/eagleDecodingKernels.h"
#include "../src/speculativeDecoding/medusaDecodingKernels.h"
#include "../runtime/common.h"
#include "../runtime/iTensor.h"

using namespace nvinfer1;
using suggestify::plugins::EagleDecodeDraftTokensPluginCreator;
using suggestify::plugins::EagleDecodeDraftTokensPlugin;
using namespace suggestify::kernels;
using namespace suggestify::kernels::speculative_decoding;
using namespace suggestify::runtime;
namespace tc = suggestify::common;

static char const* EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_VERSION{"1"};
static char const* EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_NAME{"EagleDecodeDraftTokens"};
PluginFieldCollection EagleDecodeDraftTokensPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EagleDecodeDraftTokensPluginCreator::mPluginAttributes;

EagleDecodeDraftTokensPlugin::EagleDecodeDraftTokensPlugin(
    nvinfer1::DataType type, int32_t layerIdx, int32_t numEagleLayers, bool topKSampling)
    : mDtype(type)
    , mLayerIdx(layerIdx)
    , mNumEagleLayers(numEagleLayers)
    , mTopKSampling(topKSampling)
{
    TLLM_CHECK_WITH_INFO(mTopKSampling, "Multinomial sampling is not supported yet.");
}

EagleDecodeDraftTokensPlugin::EagleDecodeDraftTokensPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDtype);
    read(d, mLayerIdx);
    read(d, mNumEagleLayers);
    read(d, mTopKSampling);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        static_cast<int>(length), static_cast<int>(d - a));
}

nvinfer1::IPluginV2DynamicExt* EagleDecodeDraftTokensPlugin::clone() const noexcept
{
    auto* plugin = new EagleDecodeDraftTokensPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs EagleDecodeDraftTokensPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex < getNbOutputs());
    TLLM_CHECK(nbInputs == 13);
    auto const batchSizeExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[0];
    auto const maxDecodingTokensExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[1];
    auto const maxPathLengthExpr = inputs[getIdx(InputIdxEntry::PATHS)].d[2];
    auto const maxDecodingDraftTokensExpr
        = exprBuilder.operation(DimensionOperation::kSUB, *maxDecodingTokensExpr, *exprBuilder.constant(1));

    auto const numEagleLayersExpr
        = exprBuilder.operation(DimensionOperation::kSUB, *maxPathLengthExpr, *exprBuilder.constant(1));
    auto const maxDecodingDraftTokensSquareExpr
        = exprBuilder.operation(DimensionOperation::kPROD, *maxDecodingDraftTokensExpr,
            *maxDecodingDraftTokensExpr);

    nvinfer1::DimsExprs ret;
    if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_DRAFT_TOKEN_IDS))
    {
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_DRAFT_LENS))
    {
        ret.nbDims = 1;
        ret.d[0] = batchSizeExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_PATHS))
    {
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingTokensExpr;
        ret.d[2] = maxPathLengthExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES))
    {
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_NEXT_EXPAND_INDICES))
    {
        ret.nbDims = 2;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = maxDecodingDraftTokensExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES))
    {
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = numEagleLayersExpr;
        ret.d[2] = maxDecodingDraftTokensSquareExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS))
    {
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = numEagleLayersExpr;
        ret.d[2] = maxDecodingDraftTokensSquareExpr;
    }
    else if (outputIndex == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR))
    {
        ret.nbDims = 3;
        ret.d[0] = batchSizeExpr;
        ret.d[1] = numEagleLayersExpr;
        ret.d[2] = maxDecodingDraftTokensSquareExpr;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "Wrong outputIndex %d in EagleDecodeDraftTokensPlugin::getOutputDimensions", outputIndex);
    }
    return ret;
}

bool EagleDecodeDraftTokensPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(nbInputs == 13 && nbOutputs == getNbOutputs());
    TLLM_CHECK(pos < nbInputs + nbOutputs);

    if (pos == getIdx(InputIdxEntry::LOGITS))
    {
        return (inOut[pos].type == mDtype) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == getIdx(InputIdxEntry::RAND_SAMPLE) || pos == getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES)
        || pos == getIdx(InputIdxEntry::INPUT_PREV_SCORES)
        || pos == nbInputs + getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES)
        || pos == nbInputs + getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES))
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == getIdx(InputIdxEntry::USE_DYNAMIC_TREE))
    {
        return (inOut[pos].type == nvinfer1::DataType::kBOOL) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT32) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void EagleDecodeDraftTokensPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

template <typename T>
size_t EagleDecodeDraftTokensPlugin::getWorkspaceSizeType(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    size_t workspaceSize{0};
    auto const numInputLogits = inputs[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const batchSize = inputs[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const vocabSizePadded = inputs[getIdx(InputIdxEntry::LOGITS)].dims.d[1];
    auto const maxDecodingTokens = inputs[getIdx(InputIdxEntry::PATHS)].dims.d[1];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    auto const maxTopK = maxDecodingDraftTokens;
    auto const mNumEagleLayers = inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES)].dims.d[1];

    if (mTopKSampling)
    {
        auto const draftTokenSamplingWorkspaceSize
            = getTopKWorkspaceSize<T>(numInputLogits, 1, maxTopK, vocabSizePadded);

        auto const topKsSize = numInputLogits * sizeof(SizeType32);

        auto const topKOffsetSize = batchSize * sizeof(SizeType32);

        auto const logitsPtrsSize = numInputLogits * sizeof(T*);

        auto const firstTopKOutputIdsPtrsSize = numInputLogits * sizeof(TokenIdType*);

        auto const firstTopKOutputIdsSize = numInputLogits * maxDecodingDraftTokens * sizeof(TokenIdType);

        auto const numSuccessorsForEachNodeSize = batchSize * maxDecodingTokens * sizeof(SizeType32);

        auto const skipDecodeSize = numInputLogits * sizeof(bool);

        auto const firstTopKOutputLogProbsSize = numInputLogits * maxDecodingDraftTokens * sizeof(float);

        auto const secondTopKSamplingWorkspaceSize = getTopKWorkspaceSize<float>(
            batchSize, 1, maxTopK, maxTopK * maxTopK);

        auto const secondTopKOutputIdsSize = batchSize * maxDecodingTokens * sizeof(TokenIdType);
        auto const secondTopKOutputIdsPtrSize = batchSize * sizeof(TokenIdType*);
        auto const secondTopKInputScoresPtrsSize = batchSize * sizeof(float*);
        auto const secondTopKOutputLogProbsSize = batchSize * maxDecodingDraftTokens * sizeof(float);

        auto const thirdTopKInputScoresPtrsSize = batchSize * sizeof(float*);
        auto const thirdTopKOutputIdsSize = batchSize * maxDecodingDraftTokens * sizeof(TokenIdType);
        auto const thirdTopKOutputIdsPtrsSize = batchSize * sizeof(TokenIdType*);
        auto const thridTopKSamplingWorkspaceSize = getTopKWorkspaceSize<float>(batchSize, 1,
 maxDecodingDraftTokens, mNumEagleLayers * maxDecodingDraftTokens * maxDecodingDraftTokens);

        SizeType32 constexpr NUM_BUFFERS{18};
        size_t workspaces[NUM_BUFFERS];
        workspaces[0] = draftTokenSamplingWorkspaceSize;
        workspaces[1] = topKsSize;
        workspaces[2] = topKOffsetSize;
        workspaces[3] = logitsPtrsSize;
        workspaces[4] = firstTopKOutputIdsPtrsSize;
        workspaces[5] = firstTopKOutputIdsSize;
        workspaces[6] = numSuccessorsForEachNodeSize;
        workspaces[7] = skipDecodeSize;
        workspaces[8] = firstTopKOutputLogProbsSize;
        workspaces[9] = secondTopKSamplingWorkspaceSize;
        workspaces[10] = secondTopKOutputIdsSize;
        workspaces[11] = secondTopKOutputIdsPtrSize;
        workspaces[12] = secondTopKInputScoresPtrsSize;
        workspaces[13] = secondTopKOutputLogProbsSize;
        workspaces[14] = thirdTopKInputScoresPtrsSize;
        workspaces[15] = thirdTopKOutputIdsSize;
        workspaces[16] = thirdTopKOutputIdsPtrsSize;
        workspaces[17] = thridTopKSamplingWorkspaceSize;
        workspaceSize = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Multinomial sampling is not supported yet.");
    }

    return workspaceSize;
}

size_t EagleDecodeDraftTokensPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
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
        TLLM_CHECK_WITH_INFO(false, "Unsupported logits type");
    }
    return 0;
}

template <typename T>
void EagleDecodeDraftTokensPlugin::doTopKSampling(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const numInputLogits = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[0];
    auto const vocabSizePadded = inputDesc[getIdx(InputIdxEntry::LOGITS)].dims.d[1];
    auto const batchSize = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[0];
    auto const maxDecodingTokens = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[1];
    auto const maxPathLen = inputDesc[getIdx(InputIdxEntry::PATHS)].dims.d[2];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    auto const maxTopK = maxDecodingDraftTokens;

    auto pluginInputLogits = static_cast<T const*>(inputs[getIdx(InputIdxEntry::LOGITS)]);
    auto pluginRandSample = static_cast<float const*>(inputs[getIdx(InputIdxEntry::RAND_SAMPLE)]);
    auto pluginInputPaths = static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::PATHS)]);
    auto numValidLogits = static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::NUM_VALID_LOGITS)]);
    auto useDynamicTree = *(static_cast<bool const*>(inputs[getIdx(InputIdxEntry::USE_DYNAMIC_TREE)]));
    auto dynamicTreeMaxTopK = *(static_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::DYNAMIC_TREE_MAX_TOPK)]));
    auto pluginInputDraftTokenIds
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_DRAFT_TOKEN_IDS)]);
    auto pluginInputDraftLens = reinterpret_cast<SizeType32 const*>(inputs[getIdx(InputIdxEntry::INPUT_DRAFT_LENS)]);
    auto pluginInputPrevScores = static_cast<float const*>(inputs[getIdx(InputIdxEntry::INPUT_PREV_SCORES)]);
    auto pluginInputCurrentExpandIndices
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_CURRENT_EXPAND_INDICES)]);
    auto pluginInputAllLayersScores = static_cast<float const*>(inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES)]);
    auto pluginInputAllLayersDraftTokenIds
        = reinterpret_cast<TokenIdType const*>(inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_DRAFT_TOKEN_IDS)]);
    auto pluginInputAllLayersDraftTokenIdsPredecessor = reinterpret_cast<SizeType32 const*>(
        inputs[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR)]);

    if (useDynamicTree)
    {
        if (mLayerIdx == 0)
        {
            TLLM_CHECK(batchSize == numInputLogits);
        }
        else
        {
            TLLM_CHECK(batchSize * dynamicTreeMaxTopK == numInputLogits);
        }
    }

    auto pluginOutputDraftTokenIds
        = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_DRAFT_TOKEN_IDS)]);
    auto pluginOutputDraftLens = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::OUTPUT_DRAFT_LENS)]);
    auto pluginOutputPaths = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::OUTPUT_PATHS)]);
    auto pluginOutputCurrentScores = static_cast<float*>(outputs[getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES)]);
    auto pluginOutputNextExpandIndices
        = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_NEXT_EXPAND_INDICES)]);
    auto pluginOutputAllLayersScores = static_cast<float*>(outputs[getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES)]);
    auto pluginOutputAllLayersDraftTokenIds
        = reinterpret_cast<TokenIdType*>(outputs[getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS)]);
    auto pluginOutputAllLayersDraftTokenIdsPredecessor
        = reinterpret_cast<SizeType32*>(outputs[getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_DRAFT_TOKEN_IDS_PREDECESSOR)]);

    int8_t* workspaceBytePtr = reinterpret_cast<int8_t*>(workspace);
    size_t offset{0};
    auto const samplingWorkspaceSize
        = getTopKWorkspaceSize<T>(numInputLogits, 1, maxTopK, vocabSizePadded);
    void* workspaceSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, samplingWorkspaceSize));

    SizeType32* topKs = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(SizeType32)));

    SizeType32* topKOffset
        = reinterpret_cast<SizeType32*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(SizeType32)));

    T const** logitsPtrs
        = reinterpret_cast<T const**>(tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(T*)));

    TokenIdType** firstTopKOutputIdsPtrs = reinterpret_cast<TokenIdType**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(TokenIdType*)));

    TokenIdType* firstTopKOutputIdsFlatten = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * maxDecodingDraftTokens * sizeof(TokenIdType)));

    SizeType32* numSuccessorsForEachNode = reinterpret_cast<SizeType32*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingTokens * sizeof(SizeType32)));

    bool* skipDecode
        = reinterpret_cast<bool*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * sizeof(bool)));

    float* firstTopKOutputLogProbs = nullptr;
    if (useDynamicTree)
    {
        firstTopKOutputLogProbs = reinterpret_cast<float*>(
            tc::nextWorkspacePtr(workspaceBytePtr, offset, numInputLogits * maxDecodingDraftTokens * sizeof(float)));
    }

    SizeType32 const secondTopKVocabSize = dynamicTreeMaxTopK * maxDecodingDraftTokens;
    auto const secondTopKSamplingWorkspaceSize
        = getTopKWorkspaceSize<float>(batchSize, 1, maxTopK, secondTopKVocabSize);
    void* workspaceScoresSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, secondTopKSamplingWorkspaceSize));

    TokenIdType* secondTopKOutputIdsFlatten = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingDraftTokens * sizeof(TokenIdType)));

    TokenIdType** secondTopKOutputIdsPtrs = reinterpret_cast<TokenIdType**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(TokenIdType*)));

    float** secondTopKInputScoresPtrs
        = reinterpret_cast<float**>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(float*)));

    float* secondTopKOutputLogProbs = reinterpret_cast<float*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingDraftTokens * sizeof(float)));

    float** thirdTopKInputScoresPtrs
        = reinterpret_cast<float**>(tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(float*)));

    TokenIdType* thirdTopKOutputIds = reinterpret_cast<TokenIdType*>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * maxDecodingDraftTokens * sizeof(TokenIdType)));

    TokenIdType** thirdTopKOutputIdsPtrs = reinterpret_cast<TokenIdType**>(
        tc::nextWorkspacePtr(workspaceBytePtr, offset, batchSize * sizeof(TokenIdType*)));

    auto const thridTopKSamplingVocabSize
        = (mNumEagleLayers - 1) * dynamicTreeMaxTopK * dynamicTreeMaxTopK + dynamicTreeMaxTopK;
    auto const thridTopKSamplingWorkspaceSize = getTopKWorkspaceSize<float>(
        batchSize, 1, maxDecodingDraftTokens, thridTopKSamplingVocabSize);
    void* workspaceThirdTopKSampling
        = reinterpret_cast<void*>(tc::nextWorkspacePtr(workspaceBytePtr, offset, thridTopKSamplingWorkspaceSize));

    invokeAssembleDraftLogitsOffsets(logitsPtrs, pluginInputLogits, firstTopKOutputIdsPtrs, firstTopKOutputIdsFlatten,
        skipDecode, numValidLogits, numInputLogits, batchSize, maxDecodingDraftTokens, vocabSizePadded, stream);
    sync_check_cuda_error();

    if (useDynamicTree)
    {
        invokeSetTopKsFromDyanmicTreeMaxTopK(
            mLayerIdx, batchSize, numInputLogits, topKs, topKOffset, dynamicTreeMaxTopK, stream);
        sync_check_cuda_error();

        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = const_cast<T*>(pluginInputLogits);
        biasSoftmaxParams.logitsPtrs = nullptr;
        biasSoftmaxParams.probs = const_cast<T*>(pluginInputLogits);
        biasSoftmaxParams.maxBeamWidth = 1;
        biasSoftmaxParams.batchSlots = nullptr;
        biasSoftmaxParams.batchSize = numInputLogits;
        biasSoftmaxParams.maxBatchSize = numInputLogits;
        biasSoftmaxParams.vocabSize = vocabSizePadded;
        biasSoftmaxParams.vocabSizePadded = vocabSizePadded;
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = false;
        biasSoftmaxParams.checkParams();

        invokeAddBiasSoftMax(biasSoftmaxParams, stream);
        sync_check_cuda_error();
    }
    else
    {
        invokeExtractTopKsFromPath(pluginInputPaths, topKs, topKOffset, numSuccessorsForEachNode, mLayerIdx, batchSize,
            maxDecodingTokens, maxPathLen, stream);
        sync_check_cuda_error();
    }

    TopKSamplingKernelParams<T> params{};
    params.logProbsPtrs = logitsPtrs;
    params.outputIdsPtrs = firstTopKOutputIdsPtrs;
    params.workspace = workspaceSampling;
    params.maxTopK = maxTopK;
    params.topKs = topKs;
    params.batchSize = numInputLogits;
    params.maxBatchSize = numInputLogits;
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = vocabSizePadded;
    params.returnAllSelectedTokens = true;
    params.skipDecode = skipDecode;
    params.outputLogProbs = firstTopKOutputLogProbs;
    params.logitsHasProbs = true;

    invokeBatchTopKSampling(params, stream);
    sync_check_cuda_error();

    if (useDynamicTree)
    {
        if (mLayerIdx != 0)
        {
            invokeUpdateScores(batchSize, numInputLogits, dynamicTreeMaxTopK, maxDecodingDraftTokens,
                firstTopKOutputLogProbs, pluginInputPrevScores, stream);
            sync_check_cuda_error();



            invokeAssembleSecondTopKSamplingInputs(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens,
                firstTopKOutputLogProbs, secondTopKInputScoresPtrs, secondTopKOutputIdsFlatten, secondTopKOutputIdsPtrs,
                stream);
            sync_check_cuda_error();

            TopKSamplingKernelParams<float> params{};
            params.logProbsPtrs = secondTopKInputScoresPtrs;
            params.outputIdsPtrs = secondTopKOutputIdsPtrs;
            params.workspace = workspaceScoresSampling;
            params.maxTopK = maxTopK;
            params.topKs = topKs;
            params.batchSize = batchSize;
            params.maxBatchSize = batchSize;
            params.maxTokensPerStep = 1;
            params.vocabSizePadded = secondTopKVocabSize;
            params.returnAllSelectedTokens = true;

            invokeBatchTopKSampling(params, stream);
            sync_check_cuda_error();
        }

        invokeCopyScoresAndDraftTokenIds(mLayerIdx, mNumEagleLayers, maxDecodingDraftTokens, batchSize, numInputLogits,
            dynamicTreeMaxTopK, topKOffset,
            pluginInputCurrentExpandIndices,
            pluginInputAllLayersScores, pluginInputAllLayersDraftTokenIds, pluginInputAllLayersDraftTokenIdsPredecessor,
            pluginOutputAllLayersScores, pluginOutputAllLayersDraftTokenIds,
            pluginOutputAllLayersDraftTokenIdsPredecessor,
            firstTopKOutputLogProbs,
            firstTopKOutputIdsFlatten,
            stream);
        sync_check_cuda_error();

        if (mLayerIdx != mNumEagleLayers - 1)
        {
            invokeUpdatePath(mLayerIdx, batchSize, dynamicTreeMaxTopK, maxDecodingTokens, maxPathLen, pluginInputPaths,
                pluginOutputPaths,
                secondTopKOutputIdsPtrs,
                pluginOutputNextExpandIndices, stream);
            sync_check_cuda_error();
        }

        if (mLayerIdx != 0)
        {
            invokeExtractScoresAndRealDraftTokensIds(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens,
                secondTopKInputScoresPtrs, secondTopKOutputIdsPtrs, firstTopKOutputIdsFlatten, secondTopKOutputLogProbs,
                stream);
            sync_check_cuda_error();
        }

        invokeUpdateDraftTokensAndLensAndCurScores(mLayerIdx, batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens,
            mLayerIdx == 0 ? firstTopKOutputIdsPtrs : secondTopKOutputIdsPtrs, pluginInputDraftTokenIds,
            pluginInputDraftLens, pluginOutputDraftTokenIds, pluginOutputDraftLens,
            mLayerIdx == 0 ? firstTopKOutputLogProbs : secondTopKOutputLogProbs, pluginOutputCurrentScores, stream);
        sync_check_cuda_error();

        if (mLayerIdx == mNumEagleLayers - 1)
        {
            invokeAssembleThridTopKSamplingInputs(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens,
                mNumEagleLayers, pluginOutputAllLayersScores, thirdTopKInputScoresPtrs, thirdTopKOutputIds,
                thirdTopKOutputIdsPtrs, stream);
            sync_check_cuda_error();

            TopKSamplingKernelParams<float> params{};
            params.logProbsPtrs = thirdTopKInputScoresPtrs;
            params.outputIdsPtrs = thirdTopKOutputIdsPtrs;
            params.workspace = workspaceThirdTopKSampling;
            params.maxTopK = maxDecodingDraftTokens;
            params.batchSize = batchSize;
            params.maxBatchSize = batchSize;
            params.maxTokensPerStep = 1;
            params.vocabSizePadded = thridTopKSamplingVocabSize;
            params.returnAllSelectedTokens = true;

            invokeBatchTopKSampling(params, stream);
            sync_check_cuda_error();

            invokeReconstructFinalPath(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens, maxDecodingTokens,
                maxPathLen, mNumEagleLayers, thirdTopKOutputIdsPtrs, pluginOutputAllLayersDraftTokenIdsPredecessor,
                pluginOutputPaths, stream);
            sync_check_cuda_error();

            invokeCopyFinalDraftTokens(batchSize, dynamicTreeMaxTopK, maxDecodingDraftTokens, mNumEagleLayers,
                thirdTopKOutputIdsPtrs, pluginOutputAllLayersDraftTokenIds, pluginOutputDraftTokenIds,
                pluginOutputDraftLens, stream);
            sync_check_cuda_error();
        }
    }
    else
    {
        invokeCopyOutputTokensIds(firstTopKOutputIdsPtrs, topKs, topKOffset, pluginInputDraftTokenIds,
            pluginInputDraftLens, numValidLogits, pluginOutputDraftTokenIds, pluginOutputDraftLens, mLayerIdx,
            batchSize, maxDecodingDraftTokens, stream);
        sync_check_cuda_error();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void EagleDecodeDraftTokensPlugin::enqueueType(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mTopKSampling)
    {
        doTopKSampling<T>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Multinomial sampling is not supported yet");
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

int EagleDecodeDraftTokensPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
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
        TLLM_CHECK_WITH_INFO(false, "Unsupported logits type");
    }

    return 0;
}

nvinfer1::DataType EagleDecodeDraftTokensPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index < getNbOutputs());
    TLLM_CHECK(index < getNbOutputs());
    if (index == getIdx(OutputIdxEntry::OUTPUT_ALL_LAYERS_SCORES)
        || index == getIdx(OutputIdxEntry::OUTPUT_CURRENT_SCORES))
    {
        return inputTypes[getIdx(InputIdxEntry::INPUT_ALL_LAYERS_SCORES)];
    }
    else
    {
        return inputTypes[getIdx(InputIdxEntry::PATHS)];
    }
}


char const* EagleDecodeDraftTokensPlugin::getPluginType() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_NAME;
}

char const* EagleDecodeDraftTokensPlugin::getPluginVersion() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_VERSION;
}

int EagleDecodeDraftTokensPlugin::getNbOutputs() const noexcept
{
    return 8;
}

int EagleDecodeDraftTokensPlugin::initialize() noexcept
{
    return 0;
}

void EagleDecodeDraftTokensPlugin::terminate() noexcept {}

size_t EagleDecodeDraftTokensPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDtype) + sizeof(mLayerIdx) + sizeof(mNumEagleLayers) + sizeof(mTopKSampling);
}

void EagleDecodeDraftTokensPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDtype);
    write(d, mLayerIdx);
    write(d, mNumEagleLayers);
    write(d, mTopKSampling);
    TLLM_CHECK(d == a + getSerializationSize());
}

void EagleDecodeDraftTokensPlugin::destroy() noexcept
{
    delete this;
}


EagleDecodeDraftTokensPluginCreator::EagleDecodeDraftTokensPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("layer_idx", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("num_eagle_layers", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("top_k_sampling", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EagleDecodeDraftTokensPluginCreator::getPluginName() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_NAME;
}

char const* EagleDecodeDraftTokensPluginCreator::getPluginVersion() const noexcept
{
    return EAGLE_DECODE_DRAFT_TOKENS_PLUGIN_VERSION;
}

PluginFieldCollection const* EagleDecodeDraftTokensPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* EagleDecodeDraftTokensPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int32_t layerIdx;
    int32_t numEagleLayers;
    nvinfer1::DataType type;
    bool topKSampling;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "layer_idx"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            layerIdx = *static_cast<int32_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "num_eagle_layers"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            numEagleLayers = *static_cast<int32_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "top_k_sampling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            topKSampling = static_cast<bool>(*static_cast<int32_t const*>(fields[i].data));
        }
    }

    try
    {
        auto* obj = new EagleDecodeDraftTokensPlugin(type, layerIdx, numEagleLayers, topKSampling);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* EagleDecodeDraftTokensPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new EagleDecodeDraftTokensPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
