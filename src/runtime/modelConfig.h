
#pragma once

#include "../common/quantization.h"
#include "common.h"
#include "lookaheadModule.h"
#include "loraModule.h"
#include "speculativeDecodingMode.h"

#include <NvInferRuntime.h>
#include <array>

namespace suggestify::runtime
{

class ModelConfig
{
public:
    static constexpr std::array kOPT_PROFILES_SPLIT_POINTS{64, 128, 256, 512, 1024};
    static constexpr SizeType32 kDEFAULT_NUM_TOKENS_PER_BLOCK = 64;

    enum class ModelVariant : std::int32_t
    {
        kGpt = 0,
        kChatGlm = 1,
        kGlm = 2,
        kMamba = 3,
        kRecurrentGemma = 4,
        kEncDec = 5,
    };

    struct RnnConfig
    {
        SizeType32 stateSize = 0;
        SizeType32 convKernel = 0;
        SizeType32 rnnHiddenSize = 0;
        SizeType32 rnnHeadSize = 0;
        SizeType32 rnnConvDimSize = 0;
    };

    enum class LayerType : std::int32_t
    {
        kATTENTION,
        kRECURRENT,
        kLINEAR,
        kNOOP,
    };

    enum class KVCacheType : std::int32_t
    {
        kCONTINUOUS,
        kPAGED,
        kDISABLED,
    };

    static KVCacheType KVCacheTypeFromString(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), ::toupper);

        if (value == "CONTINUOUS")
        {
            return KVCacheType::kCONTINUOUS;
        }
        if (value == "PAGED")
        {
            return KVCacheType::kPAGED;
        }
        if (value == "DISABLED")
        {
            return KVCacheType::kDISABLED;
        }

        throw std::invalid_argument("Invalid KV cache type: " + value);
    }

    enum class ManageWeightsType : std::int32_t
    {
        kDisabled,
        kEnabled,
    };

    explicit ModelConfig(SizeType32 vocabSize, SizeType32 nbLayers, SizeType32 nbAttentionLayers,
        SizeType32 nbRnnLayers, SizeType32 nbHeads, SizeType32 hiddenSize, nvinfer1::DataType dtype)
        : mVocabSize(vocabSize)
        , mNbLayers(nbLayers)
        , mNbAttentionLayers(nbAttentionLayers)
        , mNbRnnLayers(nbRnnLayers)
        , mNbHeads(nbHeads)
        , mHiddenSize(hiddenSize)
        , mSizePerHead(mHiddenSize / mNbHeads)
        , mDataType(dtype)
        , mUseGptAttentionPlugin(false)
        , mUseMambaConv1dPlugin(false)
        , mInputPacked{false}
        , mTokensPerBlock{kDEFAULT_NUM_TOKENS_PER_BLOCK}
        , mQuantMode{common::QuantMode::none()}
        , mMaxBatchSize(0)
        , mMaxBeamWidth(0)
        , mMaxInputLen(0)
        , mMaxSequenceLen(0)
        , mMaxNumTokens(std::nullopt)
        , mComputeContextLogits(false)
        , mComputeGenerationLogits(false)
        , mModelVariant(ModelVariant::kGpt)
        , mMaxPromptEmbeddingTableSize(0)
        , mUseMrope{false}
        , mMaxPositionEmbeddings(0)
        , mRotaryEmbeddingDim(0)
        , mContextFMHA(false)
        , mPagedContextFMHA(false)
        , mUseXQA{false}
        , mPpReduceScatter{false}
        , mUseLoraPlugin(false)
        , mMlpHiddenSize(0)
        , mUseCrossAttention(false)
        , mUsePositionEmbedding(false)
        , mUseTokenTypeEmbedding(false)
        , mSpeculativeDecodingMode(SpeculativeDecodingMode::None())
        , mLogitsDtype(nvinfer1::DataType::kFLOAT)
        , mUseShapeInference(true)
        , mManageWeightsType(ManageWeightsType::kDisabled)
        , mSkipCrossAttnBlocks(false)
    {
        CHECK_WITH_INFO(mNbLayers >= mNbAttentionLayers + mNbRnnLayers,
            "Number of layers (%d) expected to be >= number of attention (%d) + number of rnn layers (%d)", mNbLayers,
            mNbAttentionLayers, mNbRnnLayers);
        setNbKvHeads(mNbHeads);
    }

    [[nodiscard]] static std::vector<SizeType32> getOptProfilesSplitPoints() noexcept
    {
        return {kOPT_PROFILES_SPLIT_POINTS.begin(), kOPT_PROFILES_SPLIT_POINTS.end()};
    }

    [[nodiscard]] SizeType32 constexpr getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    [[nodiscard]] SizeType32 constexpr getVocabSizePadded(SizeType32 worldSize) const noexcept
    {
        return (mVocabSize + worldSize - 1) / worldSize * worldSize;
    }

    [[nodiscard]] SizeType32 countLocalLayers(
        LayerType layerType, SizeType32 pipelineParallelism = 1, SizeType32 pipelineParallelismRank = 0) const
    {
        CHECK_WITH_INFO(pipelineParallelism > 0, "Invalid pipelineParallelism: %d", pipelineParallelism);
        auto const numLocalLayers = mNbLayers / pipelineParallelism;
        auto const firstLocalLayerIt = mLayerTypes.cbegin() + (numLocalLayers * pipelineParallelismRank);
        return std::count(firstLocalLayerIt, firstLocalLayerIt + numLocalLayers, layerType);
    }

    [[nodiscard]] SizeType32 countLowerRankLayers(
        LayerType layerType, SizeType32 pipelineParallelism = 1, SizeType32 pipelineParallelismRank = 0) const
    {
        auto const numLocalLayers = mNbLayers / pipelineParallelism;
        auto const firstLocalLayer = numLocalLayers * pipelineParallelismRank;
        return std::count(mLayerTypes.cbegin(), mLayerTypes.cbegin() + firstLocalLayer, layerType);
    }

    [[nodiscard]] SizeType32 getNbLayers(SizeType32 pipelineParallelism = 1) const
    {
        return mNbLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType32 getNbAttentionLayers(
        SizeType32 pipelineParallelism = 1, SizeType32 pipelineParallelismRank = 0) const
    {
        if (mLayerTypes.empty())
        {
            LOG_DEBUG("Assuming uniform distribution of attention layers between ranks");
            return mNbAttentionLayers / pipelineParallelism;
        }
        return countLocalLayers(LayerType::kATTENTION, pipelineParallelism, pipelineParallelismRank);
    }

    [[nodiscard]] SizeType32 getNbRnnLayers(
        SizeType32 pipelineParallelism = 1, SizeType32 pipelineParallelismRank = 0) const
    {
        if (mLayerTypes.empty())
        {
            LOG_DEBUG("Assuming uniform distribution of recurrent layers between ranks");
            return mNbRnnLayers / pipelineParallelism;
        }
        return countLocalLayers(LayerType::kRECURRENT, pipelineParallelism, pipelineParallelismRank);
    }

    [[nodiscard]] SizeType32 constexpr getNbHeads() const noexcept
    {
        return mNbHeads;
    }

    [[nodiscard]] SizeType32 getNbKvHeads(SizeType32 layerIdx) const
    {
        CHECK_WITH_INFO(layerIdx < mNbAttentionLayers, "Layer index %d is out of bounds", layerIdx);
        return mNumKvHeadsPerAttentionLayer[layerIdx];
    }

    void setNbKvHeads(SizeType32 nbKvHeads)
    {
        mNumKvHeadsPerAttentionLayer = std::vector<SizeType32>(mNbAttentionLayers, nbKvHeads);
    }

    void setNbCrossKvHeads(SizeType32 nbKvHeads)
    {
        mNumKvHeadsPerCrossAttentionLayer = std::vector<SizeType32>(mNbAttentionLayers, nbKvHeads);
    }

    [[nodiscard]] SizeType32 constexpr getHiddenSize() const noexcept
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType32 constexpr getEncoderHiddenSize() const noexcept
    {
        return mEncoderHiddenSize;
    }

    void constexpr setEncoderHiddenSize(SizeType32 encoderHiddenSize) noexcept
    {
        mEncoderHiddenSize = encoderHiddenSize;
    }

    [[nodiscard]] SizeType32 constexpr getSizePerHead() const noexcept
    {
        return mSizePerHead;
    }

    void constexpr setSizePerHead(SizeType32 sizePerHead) noexcept
    {
        mSizePerHead = sizePerHead;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept
    {
        return mUseGptAttentionPlugin;
    }

    void constexpr useGptAttentionPlugin(bool useGptAttentionPlugin) noexcept
    {
        mUseGptAttentionPlugin = useGptAttentionPlugin;
    }

    [[nodiscard]] bool constexpr useMambaConv1dPlugin() const noexcept
    {
        return mUseMambaConv1dPlugin;
    }

    void constexpr useMambaConv1dPlugin(bool useMambaConv1dPlugin) noexcept
    {
        mUseMambaConv1dPlugin = useMambaConv1dPlugin;
    }

    [[nodiscard]] bool constexpr usePackedInput() const noexcept
    {
        return mInputPacked;
    }

    void constexpr usePackedInput(bool inputPacked) noexcept
    {
        mInputPacked = inputPacked;
    }

    [[nodiscard]] bool constexpr usePagedState() const noexcept
    {
        return mPagedState;
    }

    void constexpr usePagedState(bool pagedState) noexcept
    {
        mPagedState = pagedState;
    }

    [[nodiscard]] SizeType32 constexpr getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    void constexpr setTokensPerBlock(SizeType32 TokensPerBlock) noexcept
    {
        mTokensPerBlock = TokensPerBlock;
    }

    [[nodiscard]] common::QuantMode constexpr getQuantMode() const noexcept
    {
        return mQuantMode;
    }

    void constexpr setQuantMode(common::QuantMode QuantMode) noexcept
    {
        mQuantMode = QuantMode;
    }

    [[nodiscard]] bool constexpr supportsInflightBatching() const noexcept
    {
        return (isTransformerBased() && mUseGptAttentionPlugin && mInputPacked
                   && (mKVCacheType == KVCacheType::kDISABLED || mKVCacheType == KVCacheType::kPAGED))
            || (isRnnBased() && mUseMambaConv1dPlugin && mInputPacked && mPagedState);
    }

    [[nodiscard]] SizeType32 constexpr getMaxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    void constexpr setMaxBatchSize(SizeType32 maxBatchSize) noexcept
    {
        mMaxBatchSize = maxBatchSize;
    }

    [[nodiscard]] SizeType32 constexpr getMaxBeamWidth() const noexcept
    {
        return mMaxBeamWidth;
    }

    void constexpr setMaxBeamWidth(SizeType32 maxBeamWidth) noexcept
    {
        mMaxBeamWidth = maxBeamWidth;
    }

    [[nodiscard]] SizeType32 constexpr getMaxInputLen() const noexcept
    {
        return mMaxInputLen;
    }

    void constexpr setMaxInputLen(SizeType32 maxInputLen) noexcept
    {
        mMaxInputLen = maxInputLen;
    }

    [[nodiscard]] SizeType32 constexpr getMaxSequenceLen() const noexcept
    {
        return mMaxSequenceLen;
    }

    void constexpr setMaxSequenceLen(SizeType32 maxSequenceLen) noexcept
    {
        mMaxSequenceLen = maxSequenceLen;
    }

    [[nodiscard]] std::optional<SizeType32> constexpr getMaxNumTokens() const noexcept
    {
        return mMaxNumTokens;
    }

    void constexpr setMaxNumTokens(std::optional<SizeType32> maxNumTokens) noexcept
    {
        mMaxNumTokens = maxNumTokens;
    }

    [[nodiscard]] SizeType32 constexpr getMaxEncoderLen() const noexcept
    {
        return mMaxEncoderLen;
    }

    void constexpr setMaxEncoderLen(SizeType32 maxEncoderLen) noexcept
    {
        mMaxEncoderLen = maxEncoderLen;
    }

    [[nodiscard]] bool constexpr usePromptTuning() const noexcept
    {
        return mMaxPromptEmbeddingTableSize > 0;
    }

    [[nodiscard]] bool constexpr useMrope() const noexcept
    {
        return mUseMrope;
    }

    void constexpr setUseMrope(bool useMrope) noexcept
    {
        mUseMrope = useMrope;
    }

    [[nodiscard]] SizeType32 constexpr getMaxPositionEmbeddings() const noexcept
    {
        return mMaxPositionEmbeddings;
    }

    void constexpr setMaxPositionEmbeddings(SizeType32 maxPositionEmbeddings) noexcept
    {
        mMaxPositionEmbeddings = maxPositionEmbeddings;
    }

    [[nodiscard]] SizeType32 constexpr getRotaryEmbeddingDim() const noexcept
    {
        return mRotaryEmbeddingDim;
    }

    void constexpr setRotaryEmbeddingDim(SizeType32 rotaryEmbeddingDim) noexcept
    {
        mRotaryEmbeddingDim = rotaryEmbeddingDim;
    }

    [[nodiscard]] SizeType32 constexpr getMaxPromptEmbeddingTableSize() const noexcept
    {
        return mMaxPromptEmbeddingTableSize;
    }

    void constexpr setMaxPromptEmbeddingTableSize(SizeType32 maxPromptEmbeddingTableSize) noexcept
    {
        mMaxPromptEmbeddingTableSize = maxPromptEmbeddingTableSize;
    }

    [[nodiscard]] bool constexpr computeContextLogits() const noexcept
    {
        return mComputeContextLogits;
    }

    void constexpr computeContextLogits(bool computeContextLogits) noexcept
    {
        mComputeContextLogits = computeContextLogits;
    }

    [[nodiscard]] bool constexpr computeGenerationLogits() const noexcept
    {
        return mComputeGenerationLogits;
    }

    void constexpr computeGenerationLogits(bool computeGenerationLogits) noexcept
    {
        mComputeGenerationLogits = computeGenerationLogits;
    }

    [[nodiscard]] ModelVariant getModelVariant() const
    {
        return mModelVariant;
    }

    void setModelVariant(ModelVariant modelVariant)
    {
        mModelVariant = modelVariant;
    }

    [[nodiscard]] SizeType32 getMaxDecodingDraftTokens() const
    {
        return getSpeculativeDecodingMode().isNone() ? 0 : getSpeculativeDecodingModule().getMaxDecodingDraftTokens();
    }

    [[nodiscard]] SizeType32 constexpr getMaxDecodingTokens() const noexcept
    {
        return getSpeculativeDecodingMode().isNone() ? 1 : getSpeculativeDecodingModule().getMaxDecodingTokens();
    }

    void constexpr setContextFMHA(bool contextFMHA) noexcept
    {
        mContextFMHA = contextFMHA;
    }

    [[nodiscard]] bool constexpr getContextFMHA() const noexcept
    {
        return mContextFMHA;
    }

    void constexpr setPagedContextFMHA(bool pagedContextFMHA) noexcept
    {
        mPagedContextFMHA = pagedContextFMHA;
    }

    [[nodiscard]] bool constexpr getPagedContextFMHA() const noexcept
    {
        return mPagedContextFMHA;
    }

    void constexpr setPpReduceScatter(bool ppReduceScatter) noexcept
    {
        mPpReduceScatter = ppReduceScatter;
    }

    [[nodiscard]] bool constexpr getPpReduceScatter() const noexcept
    {
        return mPpReduceScatter;
    }

    [[nodiscard]] bool constexpr useLoraPlugin() const noexcept
    {
        return mUseLoraPlugin;
    }

    void constexpr useLoraPlugin(bool useLoraPlugin) noexcept
    {
        mUseLoraPlugin = useLoraPlugin;
    }

    [[nodiscard]] std::vector<LoraModule> const& getLoraModules() const noexcept
    {
        return mLoraModules;
    }

    void setLoraModules(std::vector<LoraModule> const& loraModules) noexcept
    {
        mLoraModules = loraModules;
    }

    [[nodiscard]] SizeType32 constexpr getMlpHiddenSize() const noexcept
    {
        return mMlpHiddenSize;
    }

    void constexpr setMlpHiddenSize(SizeType32 mlpHiddenSize) noexcept
    {
        mMlpHiddenSize = mlpHiddenSize;
    }

    [[nodiscard]] bool constexpr isKVCacheEnabled() const noexcept
    {
        return mKVCacheType != KVCacheType::kDISABLED;
    }

    [[nodiscard]] bool constexpr isPagedKVCache() const noexcept
    {
        return mKVCacheType == KVCacheType::kPAGED;
    }

    [[nodiscard]] bool constexpr isContinuousKVCache() const noexcept
    {
        return mKVCacheType == KVCacheType::kCONTINUOUS;
    }

    [[nodiscard]] KVCacheType constexpr getKVCacheType() const noexcept
    {
        return mKVCacheType;
    }

    void constexpr setKVCacheType(KVCacheType kvCacheType) noexcept
    {
        mKVCacheType = kvCacheType;
    }

    [[nodiscard]] bool constexpr useCrossAttention() const noexcept
    {
        return mUseCrossAttention;
    }

    void constexpr setUseCrossAttention(bool useCrossAttention) noexcept
    {
        mUseCrossAttention = useCrossAttention;
    }

    [[nodiscard]] bool constexpr usePositionEmbedding() const noexcept
    {
        return mUsePositionEmbedding;
    }

    void constexpr setUsePositionEmbedding(bool usePositionEmbedding) noexcept
    {
        mUsePositionEmbedding = usePositionEmbedding;
    }

    [[nodiscard]] bool constexpr useTokenTypeEmbedding() const noexcept
    {
        return mUseTokenTypeEmbedding;
    }

    void constexpr setUseTokenTypeEmbedding(bool useTokenTypeEmbedding) noexcept
    {
        mUseTokenTypeEmbedding = useTokenTypeEmbedding;
    }

    [[nodiscard]] SizeType32 constexpr getMaxLoraRank() const noexcept
    {
        return mMaxLoraRank;
    }

    void constexpr setMaxLoraRank(SizeType32 maxLoraRank) noexcept
    {
        mMaxLoraRank = maxLoraRank;
    }

    void setSpeculativeDecodingMode(SpeculativeDecodingMode mode) noexcept
    {
        mSpeculativeDecodingMode = mode;
    }

    [[nodiscard]] bool hasSpeculativeDecodingModule() const noexcept
    {
        return mSpeculativeDecodingModule != nullptr;
    }

    [[nodiscard]] SpeculativeDecodingModule const& getSpeculativeDecodingModule() const noexcept
    {
        CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return *mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<SpeculativeDecodingModule const> getSpeculativeDecodingModulePtr() const noexcept
    {
        CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<SpeculativeDecodingModule> getSpeculativeDecodingModulePtr() noexcept
    {
        CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return mSpeculativeDecodingModule;
    }

    void setSpeculativeDecodingModule(
        std::shared_ptr<SpeculativeDecodingModule> const& speculativeDecodingModule) noexcept
    {
        mSpeculativeDecodingModule = speculativeDecodingModule;
    }

    void resetSpeculativeDecodingModule() noexcept
    {
        mSpeculativeDecodingModule.reset();
    }

    void enableSeamlessLookaheadDecoding(SizeType32 maxDraftTokens) noexcept
    {
        setSpeculativeDecodingMode(SpeculativeDecodingMode::LookaheadDecoding());
        setSpeculativeDecodingModule(std::make_shared<LookaheadModule>(maxDraftTokens, maxDraftTokens));
    }

    void disableSeamlessLookaheadDecoding() noexcept
    {
        setSpeculativeDecodingMode(SpeculativeDecodingMode::None());
        resetSpeculativeDecodingModule();
    }

    [[nodiscard]] nvinfer1::DataType getKvDataType() const noexcept
    {
        if (getQuantMode().hasFp8KvCache())
        {
            return nvinfer1::DataType::kFP8;
        }
        if (getQuantMode().hasInt8KvCache())
        {
            return nvinfer1::DataType::kINT8;
        }

        return getDataType();
    }

    [[nodiscard]] bool constexpr isTransformerBased() const noexcept
    {
        return mModelVariant == ModelVariant::kGpt || mModelVariant == ModelVariant::kGlm
            || mModelVariant == ModelVariant::kChatGlm || mModelVariant == ModelVariant::kRecurrentGemma;
    }

    [[nodiscard]] bool hasRnnConfig() const noexcept
    {
        return mRnnConfig.has_value();
    }

    [[nodiscard]] std::optional<RnnConfig> getRnnConfig() const noexcept
    {
        return mRnnConfig;
    }

    void setRnnConfig(RnnConfig const& rnnConfig) noexcept
    {
        mRnnConfig = rnnConfig;
    }

    [[nodiscard]] bool constexpr isRnnBased() const noexcept
    {
        return mModelVariant == ModelVariant::kMamba || mModelVariant == ModelVariant::kRecurrentGemma;
    }

    [[nodiscard]] std::vector<LayerType> const& getLayerTypes() const noexcept
    {
        return mLayerTypes;
    }

    void setLayerTypes(std::vector<LayerType> const& layerTypes) noexcept
    {
        mLayerTypes = layerTypes;
    }

    [[nodiscard]] SpeculativeDecodingMode constexpr getSpeculativeDecodingMode() const noexcept
    {
        return mSpeculativeDecodingMode;
    }

    void setLogitsDtype(nvinfer1::DataType inputDtype) noexcept
    {
        mLogitsDtype = inputDtype;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getLogitsDtype() const noexcept
    {
        return mLogitsDtype;
    }

    void setUseShapeInference(bool useShapeInference) noexcept
    {
        mUseShapeInference = useShapeInference;
    }

    [[nodiscard]] bool useShapeInference() const noexcept
    {
        return mUseShapeInference;
    }

    [[nodiscard]] ManageWeightsType getManageWeightsType() const noexcept
    {
        return mManageWeightsType;
    }

    void setManageWeightsType(const ManageWeightsType manageWeightType) noexcept
    {
        mManageWeightsType = manageWeightType;
    }

    [[nodiscard]] std::string const& getModelName() const noexcept
    {
        return mModelName;
    }

    void setModelName(std::string const& modelName)
    {
        mModelName = modelName;
    }

    [[nodiscard]] std::vector<SizeType32> const& getNumKvHeadsPerLayer() const
    {
        return mNumKvHeadsPerAttentionLayer;
    }

    [[nodiscard]] std::pair<std::vector<SizeType32>::const_iterator, std::vector<SizeType32>::const_iterator>
    getNumKvHeadsPerLayerLocalRange(
        SizeType32 pipelineParallelism = 1, SizeType32 pipelineParallelismRank = 0, bool isCrossAttention = false) const
    {
        LOG_TRACE("%s start: %d", __PRETTY_FUNCTION__);
        CHECK_WITH_INFO(pipelineParallelism > 0, "Invalid pipelineParallelism: %d", pipelineParallelism);

        auto const numPrevAttnLayers
            = countLowerRankLayers(LayerType::kATTENTION, pipelineParallelism, pipelineParallelismRank);
        auto const firstLocalAttentionLayerIt = isCrossAttention
            ? mNumKvHeadsPerCrossAttentionLayer.cbegin()
            : mNumKvHeadsPerAttentionLayer.cbegin() + numPrevAttnLayers;
        auto const numLocalAttentionLayers
            = countLocalLayers(LayerType::kATTENTION, pipelineParallelism, pipelineParallelismRank);
        LOG_TRACE("%s stop: %d", __PRETTY_FUNCTION__);
        return std::make_pair(firstLocalAttentionLayerIt, firstLocalAttentionLayerIt + numLocalAttentionLayers);
    }

    void setNumKvHeadsPerLayer(std::vector<SizeType32> const& headsPerLayer)
    {
        auto const numElems = static_cast<SizeType32>(headsPerLayer.size());
        CHECK_WITH_INFO(numElems == mNbAttentionLayers,
            "Length of head_per_layer (%d) must match number of attention layers (%d)", numElems, mNbAttentionLayers);
        mNumKvHeadsPerAttentionLayer = headsPerLayer;
    }

    void setNumKvHeadsPerCrossLayer(std::vector<SizeType32> const& headsPerLayer)
    {
        auto const numElems = static_cast<SizeType32>(headsPerLayer.size());
        CHECK_WITH_INFO(numElems == mNbAttentionLayers,
            "Length of head_per_layer (%d) must match number of attention layers (%d)", numElems, mNbAttentionLayers);
        mNumKvHeadsPerCrossAttentionLayer = headsPerLayer;
    }

    [[nodiscard]] SizeType32 getSumLocalKvHeads(
        SizeType32 pipelineParallelism = 1, SizeType32 pipelineParallelismRank = 0, bool isCrossAttention = false) const
    {
        auto [cbegin, cend]
            = getNumKvHeadsPerLayerLocalRange(pipelineParallelism, pipelineParallelismRank, isCrossAttention);
        auto const sumLocalHeads = std::reduce(cbegin, cend);
        return sumLocalHeads;
    }

    [[nodiscard]] bool constexpr skipCrossAttnBlocks() const noexcept
    {
        return mSkipCrossAttnBlocks;
    }

    void constexpr setSkipCrossAttnBlocks(bool skipCrossAttnBlocks) noexcept
    {
        mSkipCrossAttnBlocks = skipCrossAttnBlocks;
    }

private:
    SizeType32 mVocabSize;
    SizeType32 mNbLayers;
    SizeType32 mNbAttentionLayers;
    SizeType32 mNbRnnLayers;
    SizeType32 mNbHeads;
    SizeType32 mHiddenSize;
    SizeType32 mSizePerHead;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mUseMambaConv1dPlugin;
    bool mInputPacked;
    bool mPagedState;
    SizeType32 mTokensPerBlock;
    common::QuantMode mQuantMode;
    SizeType32 mMaxBatchSize;
    SizeType32 mMaxBeamWidth;
    SizeType32 mMaxInputLen;
    SizeType32 mMaxSequenceLen;
    std::optional<SizeType32> mMaxNumTokens;

    bool mComputeContextLogits;
    bool mComputeGenerationLogits;
    ModelVariant mModelVariant;

    SizeType32 mMaxPromptEmbeddingTableSize;
    bool mUseMrope;
    SizeType32 mMaxPositionEmbeddings;
    SizeType32 mRotaryEmbeddingDim;

    bool mContextFMHA;
    bool mPagedContextFMHA;
    bool mUseXQA;
    bool mPpReduceScatter;

    bool mUseLoraPlugin;
    std::vector<LoraModule> mLoraModules;
    SizeType32 mMlpHiddenSize;
    SizeType32 mMaxLoraRank;

    std::optional<RnnConfig> mRnnConfig;

    KVCacheType mKVCacheType = KVCacheType::kCONTINUOUS;

    SizeType32 mMaxEncoderLen{};
    SizeType32 mEncoderHiddenSize{};
    bool mUseCrossAttention;
    bool mUsePositionEmbedding;
    bool mUseTokenTypeEmbedding;

    std::vector<LayerType> mLayerTypes;
    std::shared_ptr<SpeculativeDecodingModule> mSpeculativeDecodingModule;
    SpeculativeDecodingMode mSpeculativeDecodingMode;

    nvinfer1::DataType mLogitsDtype;
    bool mUseShapeInference;
    ManageWeightsType mManageWeightsType;
    std::string mModelName;
    std::vector<SizeType32> mNumKvHeadsPerAttentionLayer;
    std::vector<SizeType32> mNumKvHeadsPerCrossAttentionLayer;
    bool mSkipCrossAttnBlocks;
};

}
