
#include "gptJsonConfig.h"

#include "common.h"
#include "modelConfig.h"
#include "../common/assert.h"
#include "../common/logger.h"
#include "eagleModule.h"
#include "explicitDraftTokensModule.h"
#include "jsonSerialization.h"
#include "lookaheadModule.h"
#include "medusaModule.h"
#include "modelConfig.h"
#include "runtimeDefaults.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>
#include <utility>

using namespace suggestify::runtime;
namespace tc = suggestify::common;

namespace
{
using Json = typename nlohmann::json::basic_json;

template <typename FieldType>
FieldType parseJsonFieldOr(Json const& json, std::string_view name, FieldType defaultValue)
{
    auto value = defaultValue;
    try
    {
        if (json.find(name) != json.end() && !json.at(name).is_null())
        {
            value = json.at(name).template get<FieldType>();
        }
    }
    catch (nlohmann::json::out_of_range& e)
    {
        LOG_DEBUG("Parameter %s cannot be read from json:", std::string(name).c_str());
        LOG_DEBUG(e.what());
    }
    return value;
}

template <typename FieldType>
std::optional<FieldType> parseJsonFieldOptional(Json const& json, std::string_view name)
{
    std::optional<FieldType> value = std::nullopt;
    try
    {
        value = json.at(name).template get<FieldType>();
    }
    catch (nlohmann::json::out_of_range const& e)
    {
        LOG_DEBUG(e.what());
        LOG_DEBUG("Optional value for parameter %s will not be set.", std::string(name).c_str());
    }
    catch (nlohmann::json::type_error const& e)
    {
        LOG_DEBUG(e.what());
        LOG_DEBUG("Optional value for parameter %s will not be set.", std::string(name).c_str());
    }
    return value;
}

std::vector<ModelConfig::LayerType> buildLayerTypes(
    std::size_t const numLayers, std::vector<std::string> const& layerStringTypes)
{
    std::vector<ModelConfig::LayerType> result{numLayers, ModelConfig::LayerType::kATTENTION};
    if (layerStringTypes.empty())
    {
        return result;
    }

    auto constexpr layerNameAttention = "attention";
    auto constexpr layerNameRecurrent = "recurrent";
    auto constexpr layerNameLinear = "linear";
    auto constexpr layerNameNoop = "no_op";

    auto const groupSize = layerStringTypes.size();
    for (std::size_t i = 0; i < numLayers; ++i)
    {
        if (layerStringTypes[i % groupSize] == layerNameAttention)
        {
            result[i] = ModelConfig::LayerType::kATTENTION;
        }
        else if (layerStringTypes[i % groupSize] == layerNameRecurrent)
        {
            result[i] = ModelConfig::LayerType::kRECURRENT;
        }
        else if (layerStringTypes[i % groupSize] == layerNameLinear)
        {
            result[i] = ModelConfig::LayerType::kLINEAR;
        }
        else if (layerStringTypes[i % groupSize] == layerNameNoop)
        {
            result[i] = ModelConfig::LayerType::kNOOP;
        }
        else
        {
            LOG_WARNING("Unknown layer type: %s, assuming attention", layerStringTypes[i % groupSize].c_str());
        }
    }

    return result;
}

ModelConfig createModelConfig(
    Json const& json, bool engineVersionNone, SizeType32 tensorParallelism, nvinfer1::DataType dataType)
{
    auto const& config = engineVersionNone ? json.at("builder_config") : json.at("pretrained_config");

    auto const* const archField = "architecture";
    auto const* const numLayersField = engineVersionNone ? "num_layers" : "num_hidden_layers";
    auto const* const numHeadsField = engineVersionNone ? "num_heads" : "num_attention_heads";
    auto const* const numKvHeadsField = engineVersionNone ? "num_kv_heads" : "num_key_value_heads";
    auto const* const mlpHiddenSizeField = engineVersionNone ? "mlp_hidden_size" : "intermediate_size";

    auto const arch = engineVersionNone ? std::string("none") : config.at(archField).template get<std::string>();
    auto numLayers = config.at(numLayersField).template get<SizeType32>();

    if (!engineVersionNone)
    {
        auto const speculativeDecodingModeOpt = parseJsonFieldOptional<SpeculativeDecodingMode::UnderlyingType>(
            json.at("build_config"), "speculative_decoding_mode");

        if (speculativeDecodingModeOpt.has_value()
            && SpeculativeDecodingMode(speculativeDecodingModeOpt.value()).isEagle())
        {
            auto const& eagleConfig = json.at("pretrained_config").at("eagle_net_config");
            auto const numEagleNetLayers = eagleConfig.at("num_hidden_layers").template get<SizeType32>();

            numLayers += numEagleNetLayers;
        }
    }

    auto const numHeads = config.at(numHeadsField).template get<SizeType32>() / tensorParallelism;
    auto const layerStringTypes
        = parseJsonFieldOr<std::vector<std::string>>(config, "layer_types", std::vector<std::string>());
    auto const layerTypes = buildLayerTypes(numLayers, layerStringTypes);
    auto const numAttentionLayers
        = static_cast<SizeType32>(std::count(layerTypes.begin(), layerTypes.end(), ModelConfig::LayerType::kATTENTION));
    auto const numRnnLayers
        = static_cast<SizeType32>(std::count(layerTypes.begin(), layerTypes.end(), ModelConfig::LayerType::kRECURRENT));

    auto const vocabSize = config.at("vocab_size").template get<SizeType32>();
    auto const hiddenSize = config.at("hidden_size").template get<SizeType32>() / tensorParallelism;
    auto const sizePerHead = parseJsonFieldOr(config, "head_size", hiddenSize / numHeads);

    auto const logitsDtypeStr = parseJsonFieldOr(config, "logits_dtype", std::string("float32"));

    auto const numKvHeads
        = std::max(parseJsonFieldOr(config, numKvHeadsField, numHeads * tensorParallelism) / tensorParallelism, 1);

    auto const mlpHiddenSize = parseJsonFieldOptional<SizeType32>(config, mlpHiddenSizeField);

    auto numKvHeadsPerAttentionLayer
        = parseJsonFieldOr<std::vector<SizeType32>>(config, "num_kv_heads_per_layer", std::vector<SizeType32>());

    auto numKvHeadsPerCrossAttentionLayer = parseJsonFieldOr<std::vector<SizeType32>>(
        config, "num_kv_heads_per_cross_attn_layer", std::vector<SizeType32>());
    auto modelConfig
        = ModelConfig{vocabSize, numLayers, numAttentionLayers, numRnnLayers, numHeads, hiddenSize, dataType};

    if (!numKvHeadsPerAttentionLayer.empty())
    {
        std::transform(numKvHeadsPerAttentionLayer.cbegin(), numKvHeadsPerAttentionLayer.cend(),
            numKvHeadsPerAttentionLayer.begin(),
            [tensorParallelism](SizeType32 const numKvHeads)
            { return ((numKvHeads + tensorParallelism - 1) / tensorParallelism); });
        modelConfig.setNumKvHeadsPerLayer(numKvHeadsPerAttentionLayer);
    }
    else
    {
        modelConfig.setNbKvHeads(numKvHeads);
    }

    if (!numKvHeadsPerCrossAttentionLayer.empty())
    {
        std::transform(numKvHeadsPerCrossAttentionLayer.cbegin(), numKvHeadsPerCrossAttentionLayer.cend(),
            numKvHeadsPerCrossAttentionLayer.begin(),
            [tensorParallelism](SizeType32 const numKvHeads)
            { return ((numKvHeads + tensorParallelism - 1) / tensorParallelism); });
        modelConfig.setNumKvHeadsPerCrossLayer(numKvHeadsPerCrossAttentionLayer);
    }
    else
    {
        modelConfig.setNbCrossKvHeads(numKvHeads);
    }

    modelConfig.setSizePerHead(sizePerHead);
    modelConfig.setLayerTypes(layerTypes);

    auto logitsDtype = nvinfer1::DataType::kFLOAT;
    if (logitsDtypeStr == "float32")
    {
        logitsDtype = nvinfer1::DataType::kFLOAT;
    }
    else if (logitsDtypeStr == "float16")
    {
        logitsDtype = nvinfer1::DataType::kHALF;
    }
    else
    {
        THROW("Unsupported logits data type");
    }
    modelConfig.setLogitsDtype(logitsDtype);

    auto const useCrossAttention
        = arch == std::string("DecoderModel") || parseJsonFieldOr(config, "cross_attention", false);
    if (useCrossAttention)
    {
        LOG_INFO("Setting encoder max input length and hidden size for accepting visual features.");
        auto const maxEncoderLen = parseJsonFieldOr<SizeType32>(json.at("build_config"), "max_encoder_input_len", 0);
        modelConfig.setMaxEncoderLen(maxEncoderLen);
        modelConfig.setEncoderHiddenSize(hiddenSize * tensorParallelism);
    }

    auto const usePositionEmbedding = parseJsonFieldOr<bool>(config, "has_position_embedding", false);
    auto const useTokenTypeEmbedding = parseJsonFieldOr<bool>(config, "has_token_type_embedding", false);
    auto const skipCrossAttnBlocks
        = useCrossAttention && parseJsonFieldOr<bool>(config, "skip_cross_attn_blocks", false);
    modelConfig.setUseCrossAttention(useCrossAttention);
    modelConfig.setUsePositionEmbedding(usePositionEmbedding);
    modelConfig.setUseTokenTypeEmbedding(useTokenTypeEmbedding);
    if (json.count("pretrained_config"))
    {
        auto const maxPositionEmbeddings
            = parseJsonFieldOr<SizeType32>(json.at("pretrained_config"), "max_position_embeddings", 0);
        modelConfig.setMaxPositionEmbeddings(maxPositionEmbeddings);
        auto const rotaryEmbeddingDim
            = parseJsonFieldOr<SizeType32>(json.at("pretrained_config"), "rotary_embedding_dim", 0);
        modelConfig.setRotaryEmbeddingDim(rotaryEmbeddingDim);
    }
    modelConfig.setSkipCrossAttnBlocks(skipCrossAttnBlocks);

    if (mlpHiddenSize.has_value())
    {
        modelConfig.setMlpHiddenSize(mlpHiddenSize.value() / tensorParallelism);
    }

    return modelConfig;
};

void parseBuilderConfig(ModelConfig& modelConfig, Json const& builderConfig)
{
    auto const maxBatchSize = parseJsonFieldOr(builderConfig, "max_batch_size", 0);
    auto const maxBeamWidth = parseJsonFieldOr(builderConfig, "max_beam_width", 0);
    auto const maxInputLen = parseJsonFieldOr(builderConfig, "max_input_len", 0);
    auto const maxSequenceLen = parseJsonFieldOr(builderConfig, "max_seq_len", 0);
    auto const maxNumTokens = parseJsonFieldOptional<SizeType32>(builderConfig, "max_num_tokens");
    auto const maxPromptEmbeddingTableSize
        = parseJsonFieldOr<SizeType32>(builderConfig, "max_prompt_embedding_table_size", 0);
    auto const computeContextLogits = parseJsonFieldOr(builderConfig, "gather_context_logits", false);
    auto const computeGenerationLogits = parseJsonFieldOr(builderConfig, "gather_generation_logits", false);
    auto const speculativeDecodingModeOpt
        = parseJsonFieldOptional<SpeculativeDecodingMode::UnderlyingType>(builderConfig, "speculative_decoding_mode");
    auto const kvCacheTypeStr = parseJsonFieldOr<std::string>(builderConfig, "kv_cache_type", "continuous");
    auto const kvCacheType = ModelConfig::KVCacheTypeFromString(kvCacheTypeStr);
    auto const useMrope = parseJsonFieldOr(builderConfig, "use_mrope", false);

    auto it = builderConfig.find("kv_cache_type");
    if (it == builderConfig.end())
    {
        LOG_ERROR(
            "Missing kv_cache_type field in builder_config, you need to rebuild engine. Default to continuous kv "
            "cache.");
    }

    modelConfig.setMaxBatchSize(maxBatchSize);
    modelConfig.setMaxBeamWidth(maxBeamWidth);
    modelConfig.setMaxInputLen(maxInputLen);
    modelConfig.setMaxSequenceLen(maxSequenceLen);
    modelConfig.setMaxNumTokens(maxNumTokens);
    modelConfig.setMaxPromptEmbeddingTableSize(maxPromptEmbeddingTableSize);
    modelConfig.computeContextLogits(computeContextLogits);
    modelConfig.computeGenerationLogits(computeGenerationLogits);
    modelConfig.setSpeculativeDecodingMode(speculativeDecodingModeOpt.has_value()
            ? SpeculativeDecodingMode(speculativeDecodingModeOpt.value())
            : SpeculativeDecodingMode::None());
    modelConfig.setKVCacheType(kvCacheType);
    modelConfig.setUseMrope(useMrope);
}

void parsePluginConfig(ModelConfig& modelConfig, Json const& pluginConfig)
{
    auto const useGptAttentionPlugin = !pluginConfig.at("gpt_attention_plugin").is_null();
    auto const useMambaConv1dPlugin
        = pluginConfig.contains("mamba_conv1d_plugin") && !pluginConfig.at("mamba_conv1d_plugin").is_null();
    auto const removeInputPadding = pluginConfig.at("remove_input_padding").template get<bool>();
    auto const& pagedKvCache = pluginConfig.at("paged_kv_cache");
    auto const& tokensPerBlock = pluginConfig.at("tokens_per_block");
    auto const contextFMHA = pluginConfig.at("context_fmha").template get<bool>();
    auto const pagedContextFMHA = pluginConfig.at("use_paged_context_fmha").template get<bool>();
    auto const pagedState = parseJsonFieldOr(pluginConfig, "paged_state", false);
    auto const manageWeightsType = parseJsonFieldOr<bool>(pluginConfig, "manage_weights", false)
        ? ModelConfig::ManageWeightsType::kEnabled
        : ModelConfig::ManageWeightsType::kDisabled;
    auto const ppReduceScatter = parseJsonFieldOr<bool>(pluginConfig, "pp_reduce_scatter", false);

    CHECK_WITH_INFO(
        !removeInputPadding || modelConfig.getMaxNumTokens(), "Padding removal requires max_num_tokens to be set.");

    modelConfig.useGptAttentionPlugin(useGptAttentionPlugin);
    modelConfig.useMambaConv1dPlugin(useMambaConv1dPlugin);
    modelConfig.usePackedInput(removeInputPadding);
    modelConfig.usePagedState(pagedState);
    if (pagedKvCache)
    {
        modelConfig.setKVCacheType(ModelConfig::KVCacheType::kPAGED);
    }
    modelConfig.setTokensPerBlock(tokensPerBlock);
    modelConfig.setContextFMHA(contextFMHA);
    modelConfig.setPagedContextFMHA(pagedContextFMHA);
    modelConfig.setManageWeightsType(manageWeightsType);
    modelConfig.setPpReduceScatter(ppReduceScatter);
}

void parseLora(ModelConfig& modelConfig, Json const& json, Json const& pluginConfig, bool engineVersionNone,
    SizeType32 tensorParallelism)
{
    auto const& config = engineVersionNone ? json.at("builder_config") : json.at("build_config").at("lora_config");

    auto const loraMaxRank = parseJsonFieldOr(config, "max_lora_rank", SizeType32{0});
    auto const loraTargetModules = parseJsonFieldOptional<std::vector<std::string>>(config, "lora_target_modules");

    if (loraTargetModules.has_value())
    {
        auto const& loraModuleNames = loraTargetModules.value();
        auto const& numKvHeadsPerLayer = modelConfig.getNumKvHeadsPerLayer();
        if (!loraModuleNames.empty())
        {
            CHECK_WITH_INFO(std::all_of(numKvHeadsPerLayer.cbegin(), numKvHeadsPerLayer.cend(),
                                     [firstNumKvHeads = numKvHeadsPerLayer[0]](SizeType32 numKvHeads)
                                     { return numKvHeads == firstNumKvHeads; }),
                "LORA with a VGQA model is not supported");
        }
        auto const numKvHeads = numKvHeadsPerLayer.empty() ? modelConfig.getNbHeads() : numKvHeadsPerLayer[0];
        bool hasMoE = !engineVersionNone && json.at("pretrained_config").contains("moe");
        auto const numExperts = hasMoE
            ? json.at("pretrained_config").at("moe").at("num_experts").template get<SizeType32>()
            : SizeType32{0};
        modelConfig.setLoraModules(LoraModule::createLoraModules(loraTargetModules.value(), modelConfig.getHiddenSize(),
            modelConfig.getMlpHiddenSize(), modelConfig.getNbHeads(), numKvHeads, modelConfig.getSizePerHead(),
            tensorParallelism, numExperts));
    }

    modelConfig.setMaxLoraRank(loraMaxRank);

    auto useLoraPlugin = !pluginConfig.at("lora_plugin").is_null();
    if (useLoraPlugin)
    {

        if (modelConfig.getLoraModules().empty() || modelConfig.getMaxLoraRank() == 0)
        {
            LOG_WARNING("lora_plugin enabled, but no lora module enabled: setting useLoraPlugin to false");
            useLoraPlugin = false;
        }
    }
    modelConfig.useLoraPlugin(useLoraPlugin);
}

template <typename InputType>
GptJsonConfig parseJson(InputType&& input)
{
    auto constexpr allowExceptions = true;
    auto constexpr ignoreComments = true;
    auto const json = nlohmann::json::parse(std::forward<InputType>(input), nullptr, allowExceptions, ignoreComments);

    auto const engineVersion = parseJsonFieldOr(json, "version", std::string("none"));

    auto const engineVersionNone = engineVersion == std::string("none");
    if (engineVersionNone)
    {
        LOG_INFO("No engine version found in the config file, assuming engine(s) built by old builder API.");
    }
    else
    {
        LOG_INFO("Engine version %s found in the config file, assuming engine(s) built by new builder API.",
            engineVersion.c_str());
    }

    auto const& builderConfig = engineVersionNone ? json.at("builder_config") : json.at("build_config");

    auto const name = engineVersionNone ? builderConfig.at("name").template get<std::string>()
                                        : json.at("pretrained_config").at("architecture").template get<std::string>();

    auto const tensorParallelism = engineVersionNone
        ? builderConfig.at("tensor_parallel").template get<SizeType32>()
        : json.at("pretrained_config").at("mapping").at("tp_size").template get<SizeType32>();
    auto const pipelineParallelism = engineVersionNone
        ? parseJsonFieldOr(builderConfig, "pipeline_parallel", 1)
        : parseJsonFieldOr(json.at("pretrained_config").at("mapping"), "pp_size", 1);
    auto const contextParallelism = engineVersionNone
        ? parseJsonFieldOr(builderConfig, "context_parallel", 1)
        : parseJsonFieldOr(json.at("pretrained_config").at("mapping"), "cp_size", 1);
    auto const gpusPerNode = engineVersionNone ? WorldConfig::kDefaultGpusPerNode
                                               : parseJsonFieldOr(json.at("pretrained_config").at("mapping"),
                                                   "gpus_per_node", WorldConfig::kDefaultGpusPerNode);

    auto const precision = engineVersionNone ? builderConfig.at("precision").template get<std::string>()
                                             : json.at("pretrained_config").at("dtype").template get<std::string>();

    auto const dataType = [&precision]()
    {
        if (precision == "float32")
        {
            return nvinfer1::DataType::kFLOAT;
        }
        if (precision == "float16")
        {
            return nvinfer1::DataType::kHALF;
        }
        if (precision == "bfloat16")
        {
            return nvinfer1::DataType::kBF16;
        }
        THROW("Model data type '%s' not supported", precision.c_str());
    }();

    auto modelConfig = createModelConfig(json, engineVersionNone, tensorParallelism, dataType);
    modelConfig.setModelName(name);

    parseBuilderConfig(modelConfig, builderConfig);

    auto const& pluginConfig = engineVersionNone ? json.at("plugin_config") : builderConfig.at("plugin_config");
    parsePluginConfig(modelConfig, pluginConfig);

    parseLora(modelConfig, json, pluginConfig, engineVersionNone, tensorParallelism);

    auto runtimeDefaults = engineVersionNone
        ? std::nullopt
        : parseJsonFieldOptional<RuntimeDefaults>(json.at("pretrained_config"), "runtime_defaults");

    if (engineVersionNone)
    {
        auto const quantMode
            = tc::QuantMode(parseJsonFieldOr(builderConfig, "quant_mode", tc::QuantMode::none().value()));
        modelConfig.setQuantMode(quantMode);
    }
    else
    {
        auto const& quantization = json.at("pretrained_config").at("quantization");
        auto quantAlgo = parseJsonFieldOptional<std::string>(quantization, "quant_algo");
        auto kvCacheQuantAlgo = parseJsonFieldOptional<std::string>(quantization, "kv_cache_quant_algo");
        auto const quantMode = tc::QuantMode::fromQuantAlgo(quantAlgo, kvCacheQuantAlgo);
        modelConfig.setQuantMode(quantMode);
    }

    if (engineVersionNone)
    {
        if (name == std::string("chatglm_6b"))
        {
            modelConfig.setModelVariant(ModelConfig::ModelVariant::kChatGlm);
        }
        if (name == std::string("glm_10b"))
        {
            modelConfig.setModelVariant(ModelConfig::ModelVariant::kGlm);
        }
    }
    else
    {
        if (name.find("GLM") != std::string::npos)
        {
            auto const& pretrainedConfig = json.at("pretrained_config");
            auto const chatglmVersion = pretrainedConfig.at("chatglm_version").template get<std::string>();
            if (chatglmVersion == "chatglm")
            {
                modelConfig.setModelVariant(ModelConfig::ModelVariant::kChatGlm);
            }
            if (chatglmVersion == "glm")
            {
                modelConfig.setModelVariant(ModelConfig::ModelVariant::kGlm);
            }
        }
    }

    if (!engineVersionNone)
    {
        if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
        {
            auto const& pretrainedConfig = json.at("pretrained_config");

            auto const maxNumPaths = parseJsonFieldOr(pretrainedConfig, "redrafter_num_beams", 0);
            auto const maxDraftPathLen = parseJsonFieldOr(pretrainedConfig, "redrafter_draft_len_per_beam", 0);
            auto const maxDraftLen = maxNumPaths * maxDraftPathLen;

            auto explicitDraftTokensModule
                = std::make_shared<ExplicitDraftTokensModule>(maxDraftPathLen, maxDraftLen, maxNumPaths);
            modelConfig.setSpeculativeDecodingModule(explicitDraftTokensModule);
            modelConfig.setUseShapeInference(false);
        }
        else if (modelConfig.getSpeculativeDecodingMode().isMedusa())
        {
            auto const& pretrainedConfig = json.at("pretrained_config");
            auto const maxDraftLen = parseJsonFieldOr(pretrainedConfig, "max_draft_len", 0);
            auto const medusaHeads = parseJsonFieldOptional<SizeType32>(pretrainedConfig, "num_medusa_heads");
            CHECK_WITH_INFO(medusaHeads.has_value() && maxDraftLen > 0,
                "Both num_medusa_heads and max_draft_len have to be provided for Medusa model");

            auto medusaModule = std::make_shared<MedusaModule>(medusaHeads.value(), maxDraftLen);
            modelConfig.setSpeculativeDecodingModule(medusaModule);
        }
        else
        {
            auto const maxDraftLen = parseJsonFieldOr(builderConfig, "max_draft_len", 0);
            if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
            {
                CHECK_WITH_INFO(
                    maxDraftLen > 0, "max_draft_len has to be larger than 0 for Lookahead decoding model");
                auto lookaheadDecodingModule = std::make_shared<LookaheadModule>(maxDraftLen, maxDraftLen);
                modelConfig.setSpeculativeDecodingModule(lookaheadDecodingModule);
            }
            else if (modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
            {
                CHECK_WITH_INFO(
                    maxDraftLen > 0, "max_draft_len has to be larger than 0 for decoding with external draft tokens");
                auto speculativeDecodingModule
                    = std::make_shared<SpeculativeDecodingModule>(maxDraftLen, maxDraftLen, 1);
                modelConfig.setSpeculativeDecodingModule(speculativeDecodingModule);
            }
            else if (modelConfig.getSpeculativeDecodingMode().isEagle())
            {
                auto const& pretrainedConfig = json.at("pretrained_config");

                auto const numEagleLayers = parseJsonFieldOr(pretrainedConfig, "num_eagle_layers", 0);
                auto const& eagleConfig = pretrainedConfig.at("eagle_net_config");
                auto const numEagleNetLayers = eagleConfig.at("num_hidden_layers").template get<SizeType32>();
                auto const maxNonLeafNodesPerLayer
                    = pretrainedConfig.at("max_non_leaves_per_layer").template get<SizeType32>();

                CHECK_WITH_INFO(maxDraftLen > 0, "max_draft_len has to be larger than 0 for eagle decoding");
                CHECK_WITH_INFO(numEagleLayers > 0, "num_eagle_layers has to be larger than 0 for eagle decoding");
                CHECK_WITH_INFO(
                    maxNonLeafNodesPerLayer > 0, "max_non_leaves_per_layer has to be larger than 0 for eagle decoding");
                auto eagleModule = std::make_shared<EagleModule>(
                    numEagleLayers, maxDraftLen, numEagleNetLayers, maxNonLeafNodesPerLayer);
                modelConfig.setSpeculativeDecodingModule(eagleModule);
            }
        }
    }

    if (!engineVersionNone)
    {
        auto const& pretrainedConfig = json.at("pretrained_config");
        auto const architecture = pretrainedConfig.at("architecture").template get<std::string>();
        if (architecture == std::string("MambaForCausalLM"))
        {
            modelConfig.setModelVariant(ModelConfig::ModelVariant::kMamba);
        }
        else if (architecture == std::string("RecurrentGemmaForCausalLM"))
        {
            modelConfig.setModelVariant(ModelConfig::ModelVariant::kRecurrentGemma);
        }
        if (modelConfig.isRnnBased())
        {
            auto const& stateSize = pretrainedConfig.at("state_size").template get<SizeType32>();
            auto const& convKernel = pretrainedConfig.at("conv_kernel").template get<SizeType32>();
            auto const& rnnHiddenSize = pretrainedConfig.at("rnn_hidden_size").template get<SizeType32>();
            auto const& rnnConvDimSize = pretrainedConfig.at("rnn_conv_dim_size").template get<SizeType32>();
            ModelConfig::RnnConfig rnnConfig{};
            rnnConfig.stateSize = stateSize;
            rnnConfig.convKernel = convKernel;
            rnnConfig.rnnHiddenSize = rnnHiddenSize;
            rnnConfig.rnnConvDimSize = rnnConvDimSize;
            if (pretrainedConfig.contains("rnn_head_size"))
            {
                auto const& rnnHeadSize = pretrainedConfig.at("rnn_head_size").template get<SizeType32>();
                rnnConfig.rnnHeadSize = rnnHeadSize;
            }
            modelConfig.setRnnConfig(rnnConfig);
        }
    }
    else
    {
        if (name.size() >= 6 && name.substr(0, 6) == "mamba_")
        {
            modelConfig.setModelVariant(ModelConfig::ModelVariant::kMamba);
        }
        else if (name.size() >= 15 && name.substr(0, 15) == "recurrentgemma_")
        {
            modelConfig.setModelVariant(ModelConfig::ModelVariant::kRecurrentGemma);
        }
        if (modelConfig.isRnnBased())
        {
            auto const& stateSize = builderConfig.at("state_size").template get<SizeType32>();
            auto const& convKernel = builderConfig.at("conv_kernel").template get<SizeType32>();
            auto const& rnnHiddenSize = builderConfig.at("rnn_hidden_size").template get<SizeType32>();
            auto const& rnnConvDimSize = builderConfig.at("rnn_conv_dim_size").template get<SizeType32>();
            ModelConfig::RnnConfig rnnConfig{};
            rnnConfig.stateSize = stateSize;
            rnnConfig.convKernel = convKernel;
            rnnConfig.rnnHiddenSize = rnnHiddenSize;
            rnnConfig.rnnConvDimSize = rnnConvDimSize;
            if (builderConfig.contains("rnn_head_size"))
            {
                auto const& rnnHeadSize = builderConfig.at("rnn_head_size").template get<SizeType32>();
                rnnConfig.rnnHeadSize = rnnHeadSize;
            }
            modelConfig.setRnnConfig(rnnConfig);
        }
    }
    return GptJsonConfig{name, engineVersion, precision, tensorParallelism, pipelineParallelism, contextParallelism,
        gpusPerNode, modelConfig, runtimeDefaults};
}

}

std::string GptJsonConfig::engineFilename(WorldConfig const& worldConfig, std::string const& model) const
{
    CHECK_WITH_INFO(getTensorParallelism() == worldConfig.getTensorParallelism(), "tensor parallelism mismatch");
    CHECK_WITH_INFO(
        getPipelineParallelism() == worldConfig.getPipelineParallelism(), "pipeline parallelism mismatch");
    CHECK_WITH_INFO(
        getContextParallelism() == worldConfig.getContextParallelism(), "Context parallelism mismatch");
    auto pp = worldConfig.isPipelineParallel() ? "_pp" + std::to_string(worldConfig.getPipelineParallelism()) : "";
    auto cp = worldConfig.isContextParallel() ? "_cp" + std::to_string(worldConfig.getContextParallelism()) : "";
    if (getVersion() == std::string("none"))
    {
        return model + "_" + getPrecision() + "_tp" + std::to_string(worldConfig.getTensorParallelism()) + pp + cp
            + "_rank" + std::to_string(worldConfig.getRank()) + ".engine";
    }

    return "rank" + std::to_string(worldConfig.getRank()) + ".engine";
}

GptJsonConfig GptJsonConfig::parse(std::string const& json)
{
    return parseJson(json);
}

GptJsonConfig GptJsonConfig::parse(std::istream& json)
{
    return parseJson(json);
}

GptJsonConfig GptJsonConfig::parse(std::filesystem::path const& path)
{
    CHECK_WITH_INFO(std::filesystem::exists(path), std::string("File does not exist: ") + path.string());
    std::ifstream json(path);
    return parse(json);
}
