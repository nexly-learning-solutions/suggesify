#include "../plugins/api/tllmPlugin.h"

#include "../common/stringUtils.h"
#include "../runtime/tllmLogger.h"

#include "../plugins/bertAttentionPlugin/bertAttentionPlugin.h"
#include "../plugins/fp8RowwiseGemmPlugin/fp8RowwiseGemmPlugin.h"
#include "../plugins/gemmPlugin/gemmPlugin.h"
#include "../plugins/gemmSwigluPlugin/gemmSwigluPlugin.h"
#include "../plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#include "../plugins/identityPlugin/identityPlugin.h"
#include "../plugins/layernormQuantizationPlugin/layernormQuantizationPlugin.h"
#include "../plugins/lookupPlugin/lookupPlugin.h"
#include "../plugins/loraPlugin/loraPlugin.h"
#include "../plugins/lruPlugin/lruPlugin.h"
#include "../plugins/mambaConv1dPlugin/mambaConv1dPlugin.h"
#include "../plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#if ENABLE_MULTI_DEVICE
#include "../plugins/cpSplitPlugin/cpSplitPlugin.h"
#include "../plugins/ncclPlugin/allgatherPlugin.h"
#include "../plugins/ncclPlugin/allreducePlugin.h"
#include "../plugins/ncclPlugin/recvPlugin.h"
#include "../plugins/ncclPlugin/reduceScatterPlugin.h"
#include "../plugins/ncclPlugin/sendPlugin.h"
#endif
#include "../plugins/cudaStreamPlugin/cudaStreamPlugin.h"
#include "../plugins/cumsumLastDimPlugin/cumsumLastDimPlugin.h"
#include "../plugins/eaglePlugin/eagleDecodeDraftTokensPlugin.h"
#include "../plugins/eaglePlugin/eaglePrepareDrafterInputsPlugin.h"
#include "../plugins/eaglePlugin/eagleSampleAndAcceptDraftTokensPlugin.h"
#include "../plugins/lowLatencyGemmPlugin/lowLatencyGemmPlugin.h"
#include "../plugins/lowLatencyGemmSwigluPlugin/lowLatencyGemmSwigluPlugin.h"
#include "../plugins/qserveGemmPlugin/qserveGemmPlugin.h"
#include "../plugins/quantizePerTokenPlugin/quantizePerTokenPlugin.h"
#include "../plugins/quantizeTensorPlugin/quantizeTensorPlugin.h"
#include "../plugins/rmsnormQuantizationPlugin/rmsnormQuantizationPlugin.h"
#include "../plugins/selectiveScanPlugin/selectiveScanPlugin.h"
#include "../plugins/smoothQuantGemmPlugin/smoothQuantGemmPlugin.h"
#include "../plugins/topkLastDimPlugin/topkLastDimPlugin.h"
#include "../plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "../plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"

#include <array>
#include <cstdlib>

#include <NvInferRuntime.h>

namespace tc = suggestify::common;

namespace
{

nvinfer1::IPluginCreator* creatorPtr(nvinfer1::IPluginCreator& creator)
{
    return &creator;
}

nvinfer1::IPluginCreatorInterface* creatorInterfacePtr(nvinfer1::IPluginCreatorInterface& creator)
{
    return &creator;
}

auto tllmLogger = suggestify::runtime::TllmLogger();

nvinfer1::ILogger* gLogger{&tllmLogger};

class GlobalLoggerFinder : public nvinfer1::ILoggerFinder
{
public:
    nvinfer1::ILogger* findLogger() override
    {
        return gLogger;
    }
};

GlobalLoggerFinder gGlobalLoggerFinder{};

#if !defined(_MSC_VER)
[[maybe_unused]] __attribute__((constructor))
#endif
void initOnLoad()
{
    auto constexpr kLoadPlugins = "TRT_LLM_LOAD_PLUGINS";
    auto const loadPlugins = std::getenv(kLoadPlugins);
    if (loadPlugins && loadPlugins[0] == '1')
    {
        initTrtLlmPlugins(gLogger);
    }
}

bool pluginsInitialized = false;

}

namespace suggestify::plugins::api
{

LoggerManager& suggestify::plugins::api::LoggerManager::getInstance() noexcept
{
    static LoggerManager instance;
    return instance;
}

void LoggerManager::setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr)
    {
        mLoggerFinder = finder;
    }
}

[[maybe_unused]] nvinfer1::ILogger* LoggerManager::logger()
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr)
    {
        return mLoggerFinder->findLogger();
    }
    return nullptr;
}

nvinfer1::ILogger* LoggerManager::defaultLogger() noexcept
{
    return gLogger;
}
}


extern "C"
{
    bool initTrtLlmPlugins(void* logger, char const* libNamespace)
    {
        if (pluginsInitialized)
        {
            return true;
        }

        if (logger)
        {
            gLogger = static_cast<nvinfer1::ILogger*>(logger);
        }
        setLoggerFinder(&gGlobalLoggerFinder);

        auto registry = getPluginRegistry();

        {
            std::int32_t nbCreators;
            auto creators = getPluginCreators(nbCreators);

            for (std::int32_t i = 0; i < nbCreators; ++i)
            {
                auto const creator = creators[i];
                creator->setPluginNamespace(libNamespace);
                registry->registerCreator(*creator, libNamespace);
                if (gLogger)
                {
                    auto const msg = tc::fmtstr("Registered plugin creator %s version %s in namespace %s",
                        creator->getPluginName(), creator->getPluginVersion(), libNamespace);
                    gLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, msg.c_str());
                }
            }
        }

        {
            std::int32_t nbCreators;
            auto creators = getCreators(nbCreators);

            for (std::int32_t i = 0; i < nbCreators; ++i)
            {
                auto const creator = creators[i];
                registry->registerCreator(*creator, libNamespace);
            }
        }

        pluginsInitialized = true;
        return true;
    }

    [[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder* finder)
    {
        suggestify::plugins::api::LoggerManager::getInstance().setLoggerFinder(finder);
    }

    [[maybe_unused]] nvinfer1::IPluginCreator* const* getPluginCreators(std::int32_t& nbCreators)
    {
        static suggestify::plugins::IdentityPluginCreator identityPluginCreator;
        static suggestify::plugins::BertAttentionPluginCreator bertAttentionPluginCreator;
        static suggestify::plugins::GPTAttentionPluginCreator gptAttentionPluginCreator;
        static suggestify::plugins::GemmPluginCreator gemmPluginCreator;
        static suggestify::plugins::GemmSwigluPluginCreator gemmSwigluPluginCreator;
        static suggestify::plugins::Fp8RowwiseGemmPluginCreator fp8RowwiseGemmPluginCreator;
        static suggestify::plugins::MixtureOfExpertsPluginCreator moePluginCreator;
#if ENABLE_MULTI_DEVICE
        static suggestify::plugins::SendPluginCreator sendPluginCreator;
        static suggestify::plugins::RecvPluginCreator recvPluginCreator;
        static suggestify::plugins::AllreducePluginCreator allreducePluginCreator;
        static suggestify::plugins::AllgatherPluginCreator allgatherPluginCreator;
        static suggestify::plugins::ReduceScatterPluginCreator reduceScatterPluginCreator;
#endif
        static suggestify::plugins::SmoothQuantGemmPluginCreator smoothQuantGemmPluginCreator;
        static suggestify::plugins::QServeGemmPluginCreator qserveGemmPluginCreator;
        static suggestify::plugins::LayernormQuantizationPluginCreator layernormQuantizationPluginCreator;
        static suggestify::plugins::QuantizePerTokenPluginCreator quantizePerTokenPluginCreator;
        static suggestify::plugins::QuantizeTensorPluginCreator quantizeTensorPluginCreator;
        static suggestify::plugins::RmsnormQuantizationPluginCreator rmsnormQuantizationPluginCreator;
        static suggestify::plugins::WeightOnlyGroupwiseQuantMatmulPluginCreator
            weightOnlyGroupwiseQuantMatmulPluginCreator;
        static suggestify::plugins::WeightOnlyQuantMatmulPluginCreator weightOnlyQuantMatmulPluginCreator;
        static suggestify::plugins::LookupPluginCreator lookupPluginCreator;
        static suggestify::plugins::LoraPluginCreator loraPluginCreator;
        static suggestify::plugins::SelectiveScanPluginCreator selectiveScanPluginCreator;
        static suggestify::plugins::MambaConv1dPluginCreator mambaConv1DPluginCreator;
        static suggestify::plugins::lruPluginCreator lruPluginCreator;
        static suggestify::plugins::CumsumLastDimPluginCreator cumsumLastDimPluginCreator;
        static suggestify::plugins::TopkLastDimPluginCreator topkLastDimPluginCreator;
        static suggestify::plugins::LowLatencyGemmPluginCreator lowLatencyGemmPluginCreator;
        static suggestify::plugins::LowLatencyGemmSwigluPluginCreator lowLatencyGemmSwigluPluginCreator;
        static suggestify::plugins::EagleDecodeDraftTokensPluginCreator eagleDecodeDraftTokensPluginCreator;
        static suggestify::plugins::EagleSampleAndAcceptDraftTokensPluginCreator
            eagleSampleAndAcceptDraftTokensPluginCreator;
        static suggestify::plugins::CudaStreamPluginCreator cudaStreamPluginCreator;

        static std::array pluginCreators
            = { creatorPtr(identityPluginCreator),
                  creatorPtr(bertAttentionPluginCreator),
                  creatorPtr(gptAttentionPluginCreator),
                  creatorPtr(gemmPluginCreator),
                  creatorPtr(gemmSwigluPluginCreator),
                  creatorPtr(fp8RowwiseGemmPluginCreator),
                  creatorPtr(moePluginCreator),
#if ENABLE_MULTI_DEVICE
                  creatorPtr(sendPluginCreator),
                  creatorPtr(recvPluginCreator),
                  creatorPtr(allreducePluginCreator),
                  creatorPtr(allgatherPluginCreator),
                  creatorPtr(reduceScatterPluginCreator),
#endif
                  creatorPtr(smoothQuantGemmPluginCreator),
                  creatorPtr(qserveGemmPluginCreator),
                  creatorPtr(layernormQuantizationPluginCreator),
                  creatorPtr(quantizePerTokenPluginCreator),
                  creatorPtr(quantizeTensorPluginCreator),
                  creatorPtr(rmsnormQuantizationPluginCreator),
                  creatorPtr(weightOnlyGroupwiseQuantMatmulPluginCreator),
                  creatorPtr(weightOnlyQuantMatmulPluginCreator),
                  creatorPtr(lookupPluginCreator),
                  creatorPtr(loraPluginCreator),
                  creatorPtr(selectiveScanPluginCreator),
                  creatorPtr(mambaConv1DPluginCreator),
                  creatorPtr(lruPluginCreator),
                  creatorPtr(cumsumLastDimPluginCreator),
                  creatorPtr(topkLastDimPluginCreator),
                  creatorPtr(lowLatencyGemmPluginCreator),
                  creatorPtr(eagleDecodeDraftTokensPluginCreator),
                  creatorPtr(eagleSampleAndAcceptDraftTokensPluginCreator),
                  creatorPtr(lowLatencyGemmSwigluPluginCreator),
                  creatorPtr(cudaStreamPluginCreator),
              };
        nbCreators = pluginCreators.size();
        return pluginCreators.data();
    }

    [[maybe_unused]] nvinfer1::IPluginCreatorInterface* const* getCreators(std::int32_t& nbCreators)
    {
        static suggestify::plugins::EaglePrepareDrafterInputsPluginCreator eaglePrepareDrafterInputsPluginCreator;
#if ENABLE_MULTI_DEVICE
        static suggestify::plugins::CpSplitPluginCreator cpSplitPluginCreator;
#endif

        static std::array creators
            = { creatorInterfacePtr(eaglePrepareDrafterInputsPluginCreator),
#if ENABLE_MULTI_DEVICE
                  creatorInterfacePtr(cpSplitPluginCreator),
#endif
              };
        nbCreators = creators.size();
        return creators.data();
    }
}
