
#pragma once

#include <cstdint>
#include <mutex>

namespace nvinfer1
{
class ILoggerFinder;
class ILogger;

namespace v_1_0
{
class IPluginCreator;
class IPluginCreatorV3One;
class IPluginCreatorInterface;
}

}

namespace suggestify::plugins::api
{

auto constexpr kDefaultNamespace = "suggestify";

class LoggerManager
{
public:
    void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

    [[maybe_unused]] nvinfer1::ILogger* logger();

    static LoggerManager& getInstance() noexcept;

    static nvinfer1::ILogger* defaultLogger() noexcept;

private:
    LoggerManager() = default;

    nvinfer1::ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex mMutex;
};
}

extern "C"
{
    bool initTrtLlmPlugins(void* logger = suggestify::plugins::api::LoggerManager::defaultLogger(),
        char const* libNamespace = suggestify::plugins::api::kDefaultNamespace);

    [[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder* finder);
    [[maybe_unused]] nvinfer1::v_1_0::IPluginCreator* const* getPluginCreators(std::int32_t& nbCreators);
    [[maybe_unused]] nvinfer1::v_1_0::IPluginCreatorInterface* const* getCreators(std::int32_t& nbCreators);
}
