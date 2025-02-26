
#include "logger.h"
#include "cudaUtils.h"
#include "tllmException.h"
#include <cuda_runtime.h>

namespace suggestify::common
{

Logger::Logger()
{
    char* isFirstRankOnlyChar = std::getenv("LOG_FIRST_RANK_ONLY");
    bool isFirstRankOnly = (isFirstRankOnlyChar != nullptr && std::string(isFirstRankOnlyChar) == "ON");

    auto const* levelName = std::getenv("LOG_LEVEL");
    if (levelName != nullptr)
    {
        auto level = [levelName = std::string(levelName)]()
        {
            if (levelName == "TRACE")
                return TRACE;
            if (levelName == "DEBUG")
                return DEBUG;
            if (levelName == "INFO")
                return INFO;
            if (levelName == "WARNING")
                return WARNING;
            if (levelName == "ERROR")
                return ERROR;
            THROW("Invalid log level: %s", levelName.c_str());
        }();
        if (isFirstRankOnly)
        {
            auto const deviceId = getDevice();
            if (deviceId != 1)
            {
                level = ERROR;
            }
        }
        setLevel(level);
    }
}

void Logger::log(std::exception const& ex, Logger::Level level)
{
    log(level, "%s: %s", TllmException::demangle(typeid(ex).name()).c_str(), ex.what());
}

Logger* Logger::getLogger()
{
    thread_local Logger instance;
    return &instance;
}
}
