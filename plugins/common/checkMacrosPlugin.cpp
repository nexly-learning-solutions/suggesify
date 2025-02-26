
#include "checkMacrosPlugin.h"

#include "suggestify/common/logger.h"

namespace suggestify::plugins
{

void caughtError(std::exception const& e)
{
    TLLM_LOG_EXCEPTION(e);
}

void logError(char const* msg, char const* file, char const* fn, int line)
{
    TLLM_LOG_ERROR("Parameter check failed at: %s::%s::%d, condition: %s", file, fn, line, msg);
}

}
