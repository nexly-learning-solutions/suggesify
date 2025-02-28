
#include "checkMacrosPlugin.h"

#include "../common/logger.h"

namespace suggestify::plugins
{

void caughtError(std::exception const& e)
{
    LOG_EXCEPTION(e);
}

void logError(char const* msg, char const* file, char const* fn, int line)
{
    LOG_ERROR("Parameter check failed at: %s::%s::%d, condition: %s", file, fn, line, msg);
}

}
