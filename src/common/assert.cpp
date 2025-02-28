
#include "assert.h"

namespace
{

bool initCheckDebug()
{
    auto constexpr kDebugEnabled = "DEBUG_MODE";
    auto const debugEnabled = std::getenv(kDebugEnabled);
    return debugEnabled && debugEnabled[0] == '1';
}
}

bool DebugConfig::isCheckDebugEnabled()
{
    static bool const debugEnabled = initCheckDebug();
    return debugEnabled;
}
