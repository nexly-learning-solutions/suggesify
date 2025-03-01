
#pragma once

#include "stringUtils.h"
#include "tllmException.h"

#include <string>

namespace suggestify::common
{
[[noreturn]] inline void throwRuntimeError(char const* const file, int const line, std::string const& info = "")
{
    throw TllmException(file, line, fmtstr("[nexly][ERROR] Assertion failed: %s", info.c_str()));
}

}

class DebugConfig
{
public:
    static bool isCheckDebugEnabled();
};

#if defined(_WIN32)
#define LIKELY(x) (__assume((x) == 1), (x))
#define UNLIKELY(x) (__assume((x) == 0), (x))
#else
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                               \
                                            : suggestify::common::throwRuntimeError(__FILE__, __LINE__, #val);       \
    } while (0)

#define CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : suggestify::common::throwRuntimeError(                                                                     \
            __FILE__, __LINE__, suggestify::common::fmtstr(info, ##__VA_ARGS__));                                    \
    } while (0)

#define CHECK_DEBUG(val)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : suggestify::common::throwRuntimeError(__FILE__, __LINE__, #val);   \
        }                                                                                                              \
    } while (0)

#define CHECK_DEBUG_WITH_INFO(val, info, ...)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            LIKELY(static_cast<bool>(val))                                                                        \
            ? ((void) 0)                                                                                               \
            : suggestify::common::throwRuntimeError(                                                                 \
                __FILE__, __LINE__, suggestify::common::fmtstr(info, ##__VA_ARGS__));                                \
        }                                                                                                              \
    } while (0)

#define THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define WRAP(ex)                                                                                                  \
    NEW_EXCEPTION("%s: %s", suggestify::common::TllmException::demangle(typeid(ex).name()).c_str(), ex.what())
