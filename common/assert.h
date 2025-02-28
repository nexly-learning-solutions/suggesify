
#pragma once

#include "stringUtils.h"
#include "tllmException.h"

#include <string>

namespace suggestify::common
{
[[noreturn]] inline void throwRuntimeError(char const* const file, int const line, std::string const& info = "")
{
    throw TllmException(file, line, fmtstr("[TensorRT-LLM][ERROR] Assertion failed: %s", info.c_str()));
}

}

class DebugConfig
{
public:
    static bool isCheckDebugEnabled();
};

#if defined(_WIN32)
#define TLLM_LIKELY(x) (__assume((x) == 1), (x))
#define TLLM_UNLIKELY(x) (__assume((x) == 0), (x))
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#define TLLM_UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define TLLM_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                               \
                                            : suggestify::common::throwRuntimeError(__FILE__, __LINE__, #val);       \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : suggestify::common::throwRuntimeError(                                                                     \
            __FILE__, __LINE__, suggestify::common::fmtstr(info, ##__VA_ARGS__));                                    \
    } while (0)

#define TLLM_CHECK_DEBUG(val)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TLLM_UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val)) ? ((void) 0)                                                           \
                                                : suggestify::common::throwRuntimeError(__FILE__, __LINE__, #val);   \
        }                                                                                                              \
    } while (0)

#define TLLM_CHECK_DEBUG_WITH_INFO(val, info, ...)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TLLM_UNLIKELY(DebugConfig::isCheckDebugEnabled()))                                                         \
        {                                                                                                              \
            TLLM_LIKELY(static_cast<bool>(val))                                                                        \
            ? ((void) 0)                                                                                               \
            : suggestify::common::throwRuntimeError(                                                                 \
                __FILE__, __LINE__, suggestify::common::fmtstr(info, ##__VA_ARGS__));                                \
        }                                                                                                              \
    } while (0)

#define TLLM_THROW(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        throw NEW_TLLM_EXCEPTION(__VA_ARGS__);                                                                         \
    } while (0)

#define TLLM_WRAP(ex)                                                                                                  \
    NEW_TLLM_EXCEPTION("%s: %s", suggestify::common::TllmException::demangle(typeid(ex).name()).c_str(), ex.what())
