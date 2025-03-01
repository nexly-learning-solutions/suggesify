
#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "assert.h"
#include "stringUtils.h"

namespace suggestify::common
{

class Logger
{

#ifdef _WIN32
#undef ERROR
#endif

public:
    enum Level
    {
        TRACE = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40
    };

    static Logger* getLogger();

    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

#if defined(_MSC_VER)
    template <typename... Args>
    void log(Level level, char const* format, Args const&... args);

    template <typename... Args>
    void log(Level level, int rank, char const* format, Args const&... args);
#else
    template <typename... Args>
    void log(Level level, char const* format, Args const&... args) __attribute__((format(printf, 3, 0)));

    template <typename... Args>
    void log(Level level, int rank, char const* format, Args const&... args) __attribute__((format(printf, 4, 0)));
#endif

    template <typename... Args>
    void log(Level level, std::string const& format, Args const&... args)
    {
        return log(level, format.c_str(), args...);
    }

    template <typename... Args>
    void log(Level const level, int const rank, std::string const& format, Args const&... args)
    {
        return log(level, rank, format.c_str(), args...);
    }

    void log(std::exception const& ex, Level level = Level::ERROR);

    Level getLevel() const
    {
        return level_;
    }

    void setLevel(Level const level)
    {
        level_ = level;
        log(INFO, "Set logger level to %s", getLevelName(level));
    }

    bool isEnabled(Level const level) const
    {
        return level_ <= level;
    }

private:
    static auto constexpr kPREFIX = "[nexly]";

#ifndef NDEBUG
    Level const DEFAULT_LOG_LEVEL = DEBUG;
#else
    Level const DEFAULT_LOG_LEVEL = INFO;
#endif
    Level level_ = DEFAULT_LOG_LEVEL;

    Logger();

    static inline char const* getLevelName(Level const level)
    {
        switch (level)
        {
        case TRACE: return "TRACE";
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        }

        THROW("Unknown log level: %d", level);
    }

    static inline std::string getPrefix(Level const level)
    {
        return fmtstr("%s[%s] ", kPREFIX, getLevelName(level));
    }

    static inline std::string getPrefix(Level const level, int const rank)
    {
        return fmtstr("%s[%s][%d] ", kPREFIX, getLevelName(level), rank);
    }
};

template <typename... Args>
void Logger::log(Logger::Level level, char const* format, Args const&... args)
{
    if (isEnabled(level))
    {
        auto const fmt = getPrefix(level) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}

template <typename... Args>
void Logger::log(Logger::Level const level, int const rank, char const* format, Args const&... args)
{
    if (isEnabled(level))
    {
        auto const fmt = getPrefix(level, rank) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}

#define LOG(level, ...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        auto* const logger = suggestify::common::Logger::getLogger();                                                \
        if (logger->isEnabled(level))                                                                                  \
        {                                                                                                              \
            logger->log(level, __VA_ARGS__);                                                                           \
        }                                                                                                              \
    } while (0)

#define LOG_TRACE(...) LOG(suggestify::common::Logger::TRACE, __VA_ARGS__)
#define LOG_DEBUG(...) LOG(suggestify::common::Logger::DEBUG, __VA_ARGS__)
#define LOG_INFO(...) LOG(suggestify::common::Logger::INFO, __VA_ARGS__)
#define LOG_WARNING(...) LOG(suggestify::common::Logger::WARNING, __VA_ARGS__)
#define LOG_ERROR(...) LOG(suggestify::common::Logger::ERROR, __VA_ARGS__)
#define LOG_EXCEPTION(ex, ...) suggestify::common::Logger::getLogger()->log(ex, ##__VA_ARGS__)
}
