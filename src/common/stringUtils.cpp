
#include "stringUtils.h"
#include "assert.h"

#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <string>

namespace suggestify::common
{

namespace
{
std::string vformat(char const* fmt, va_list args)
{
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
        return "";

    std::string stringBuf(size, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    CHECK_WITH_INFO(size2 == size, std::string(std::strerror(errno)));

    return stringBuf;
}

}

std::string fmtstr(char const* format, ...)
{
    va_list args;
    va_start(args, format);
    std::string result = vformat(format, args);
    va_end(args);
    return result;
};

std::unordered_set<std::string> str2set(std::string const& input, char delimiter)
{
    std::unordered_set<std::string> values;
    if (!input.empty())
    {
        std::stringstream valStream(input);
        std::string val;
        while (std::getline(valStream, val, delimiter))
        {
            if (!val.empty())
            {
                values.insert(val);
            }
        }
    }
    return values;
};

}
