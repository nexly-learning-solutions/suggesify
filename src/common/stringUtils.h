
#pragma once

#if ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace suggestify::common
{
#if ENABLE_BF16
static inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& stream, __nv_bfloat16 const& val)
{
    stream << __bfloat162float(val);
    return stream;
}
#endif

static inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& stream, __half const& val)
{
    stream << __half2float(val);
    return stream;
}

inline std::string fmtstr(std::string const& s)
{
    return s;
}

inline std::string fmtstr(std::string&& s)
{
    return s;
}

#if defined(_MSC_VER)
std::string fmtstr(char const* format, ...);
#else
std::string fmtstr(char const* format, ...) __attribute__((format(printf, 1, 2)));
#endif

#if defined(_WIN32)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

auto constexpr kDefaultDelimiter = ", ";

template <typename U, typename TStream, typename T>
inline TStream& arr2outCasted(TStream& out, T* arr, size_t size, char const* delim = kDefaultDelimiter)
{
    out << "(";
    if (size > 0)
    {
        for (size_t i = 0; i < size - 1; ++i)
        {
            out << static_cast<U>(arr[i]) << delim;
        }
        out << static_cast<U>(arr[size - 1]);
    }
    out << ")";
    return out;
}

template <typename TStream, typename T>
inline TStream& arr2out(TStream& out, T* arr, size_t size, char const* delim = kDefaultDelimiter)
{
    return arr2outCasted<T>(out, arr, size, delim);
}

template <typename T>
inline std::string arr2str(T* arr, size_t size, char const* delim = kDefaultDelimiter)
{
    std::stringstream ss;
    return arr2out(ss, arr, size, delim).str();
}

template <typename T>
inline std::string vec2str(std::vector<T> const& vec, char const* delim = kDefaultDelimiter)
{
    return arr2str(vec.data(), vec.size(), delim);
}

inline bool strStartsWith(std::string const& str, std::string const& prefix)
{
    return str.rfind(prefix, 0) == 0;
}

std::unordered_set<std::string> str2set(std::string const& input, char delimiter);

}
