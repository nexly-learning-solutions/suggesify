
#pragma once

#include <string>

#ifndef _WIN32
#include <pthread.h>
#endif

namespace suggestify::common
{

inline bool setThreadName(std::string const& name)
{
#ifdef _WIN32
    return false;
#else
    auto const ret = pthread_setname_np(pthread_self(), name.c_str());
    return !ret;
#endif
}

}
