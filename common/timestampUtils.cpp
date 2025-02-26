
#include <chrono>
#include <iomanip>
#include <sstream>

#include "timestampUtils.h"

namespace suggestify::common
{

std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&now_t);

    auto epoch_to_now = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch_to_now);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(epoch_to_now - seconds);

    std::ostringstream stream;
    stream << std::put_time(&tm, "%m-%d-%Y %H:%M:%S");
    stream << "." << std::setfill('0') << std::setw(6) << us.count();
    return stream.str();
}

}
