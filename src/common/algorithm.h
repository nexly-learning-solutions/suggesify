
#pragma once

namespace suggestify
{

struct Algorithm
{
    Algorithm() = default;
    Algorithm(Algorithm&&) = default;
    Algorithm& operator=(Algorithm&&) = default;
    Algorithm(Algorithm const&) = delete;
    Algorithm& operator=(Algorithm const&) = delete;
};

}
