
#pragma once

#include <nvtx3/nvtx3.hpp>

#include <array>

namespace suggestify::common::nvtx
{
inline nvtx3::color nextColor()
{
#ifndef NVTX_DISABLE
    constexpr std::array kColors{nvtx3::color{0xff00ff00}, nvtx3::color{0xff0000ff}, nvtx3::color{0xffffff00},
        nvtx3::color{0xffff00ff}, nvtx3::color{0xff00ffff}, nvtx3::color{0xffff0000}, nvtx3::color{0xffffffff}};
    constexpr auto numColors = kColors.size();

    static thread_local std::size_t colorId = 0;
    auto const color = kColors[colorId];
    colorId = colorId + 1 >= numColors ? 0 : colorId + 1;
    return color;
#else
    return nvtx3::color{0};
#endif
}

}

#define NVTX3_SCOPED_RANGE_WITH_NAME(range, name)                                                                      \
    ::nvtx3::scoped_range range(::suggestify::common::nvtx::nextColor(), name)
#define NVTX3_SCOPED_RANGE(range) NVTX3_SCOPED_RANGE_WITH_NAME(range##_range, #range)
