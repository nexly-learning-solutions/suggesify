
#pragma once

#include "logger.h"
#include <NvInferRuntime.h>

namespace suggestify::common
{

constexpr static size_t getDTypeSize(nvinfer1::DataType type)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#endif
    switch (type)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32: [[fallthrough]];
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16: [[fallthrough]];
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL: [[fallthrough]];
    case nvinfer1::DataType::kUINT8: [[fallthrough]];
    case nvinfer1::DataType::kINT8: [[fallthrough]];
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4: TLLM_THROW("Cannot determine size of INT4 data type");
    default: return 0;
    }
    return 0;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

}
