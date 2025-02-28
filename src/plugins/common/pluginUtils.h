
#pragma once

#include <NvInferRuntime.h>

#include "../common/logger.h"

namespace suggestify::plugins::utils
{
using DimType64 = int64_t;

inline DimType64 computeMDimension(bool transA, nvinfer1::Dims const& dims)
{
    DimType64 M{1};
    if (transA)
    {
        for (int i = dims.nbDims - 1; i > 0; --i)
        {
            M *= dims.d[i];
        }
    }
    else
    {
        for (int i = 0; i < dims.nbDims - 1; ++i)
        {
            M *= dims.d[i];
        }
    }
    return M;
}

inline DimType64 computeNDimension(bool transB, nvinfer1::Dims const& dims)
{
    DimType64 N{1};
    if (transB)
    {
        for (int32_t i = 0; i < dims.nbDims - 1; ++i)
        {
            N *= dims.d[i];
        }
    }
    else
    {
        for (int32_t i = dims.nbDims - 1; i > 0; --i)
        {
            N *= dims.d[i];
        }
    }
    return N;
}

inline std::int32_t logErrorReturn0(char const* variable)
{
    LOG_ERROR("Value of %s is out of range for int32_t", variable);
    return 0;
}

#define INT32_CAST(value)                                                                                         \
    ((value > 0x7FFFFFFFLL || value < -0x80000000LL) ? suggestify::plugins::utils::logErrorReturn0(#value)           \
                                                     : static_cast<int32_t>(value))

}
