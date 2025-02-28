
#pragma once

#include <cuda_runtime.h>

namespace suggestify
{
namespace common
{


template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}


}
}
