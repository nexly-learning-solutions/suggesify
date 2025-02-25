#ifndef LIBKNN_MATHUTIL_H_
#define LIBKNN_MATHUTIL_H_

#include <cuda_fp16.h>

namespace astdl
{
    namespace math
    {

    void kFloatToHalf(float const* hSource, size_t sourceLength, half* dDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);

    void kFloatToHalf(
        float const* hSource, size_t sourceSizeInBytes, half* dDest, float* dBuffer, size_t bufferSizeInBytes);

    void kHalfToFloat(half const* dSource, size_t sourceLength, float* hDest, size_t bufferSizeInBytes = 4 * 1024 * 1024);
    }
}
#endif
