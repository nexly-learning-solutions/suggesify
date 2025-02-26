
#pragma once

#include "cudaBf16Fallbacks.cuh"
#include "cudaFp8Utils.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>

namespace suggestify
{
namespace common
{

template <typename T>
struct QuantTypeStaticVals;

template <>
struct QuantTypeStaticVals<int8_t>
{
    static constexpr float MAX_VAL = 127.f;
    static constexpr float MIN_SCALING_FACTOR = 0.f;
    static constexpr float MIN_SCALING_FACTOR_RCP = FLT_MAX;
};

#ifdef ENABLE_FP8

template <>
struct QuantTypeStaticVals<__nv_fp8_e4m3>
{
    static constexpr float MAX_VAL = 448.f;
    static constexpr float MIN_SCALING_FACTOR = 1.0f / (448.f * 512.f);
    static constexpr float MIN_SCALING_FACTOR_RCP = (448.f * 512.f);
};

#endif

}
}
