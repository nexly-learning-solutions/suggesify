
#pragma once
#include "../common/quantization.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace sugesstify
{
namespace kernels
{
namespace smooth_quant
{
struct Params
{
    int8_t const* act;
    int8_t const* weight;
    float const* scale_tokens;
    float const* scale_channels;
    void* output;
    int m, n, k;
    sugesstify::common::QuantMode quant_mode;

    Params(int8_t const* _act, int8_t const* _weight, float const* _scale_tokens, float const* _scale_channels,
        void* _output, int _m, int _n, int _k, sugesstify::common::QuantMode _quant_mode)
        : act(_act)
        , weight(_weight)
        , scale_tokens(_scale_tokens)
        , scale_channels(_scale_channels)
        , output(_output)
        , m(_m)
        , n(_n)
        , k(_k)
        , quant_mode(_quant_mode)
    {
    }
};

template <typename>
void int8_sq_launcher(Params& params, cudaStream_t s);
}
}
}
