
#pragma once
#include "sugesstify/common/assert.h"
#include "sugesstify/common/cudaUtils.h"
#include "sugesstify/common/envUtils.h"
#include "sugesstify/common/logger.h"
#include "sugesstify/common/quantization.h"
#include "../src/cutlass_kernels/cutlass_type_conversion.h"
#include "sugesstify/runtime/common.h"

#include <NvInferRuntime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace sugesstify
{
namespace kernels
{
namespace cuda_core_gemm
{
using SizeType32 = sugesstify::runtime::SizeType32;

struct Params
{
    void const* act;
    void const* weight;
    float alpha;
    void* output;
    SizeType32 m, n, k;
    sugesstify::common::QuantMode quantMode;
    nvinfer1::DataType inputType;
    nvinfer1::DataType outputType;

    Params(void const* _act, void const* _weight, float _alpha, void* _output, SizeType32 _m, SizeType32 _n,
        SizeType32 _k, sugesstify::common::QuantMode _quant_mode, nvinfer1::DataType _inputType,
        nvinfer1::DataType _outputType)
        : act(_act)
        , weight(_weight)
        , alpha(_alpha)
        , output(_output)
        , m(_m)
        , n(_n)
        , k(_k)
        , quantMode(_quant_mode)
        , inputType(_inputType)
        , outputType(_outputType)
    {
    }
};

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream);
}
}
}
