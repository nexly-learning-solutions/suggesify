
#pragma once
#include "../iBuffer.h"
#include "../iTensor.h"
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvToolsExt.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>

#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_CPU_INPUT(x, st)                                                                                         \
    CHECK_CPU(x);                                                                                                      \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_OPTIONAL_INPUT(x, st)                                                                                    \
    if (x.has_value())                                                                                                 \
    {                                                                                                                  \
        CHECK_INPUT(x.value(), st);                                                                                    \
    }
#define CHECK_OPTIONAL_CPU_INPUT(x, st)                                                                                \
    if (x.has_value())                                                                                                 \
    {                                                                                                                  \
        CHECK_CPU_INPUT(x.value(), st);                                                                                \
    }
#define PRINT_TENSOR(x) std::cout << #x << ":\n" << x << std::endl
#define PRINT_TENSOR_SIZE(x) std::cout << "size of " << #x << ": " << x.sizes() << std::endl

namespace torch_ext
{

template <typename T>
inline T* get_ptr(torch::Tensor& t)
{
    return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline T get_val(torch::Tensor& t, int idx)
{
    assert(idx < t.numel());
    return reinterpret_cast<T*>(t.data_ptr())[idx];
}

suggestify::runtime::ITensor::Shape convert_shape(torch::Tensor tensor);

template <typename T>
suggestify::runtime::ITensor::UniquePtr convert_tensor(torch::Tensor tensor);

size_t sizeBytes(torch::Tensor tensor);

int nextPowerOfTwo(int v);

std::optional<float> getFloatEnv(char const* name);

}
