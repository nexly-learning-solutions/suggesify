#pragma once

#include "cutlass/gemm_coord.h"
#include <NvInferRuntime.h>

namespace sugesstify
{
namespace kernels
{

int64_t getSplitkGroupedGemmParamsWorkSpaceSize(int64_t problem_count);

void splitkGroupedGemm(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkspace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
    bool isLoraIn, nvinfer1::DataType dataType, int splitKSlices, int minKN, cudaStream_t stream);

}

}
