#pragma once

#include "cutlass/gemm_coord.h"
#include <NvInferRuntime.h>

namespace suggestify
{
namespace kernels
{

int64_t getGroupedGemmParamsWorkSpaceSize(int64_t problem_count);

void groupedGemm(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkspace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
    bool isLoraIn, nvinfer1::DataType dataType, int minKN, cudaStream_t stream);

}

}
