#pragma once
#include "cublasMMWrapper.h"
#include "dataType.h"
#include <cassert>
#include <string>
#include <vector>

namespace suggestify::kernels
{

using CublasGemmWrapper = suggestify::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;
using Config = cublasLtMatmulHeuristicResult_t;

class LoraImpl
{
public:
    LoraImpl() = delete;

    LoraImpl(int in_hidden_size, std::vector<int> out_hidden_sizes, int transA, int transB, int num_lora_modules,
        nvinfer1::DataType type, int max_low_rank, std::shared_ptr<CublasGemmWrapper> cublasWrapper);

    ~LoraImpl() = default;

    size_t getWorkspaceSize(
        int64_t const numTokens, int64_t const numReqs, nvinfer1::DataType const type) const noexcept;
    void setBestTactic(std::optional<Config> config);
    int run(int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
        void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace, cudaStream_t stream);

    void setGemmConfig();

public:
    int mTransA;
    int mTransB;
    nvinfer1::DataType mType;
    int mNumLoraModules;

    CublasGemmWrapperPtr mCublasWrapper;

private:
    int mInHiddenSize;
    std::vector<int> mOutHiddenSizes;
    int mMaxLowRank;
    int const mSplitKSlices = 16;

    std::optional<Config> mBestConfig;
};

}
