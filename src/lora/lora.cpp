
#include "../src/lora/lora.h"

#include "assert.h"
#include "cublasMMWrapper.h"
#include "cudaUtils.h"
#include "../src/groupGemm.h"
#include "../src/splitkGroupGemm.h"
#include "iBuffer.h"

#include <algorithm>

using namespace nvinfer1;
using namespace suggestify::common;
using suggestify::kernels::LoraImpl;
using suggestify::kernels::CublasGemmWrapperPtr;

namespace suggestify::kernels
{

void _getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb, int& m, int& n, int& k, int& lda, int& ldb,
    int& ldc, bool transA, bool transB, int M, int N, int K)
{
    transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    m = N;
    n = M;
    k = K;
    lda = transB ? K : N;
    ldb = transA ? M : K;
    ldc = N;
}

void _runGemm(int const M, int const N, int const K, bool const transA, bool const transB,
    const nvinfer1::DataType type, CublasGemmWrapperPtr const& cublasWrapperPtr, void const* act, void const* weight,
    void* output, std::optional<cublasLtMatmulHeuristicResult_t> const& heuristic, void* workspace, cudaStream_t stream)
{
    cublasWrapperPtr->setStream(stream);
    cublasWrapperPtr->setWorkspace(workspace);

    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    _getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K);

    cublasWrapperPtr->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    cublasWrapperPtr->Gemm(transa, transb, m, n, k, weight, lda, act, ldb, output, ldc, heuristic);
    cublasWrapperPtr->destroyDescriptors();
}

LoraImpl::LoraImpl(int in_hidden_size, std::vector<int> out_hidden_sizes, int transA, int transB, int num_lora_modules,
    nvinfer1::DataType type, int max_low_rank, std::shared_ptr<CublasGemmWrapper> cublasWrapper)
    : mInHiddenSize(in_hidden_size)
    , mTransA(transA)
    , mTransB(transB)
    , mNumLoraModules(num_lora_modules)
    , mType(type)
    , mMaxLowRank(max_low_rank)
    , mCublasWrapper(cublasWrapper)
{
    mOutHiddenSizes.resize(mNumLoraModules);
    mOutHiddenSizes.assign(out_hidden_sizes.begin(), out_hidden_sizes.end());
    LOG_DEBUG("%s", __PRETTY_FUNCTION__);
}

void LoraImpl::setGemmConfig()
{
    LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    if (mType == DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if (mType == DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif
}

int64_t getLowRankWorkSpaceSize(int64_t numTokens, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t typeSize)
{
    return divUp(numTokens * maxLoraModuleNum * maxLowRank * typeSize, 16) * 16;
}

int64_t getGemmParamsWorkSpaceSize(int64_t nbReq)
{
    return std::max(getSplitkGroupedGemmParamsWorkSpaceSize(nbReq), getGroupedGemmParamsWorkSpaceSize(nbReq));
}

int64_t getSplitkGroupedGemmWorkSpaceSize(
    int64_t numTokens, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t splitKSlices)
{
    return divUp(numTokens * maxLoraModuleNum * maxLowRank * sizeof(float) * splitKSlices, 16) * 16;
}

int64_t getGemmWorkSpaceSize(int64_t numTokens, int64_t maxLoraModuleNum, int64_t maxLowRank, int64_t splitKSlices)
{
    return std::max((int64_t) CUBLAS_WORKSPACE_SIZE,
        getSplitkGroupedGemmWorkSpaceSize(numTokens, maxLoraModuleNum, maxLowRank, splitKSlices));
}

size_t LoraImpl::getWorkspaceSize(
    int64_t const numTokens, int64_t const numReqs, nvinfer1::DataType const type) const noexcept
{
    LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    auto const typeSize = suggestify::common::getDTypeSize(type);

    return (size_t) getGemmWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, mSplitKSlices)
        + getLowRankWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, typeSize)
        + getGemmParamsWorkSpaceSize(std::min(numReqs, numTokens) * mNumLoraModules);
}

void LoraImpl::setBestTactic(std::optional<Config> config)
{
    mBestConfig = std::move(config);
}

int LoraImpl::run(int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
    void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace, cudaStream_t stream)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (numTokens == 0)
    {
        return 0;
    }

    auto const typeSize = suggestify::runtime::BufferDataType(mType).getSize();
    setGemmConfig();

    int64_t GemmWorkSpaceSize = getGemmWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, mSplitKSlices);
    int64_t groupGemmParamsWorkSpaceSize = getGemmParamsWorkSpaceSize(std::min(numReqs, numTokens) * mNumLoraModules);
    void* gemmWorkSpace = workspace;
    void* lowRankWorkSpace = static_cast<char*>(gemmWorkSpace) + GemmWorkSpaceSize;
    void* groupGemmParamsWorkSpace = static_cast<char*>(lowRankWorkSpace)
        + getLowRankWorkSpaceSize(numTokens, mNumLoraModules, mMaxLowRank, typeSize);

    for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
    {
        size_t size = numTokens * mOutHiddenSizes[loraModuleIdx];
        cudaMemsetAsync(outputs[loraModuleIdx], 0, size * typeSize, stream);
    }

    char* useUnifiedGemmChar = std::getenv("LORA_USE_UNIFIED_GEMM");
    bool useUnifiedGemm = (useUnifiedGemmChar == nullptr || std::string(useUnifiedGemmChar) != "OFF");

    for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
    {
        auto const loraRankModule = loraRanks[loraModuleIdx * numTokens];
        void const* const* loraWeightsPtrModule = &loraWeightsPtr[loraModuleIdx * numTokens * 2];
        for (int rowId = 0; rowId < numTokens; rowId++)
        {
            if (loraWeightsPtrModule[rowId * 2] != loraWeightsPtrModule[0]
                || loraWeightsPtrModule[rowId * 2 + 1] != loraWeightsPtrModule[1]
                || loraRanks[loraModuleIdx * numTokens + rowId] != loraRankModule)
            {
                useUnifiedGemm = false;
            }
        }
    }

    if (useUnifiedGemm)
    {
        for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
        {
            int64_t const* loraWeightsPtrModule
                = reinterpret_cast<int64_t const*>(&loraWeightsPtr[loraModuleIdx * numTokens * 2]);

            int M = numTokens;

            auto const lora_rank = loraRanks[loraModuleIdx * numTokens];

            auto const N = lora_rank;

            if (N > 0)
            {
                CHECK_WITH_INFO(N <= mMaxLowRank,
                    fmtstr("Invalid low_rank (%d). low_rank must be smaller than mMaxLowRank (%d)", N, mMaxLowRank));
                auto const K = mInHiddenSize;
                auto const N2 = mOutHiddenSizes[loraModuleIdx];

                void* lora_in_weight
                    = reinterpret_cast<void*>(loraWeightsPtrModule[0] + K * N * typeSize * weightIndex);
                void* lora_out_weight
                    = reinterpret_cast<void*>(loraWeightsPtrModule[1] + N2 * N * typeSize * weightIndex);
                void* output = outputs[loraModuleIdx];

                _runGemm(M, N, K, mTransA, mTransB, mType, mCublasWrapper, input, lora_in_weight, lowRankWorkSpace,
                    mBestConfig, gemmWorkSpace, stream);

                _runGemm(M, N2, N, false, mTransB, mType, mCublasWrapper, lowRankWorkSpace, lora_out_weight, output,
                    mBestConfig, gemmWorkSpace, stream);
            }
        }
    }
    else
    {
        std::vector<cutlass::gemm::GemmCoord> problem_sizes;
        problem_sizes.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrA;
        ptrA.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrB;
        ptrB.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrC;
        ptrC.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrD;
        ptrD.reserve(numTokens * mNumLoraModules);

        std::vector<cutlass::gemm::GemmCoord> problem_sizes_2;
        problem_sizes_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrA_2;
        ptrA_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrB_2;
        ptrB_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrC_2;
        ptrC_2.reserve(numTokens * mNumLoraModules);
        std::vector<void*> ptrD_2;
        ptrD_2.reserve(numTokens * mNumLoraModules);

        int minKN = mInHiddenSize;
        for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
        {
            int64_t const* loraWeightsPtrModule
                = reinterpret_cast<int64_t const*>(&loraWeightsPtr[loraModuleIdx * numTokens * 2]);
            int32_t const* loraRanksModule = &loraRanks[loraModuleIdx * numTokens];

            int rowId = 0;
            int handled_token_num = 0;
            while (rowId < numTokens)
            {
                auto const lora_rank = loraRanksModule[rowId];
                auto const N = lora_rank;
                int count = 0;
                size_t M = 0;
                while (rowId + count < numTokens && lora_rank == loraRanksModule[rowId + count]
                    && loraWeightsPtrModule[rowId * 2] == loraWeightsPtrModule[(rowId + count) * 2]
                    && loraWeightsPtrModule[rowId * 2 + 1] == loraWeightsPtrModule[(rowId + count) * 2 + 1])
                {
                    M += 1;
                    count++;
                }

                if (N > 0)
                {
                    CHECK_WITH_INFO(N <= mMaxLowRank,
                        fmtstr(
                            "Invalid low_rank (%d). low_rank must be smaller than mMaxLowRank (%d)", N, mMaxLowRank));

                    auto const K = mInHiddenSize;
                    minKN = std::min(minKN, N);
                    minKN = std::min(minKN, K);

                    cutlass::gemm::GemmCoord problem(M, N, K);
                    problem_sizes.push_back(problem);

                    ptrA.push_back(static_cast<void*>(
                        static_cast<char*>(const_cast<void*>(input)) + handled_token_num * K * typeSize));
                    ptrB.push_back(
                        reinterpret_cast<void*>(loraWeightsPtrModule[rowId * 2] + K * N * typeSize * weightIndex));
                    ptrC.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * numTokens * mMaxLowRank + handled_token_num * mMaxLowRank) * typeSize));
                    ptrD.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * numTokens * mMaxLowRank + handled_token_num * mMaxLowRank) * typeSize));

                    auto const N2 = mOutHiddenSizes[loraModuleIdx];
                    cutlass::gemm::GemmCoord problem_2(M, N2, N);
                    problem_sizes_2.push_back(problem_2);
                    ptrA_2.push_back(static_cast<void*>(static_cast<char*>(lowRankWorkSpace)
                        + (loraModuleIdx * numTokens * mMaxLowRank + handled_token_num * mMaxLowRank) * typeSize));
                    ptrB_2.push_back(
                        reinterpret_cast<void*>(loraWeightsPtrModule[rowId * 2 + 1] + N2 * N * typeSize * weightIndex));
                    ptrC_2.push_back(static_cast<void*>(
                        static_cast<char*>(outputs[loraModuleIdx]) + handled_token_num * N2 * typeSize));
                    ptrD_2.push_back(static_cast<void*>(
                        static_cast<char*>(outputs[loraModuleIdx]) + handled_token_num * N2 * typeSize));
                }
                handled_token_num += M;
                rowId += count;
            }
            CHECK(handled_token_num == numTokens);
        }
        if (problem_sizes.size() > 0)
        {
            CHECK_WITH_INFO(mTransA == false && mTransB == true,
                fmtstr("Invalid transA (%d) transB (%d). transA must be false, transB must be true", int(mTransA),
                    int(mTransB)));
            splitkGroupedGemm(problem_sizes, ptrA, ptrB, ptrC, ptrD, groupGemmParamsWorkSpace,
                groupGemmParamsWorkSpaceSize, gemmWorkSpace, GemmWorkSpaceSize, true, mType, mSplitKSlices, minKN,
                stream);
            sync_check_cuda_error();
            groupedGemm(problem_sizes_2, ptrA_2, ptrB_2, ptrC_2, ptrD_2, groupGemmParamsWorkSpace,
                groupGemmParamsWorkSpaceSize, gemmWorkSpace, GemmWorkSpaceSize, false, mType, minKN, stream);
            sync_check_cuda_error();
        }
    }

    return 0;
}

}
