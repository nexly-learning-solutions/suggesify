
#pragma once

#include "cudaUtils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <optional>
#include <string>

namespace suggestify
{
namespace common
{

class CublasMMWrapper
{
protected:
    std::shared_ptr<cublasHandle_t> mCublasHandle;
    std::shared_ptr<cublasLtHandle_t> mCublasLtHandle;

    cudaDataType_t mAType{};
    cudaDataType_t mBType{};
    cudaDataType_t mCType{};
    cublasComputeType_t mComputeType{};
    cudaDataType_t mScaleType{};

    cublasLtMatmulDesc_t mOperationDesc{NULL};
    cublasLtMatrixLayout_t mADesc{NULL};
    cublasLtMatrixLayout_t mBDesc{NULL};
    cublasLtMatrixLayout_t mCDesc{NULL};

    cudaStream_t mStream;

    void* mCublasWorkspace = nullptr;

private:
    bool descriptorsCreated() const
    {
        return mOperationDesc != NULL && mADesc != NULL && mBDesc != NULL && mCDesc != NULL;
    }

public:
    CublasMMWrapper(std::shared_ptr<cublasHandle_t> cublasHandle, std::shared_ptr<cublasLtHandle_t> cublasLtHandle,
        cudaStream_t stream, void* workspace);

    ~CublasMMWrapper();

    CublasMMWrapper(CublasMMWrapper const& wrapper);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k, void const* A,
        int const lda, void const* B, int const ldb, void* C, int const ldc);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k, void const* A,
        int const lda, void const* B, int const ldb, void* C, int const ldc,
        std::optional<cublasLtMatmulHeuristicResult_t> const& algo);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k, void const* A,
        int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta,
        std::optional<cublasLtMatmulHeuristicResult_t> const& algo);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k, void const* A,
        int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta);

    void Gemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k, void const* A,
        int const lda, void const* B, int const ldb, void* C, int const ldc, float f_alpha, float f_beta,
        cublasLtMatmulAlgo_t const& algo, bool hasAlgo, bool usingCublasLt);

    void stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
        void const* A, int const lda, const int64_t strideA, void const* B, int const ldb, const int64_t strideB,
        void* C, int const ldc, const int64_t strideC, int const batchCount, float const f_alpha = 1.0f,
        float const f_beta = 0.0f);

    void stridedBatchedGemm(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
        float const f_alpha, void const* A, cudaDataType_t AType, int const lda, const int64_t strideA, void const* B,
        cudaDataType_t BType, int const ldb, const int64_t strideB, float const f_beta, void* C, cudaDataType_t CType,
        int const ldc, const int64_t strideC, int const batchCount, cudaDataType_t computeType);

    bool checkTactic(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
        int const lda, int const ldb, int const ldc, cublasLtMatmulAlgo_t const& algo);

    std::vector<cublasLtMatmulHeuristicResult_t> getTactics(cublasOperation_t transa, cublasOperation_t transb,
        int const m, int const n, int const k, int const lda, int const ldb, int const ldc);

    std::vector<cublasLtMatmulHeuristicResult_t> getTactics(cublasLtHandle_t lightHandle,
        cublasLtMatmulDesc_t computeDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
        cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc);

    using MatrixLayout = std::tuple<cudaDataType_t, cublasLtOrder_t, uint64_t, uint64_t>;
    using cache_idx_t = std::tuple<cublasLtMatmulDesc_t, std::array<MatrixLayout, 4>>;

    MatrixLayout createMatrixLayout(cublasLtMatrixLayout_t Mdesc);

    void setWorkspace(void* workspace);

    void setFP32GemmConfig();
    void setFP16GemmConfig(cudaDataType_t outputType = CUDA_R_16F);
#ifdef ENABLE_BF16
    void setBF16GemmConfig(cudaDataType_t outputType = CUDA_R_16BF);
#endif
#ifdef ENABLE_FP8
    void setFP8GemmConfig(cudaDataType_t outputType = CUDA_R_16F);
#endif

    void setStream(cudaStream_t stream);

    void setGemmConfig(cudaDataType_t aType, cudaDataType_t bType, cudaDataType_t cType, cudaDataType_t computeType);

    CublasDataType getCublasDataType(cudaDataType_t data_type);

    void createDescriptors(cublasOperation_t transa, cublasOperation_t transb, int const m, int const n, int const k,
        int const lda, int const ldb, int const ldc, int8_t fastAcc = 0);
    void setScaleDescriptors(void* scale_a, void* scale_b);
    void destroyDescriptors();

    cublasHandle_t getCublasHandle()
    {
        return *(this->mCublasHandle);
    }

    cublasLtHandle_t getCublasLtHandle() const
    {
        return *(this->mCublasLtHandle);
    }
};

}

}
