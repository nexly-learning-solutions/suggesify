#include "suggestify/common/cudaUtils.h"
#include "suggestify/plugins/common/plugin.h"
#include "suggestify/plugins/gemmPlugin/gemmPlugin.h"
#include "suggestify/runtime/torchUtils.h"
#include "thUtils.h"
#include <cublasLt.h>
#include <torch/extension.h>

using torch::Tensor;

namespace torch_ext
{

namespace
{

using suggestify::common::check;

void cublas_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b)
{

    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[1];
    int32_t k = a.sizes()[1];

    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    auto cublasWrapper = getCublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, nullptr);

    auto const dtype = out.dtype();
    cudaDataType_t aType = CUDA_R_8F_E4M3;
    cudaDataType_t bType = CUDA_R_8F_E4M3;
    cudaDataType_t outType;
    if (dtype == torch::kFloat32)
    {
        outType = CUDA_R_32F;
    }
    else if (dtype == torch::kHalf)
    {
        outType = CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (dtype == torch::kBFloat16)
    {
        outType = CUDA_R_16BF;
    }
#endif
    else
    {
        THROW("Unsupported output data type");
    }
    cublasComputeType_t compType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scalarType = CUDA_R_32F;
    cublasWrapper->setGemmConfig(aType, bType, outType,scalarType);

    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(CUBLAS_WORKSPACE_SIZE, workspace_options);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    auto* ws_ptr = static_cast<void*>(workspace.data_ptr());
    auto* a_scale = static_cast<void*>(scale_a.data_ptr());
    auto* b_scale = static_cast<void*>(scale_b.data_ptr());

    cublasWrapper->setStream(stream);
    cublasWrapper->setWorkspace(ws_ptr);

    cublasLtMatmulAlgo_t algo;
    int32_t const mp2 = nextPowerOfTwo(m);

    int const algoID = 52;
    check_cuda_error(
        cublasLtMatmulAlgoInit(*cublasLtHandle, compType, scalarType, aType, bType, outType, outType, algoID, &algo));
    int tileID = CUBLASLT_MATMUL_TILE_256x128;
    int swizzle = 0;
    uint16_t cta = CUBLASLT_CLUSTER_SHAPE_2x1x1;
    if (mp2 <= 64)
    {
        tileID = CUBLASLT_MATMUL_TILE_64x64;
        swizzle = 1;
        if (n > k)
            cta = CUBLASLT_CLUSTER_SHAPE_13x1x1;
        else
            cta = CUBLASLT_CLUSTER_SHAPE_10x1x1;
    }
    else if (mp2 <= 256)
    {
        if (n > k)
            tileID = CUBLASLT_MATMUL_TILE_192x128;
        else
            tileID = CUBLASLT_MATMUL_TILE_128x128;
        swizzle = 1;
        cta = CUBLASLT_CLUSTER_SHAPE_1x2x1;
    }
    else if (mp2 <= 2048)
    {
        if (n > k)
            tileID = CUBLASLT_MATMUL_TILE_160x128;
        else
            tileID = CUBLASLT_MATMUL_TILE_256x128;
    }
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileID, sizeof(tileID)));
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cta, sizeof(cta)));
    int const stagesID = CUBLASLT_MATMUL_STAGES_128xAUTO;
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesID, sizeof(stagesID)));
    int const numsK = -1;
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numsK, sizeof(numsK)));
    int const reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    check_cuda_error(cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof(reduction)));
    check_cuda_error(
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle)));

    cublasWrapper->createDescriptors(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,k,k,n,1);
    cublasWrapper->setScaleDescriptors(a_scale, b_scale);
    cublasWrapper->Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,b_ptr,k,a_ptr,k, out_ptr,
n, 1.0F, 0.0F, algo, true, true);
    cublasWrapper->destroyDescriptors();
}

}

Tensor& cublas_scaled_mm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype, Tensor& out)
{
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(out);

    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(out.sizes()[0] == mat_a.sizes()[0] && mat_a.sizes()[1] == mat_b.sizes()[0]
        && mat_b.sizes()[1] == out.sizes()[1]);
    TORCH_CHECK(scale_a.numel() == 1 || scale_a.numel() == mat_a.sizes()[0]);
    TORCH_CHECK(scale_b.numel() == 1 || scale_b.numel() == mat_b.sizes()[1]);

    TORCH_CHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1);
    TORCH_CHECK(mat_b.strides()[0] == 1);
    TORCH_CHECK(out.strides()[0] % 16 == 0 && mat_b.strides()[1] % 16 == 0);
    TORCH_CHECK(scale_a.is_contiguous() && scale_b.is_contiguous());

    TORCH_CHECK(!bias.has_value(), "bias is not support yet");

    TORCH_CHECK(mat_a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(mat_b.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");

    cublas_gemm_caller(out, mat_a, mat_b, scale_a, scale_b);
    return out;
}

Tensor cublas_scaled_mm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    auto const out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    Tensor out = at::empty({mat_a.sizes()[0], mat_b.sizes()[1]}, mat_a.options().dtype(out_dtype_));
    return cublas_scaled_mm_out(mat_a, mat_b, scale_a, scale_b, bias, out_dtype, out);
}

}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cublas_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor? bias,"
        " ScalarType? out_dtype) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cublas_scaled_mm", &torch_ext::cublas_scaled_mm);
}
