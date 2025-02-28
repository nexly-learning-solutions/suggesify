
#include "../common/cudaTypeUtils.cuh"
#include "../common/quantTypeUtils.cuh"
#include "../common/reduceKernelUtils.cuh"
#include "rmsnormKernels.h"

using namespace sugesstify::common;

namespace sugesstify
{
namespace kernels
{

template <typename Tf, typename T>
__inline__ __device__ Tf compute_rmsnorm(Tf val, float s_variance, T const* gamma, T const* beta, int i)
{
    Tf ret = val * s_variance * cuda_cast<Tf>(gamma[i]);
    if (beta != nullptr)
    {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}

template <typename T, typename QuantT, bool USE_SHMEM>
__global__ void generalRmsNorm(T const* input, T const* gamma, T const* beta, T* normed_output, float const eps,
    int tokens, int hidden_dim, float const* clampPtr, float const* scale_orig_quant_per_tensor,
    float* scale_orig_quant_per_token, float* sum_per_token, QuantT* normed_output_quant, bool hasFp8MinScaling)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using QuantT_packed_t = typename packed_as<QuantT, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    T const clampMin = cuda_cast<T>(clampPtr ? clampPtr[0] : -FLT_MAX);
    T const clampMax = cuda_cast<T>(clampPtr ? clampPtr[1] : FLT_MAX);

    static constexpr float MAX_QUANT_VAL = QuantTypeStaticVals<QuantT>::MAX_VAL;
    static constexpr float MIN_SCALING_FACTOR = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR;
    static constexpr float MIN_SCALING_FACTOR_RCP = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR_RCP;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);

    __shared__ float s_variance;

    int const tidx = threadIdx.x;
    int const bidx = blockIdx.x;

    float variance = 0.0f;
    float local_var_sum = 0.0f;

    int const n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        T const val = input[bidx * n_elems + i];
        if (USE_SHMEM)
        {
            shmem[i] = val;
        }

        float_packed_t const val_f = cuda_cast<float_packed_t>(val);

        local_var_sum += cuda_sum<float>(val_f * val_f);
    }

    float packed[1] = {local_var_sum};
    blockReduceSumV2<float, 1>(packed);
    variance = packed[0];

    if (threadIdx.x == 0)
    {
        variance = (variance / hidden_dim);
        s_variance = rsqrtf(variance + eps);
    }
    __syncthreads();

    bool const with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    bool const with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    bool const with_per_token_sum = sum_per_token != nullptr;

    float_packed_t const scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? *scale_orig_quant_per_tensor : 0.0f);
    T_scalar amax = 1e-6f;
    float local_sum = 0.f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        int const index = bidx * n_elems + i;
        float_packed_t const val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : input[index]);
        T val = cuda_cast<T>(compute_rmsnorm(val_f, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            val = cuda_clamp(val, clampMin, clampMax);
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if (USE_SHMEM)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            val = cuda_clamp(val, clampMin, clampMax);
            reinterpret_cast<QuantT_packed_t*>(normed_output_quant)[index]
                = cuda_cast<QuantT_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }

        if (with_per_token_sum)
        {
            local_sum += cuda_sum<float>(cuda_cast<float_packed_t>(val));
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        float const dynamic_per_token_scale
            = hasFp8MinScaling ? fminf(MAX_QUANT_VAL / abs_max_f, MIN_SCALING_FACTOR_RCP) : (MAX_QUANT_VAL / abs_max_f);
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            int const index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : input[index]);
            if (!USE_SHMEM)
            {
                val_f = compute_rmsnorm(val_f, s_variance, gamma, beta, i);
            }

            reinterpret_cast<QuantT_packed_t*>(normed_output_quant)[index]
                = cuda_cast<QuantT_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = hasFp8MinScaling
                ? cuda_max(abs_max_f / MAX_QUANT_VAL, MIN_SCALING_FACTOR)
                : abs_max_f / MAX_QUANT_VAL;
        }
    }

    if (with_per_token_sum)
    {
        float packed_sum[1] = {local_sum};
        blockReduceSumV2<float, 1>(packed_sum);
        if (tidx == 0)
        {
            sum_per_token[bidx] = packed_sum[0];
        }
    }
}

template <typename T, typename QuantT>
void dispatch_rmsnorm_type_square_method(T const* input, T const* gamma, T const* beta, T* normed_output,
    float const eps, int tokens, int hidden_dim, float const* clampPtr, float const* scale_orig_quant_per_tensor,
    float* scale_orig_quant_per_token, float* sum_per_token, QuantT* normed_output_quant, bool const hasFp8MinScaling,
    dim3 const grid, dim3 const block, size_t const shmem_size, cudaStream_t stream)
{
    bool use_shmem = true;
    if (shmem_size >= (48 << 10))
    {
        cudaError_t ret = cudaFuncSetAttribute(
            generalRmsNorm<T, QuantT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        use_shmem = (ret == cudaSuccess);
    }

    if (use_shmem)
    {
        generalRmsNorm<T, QuantT, true><<<grid, block, shmem_size, stream>>>(input, gamma, beta, normed_output, eps,
            tokens, hidden_dim, clampPtr, scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token,
            normed_output_quant, hasFp8MinScaling);
    }
    else
    {
        generalRmsNorm<T, QuantT, false><<<grid, block, shmem_size, stream>>>(input, gamma, beta, normed_output, eps,
            tokens, hidden_dim, clampPtr, scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token,
            normed_output_quant, hasFp8MinScaling);
    }
}

template <typename T, typename QuantT>
void dispatch_rmsnorm_type(T const* input, T const* gamma, T const* beta, T* normed_output, float const eps, int tokens,
    int hidden_dim, float const* clampPtr, float const* scale_orig_quant_per_tensor, float* scale_orig_quant_per_token,
    float* sum_per_token, QuantT* normed_output_quant, bool const hasFp8MinScaling, dim3 const grid, dim3 const block,
    size_t const shmem_size, cudaStream_t stream)
{
    dispatch_rmsnorm_type_square_method(input, gamma, beta, normed_output, eps, tokens, hidden_dim, clampPtr,
        scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token, normed_output_quant, hasFp8MinScaling,
        grid, block, shmem_size, stream);
}

template <typename T, typename QuantT>
void invokeGeneralRmsNorm(T* out, T const* input, T const* gamma, T const* beta, float const eps, int const tokens,
    int const hidden_dim, QuantMode quantMode, cudaStream_t stream, float const* clampPtr, float const* scale,
    float* dynamic_scale, float* sum_per_token, QuantT* normed_output_quant)
{
    dim3 grid(tokens);
    dim3 block(min(hidden_dim, 1024));
    block.x = 32 * ((block.x + 31) / 32);

    constexpr size_t vec_size = 2;
    size_t const shmem_size = hidden_dim * sizeof(T);
    bool const use_vec_type = (hidden_dim % vec_size == 0)
        && (std::is_same<T, half>::value
#ifdef ENABLE_BF16
            || std::is_same<T, __nv_bfloat16>::value
#endif
        );

    bool hasFp8MinScaling = quantMode.hasFp8RowWise();

    if (use_vec_type)
    {
        using Tp = typename packed_as<T, vec_size>::type;
        dispatch_rmsnorm_type(reinterpret_cast<Tp const*>(input), reinterpret_cast<Tp const*>(gamma),
            reinterpret_cast<Tp const*>(beta), reinterpret_cast<Tp*>(out), eps, tokens, hidden_dim, clampPtr, scale,
            dynamic_scale, sum_per_token, normed_output_quant, hasFp8MinScaling, grid, block, shmem_size, stream);
    }
    else
    {
        dispatch_rmsnorm_type(input, gamma, beta, out, eps, tokens, hidden_dim, clampPtr, scale, dynamic_scale,
            sum_per_token, normed_output_quant, hasFp8MinScaling, grid, block, shmem_size, stream);
    }
}

#define INSTANTIATE_GENERAL_RMSNORM(T, QuantT)                                                                         \
    template void invokeGeneralRmsNorm(T* out, const T* input, const T* gamma, const T* beta, const float eps,         \
        const int tokens, const int hidden_dim, QuantMode quantMode, cudaStream_t stream, float const* clampPtr,       \
        const float* scale, float* dynamic_scale, float* sum_per_token, QuantT* normed_output_quant);

INSTANTIATE_GENERAL_RMSNORM(float, int8_t);
INSTANTIATE_GENERAL_RMSNORM(half, int8_t);

#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_RMSNORM(__nv_bfloat16, int8_t);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_GENERAL_RMSNORM(float, __nv_fp8_e4m3);
INSTANTIATE_GENERAL_RMSNORM(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_RMSNORM(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

}
}
