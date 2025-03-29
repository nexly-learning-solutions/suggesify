#include <cutlass/numeric_conversion.h>
#include "../src/weightOnlyBatchedGemv/cudaCoreGemm.h"
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <type_traits>

namespace sugesstify {
namespace kernels {
namespace cuda_core_gemm {

constexpr int kWarpSize = 32;
constexpr int kAlignmentInputFP8 = 16;
constexpr int kAlignmentInputHalf = 8;
constexpr int kAlignmentInputBfloat16 = 8;
constexpr int kAlignmentInputFloat = 4;
constexpr int kMinTileM = 1;
constexpr int kMaxTileM = 16;
constexpr int kTileN = 2;
constexpr int kBlockSize = 128;

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t status = (call);                                     \
        if (status != cudaSuccess) {                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)     \
                      << " at " << __FILE__ << ":" << __LINE__           \
                      << " - error code: " << static_cast<int>(status)   \
                      << std::endl;                                      \
            std::cerr << "Halting execution due to CUDA error." << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#ifdef DEBUG
#define DEBUG_PRINT(...)                                                        \
    do {                                                                        \
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << __VA_ARGS__ << std::endl; \
    } while (0)
#else
#define DEBUG_PRINT(...)
#endif


template <typename SizeType32>
std::pair<dim3, dim3> getGridBlockDims(SizeType32 m, SizeType32 n, int TILE_M, int TILE_N, int BLOCK_SIZE) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((m + TILE_M - 1) / TILE_M, (n + TILE_N - 1) / TILE_N);
    return std::make_pair(grid, block);
}

template <typename T>
struct IsSupportedInputType : std::false_type {};
template <> struct IsSupportedInputType<__nv_fp8_e4m3> : std::true_type {};
template <> struct IsSupportedInputType<half> : std::true_type {};
template <> struct IsSupportedInputType<__nv_bfloat16> : std::true_type {};
template <> struct IsSupportedInputType<float> : std::true_type {};

template <typename InputType, typename OutputType, SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
__global__ void cudaCoreGemm(InputType const* __restrict__ act, InputType const* __restrict__ weight, float alpha,
    OutputType* __restrict__ output, SizeType32 m, SizeType32 n, SizeType32 k) {

    using VecType = int4;
    using CvtInputType = typename sugesstify::kernels::cutlass_kernels::TllmToCutlassTypeAdapter<InputType>::type;
    using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 4>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;

    constexpr SizeType32 kStepK = static_cast<SizeType32>(128 / (8 * sizeof(InputType)));
    constexpr SizeType32 kTileK = kStepK * BLOCK_SIZE;
    constexpr SizeType32 kCvtCount = static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

    auto tileIdM = static_cast<SizeType32>(blockIdx.x * TILE_M);
    auto tileIdN = static_cast<SizeType32>(blockIdx.y * TILE_N);
    auto tid = static_cast<SizeType32>(threadIdx.x);

    float tile_a[kStepK];
    float tile_w[TILE_N * kStepK];
    float acc[TILE_M * TILE_N];

#pragma unroll
    for (SizeType32 i = 0; i < TILE_M * TILE_N; ++i) {
        acc[i] = 0;
    }

    act += tileIdM * k;
    weight += tileIdN * k;
    output += tileIdM * n + tileIdN;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    for (SizeType32 idxK = tid * kStepK; idxK < k; idxK += kTileK) {

        for (SizeType32 i = 0; i < TILE_N; ++i) {
            auto tile_w_quantized = reinterpret_cast<VecType const*>(weight + i * k + idxK)[0];
#pragma unroll
            for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
                reinterpret_cast<CvtResType*>(tile_w)[i * kCvtCount + cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvtIdx]);
            }
        }

        for (SizeType32 i = 0; i < TILE_M; ++i) {
            auto tile_a_quantized = reinterpret_cast<VecType const*>(act + i * k + idxK)[0];
#pragma unroll
            for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
                reinterpret_cast<CvtResType*>(tile_a)[cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvtIdx]);
            }
#pragma unroll
            for (SizeType32 j = 0; j < TILE_N; ++j) {
#pragma unroll
                for (SizeType32 l = 0; l < kStepK; ++l) {
                    acc[i * TILE_N + j] = fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
                }
            }
        }
    }

    typedef cub::WarpReduce<float> WarpReduce;
    static constexpr SizeType32 kWarpNum = BLOCK_SIZE / kWarpSize;
    SizeType32 warpId = tid / kWarpSize, laneId = tid % kWarpSize;

    __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];

#pragma unroll
    for (SizeType32 mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
        for (SizeType32 ni = 0; ni < TILE_N; ++ni) {
            float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
            if (laneId == 0) {
                shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();

    for (SizeType32 ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE) {
        SizeType32 mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (SizeType32 jj = 0; jj < kWarpNum; ++jj) {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val * alpha);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}


template <typename InputType, typename OutputType, SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
void cudaCoreGemmKernel(Params const& params, cudaStream_t stream) {
    auto [grid, block] = getGridBlockDims<SizeType32>(params.m, params.n, TILE_M, TILE_N, BLOCK_SIZE);

    if (sugesstify::common::getEnvEnablePDL()) {
        DEBUG_PRINT("Enabling PDL for cudaCoreGemmKernel");
        cudaLaunchConfig_t kernelConfig = {0};
        kernelConfig.gridDim = grid;
        kernelConfig.blockDim = block;
        kernelConfig.dynamicSmemBytes = 0;
        kernelConfig.stream = stream;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        kernelConfig.attrs = attribute;
        kernelConfig.numAttrs = 1;

        CUDA_CHECK(cudaLaunchKernelEx(&kernelConfig, cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>,
            reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight),
            params.alpha, reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k));
    } else {
        CUDA_CHECK(cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE><<<grid, block, 0, stream>>>(
            reinterpret_cast<InputType const*>(params.act), reinterpret_cast<InputType const*>(params.weight),
            params.alpha, reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k));
    }
}

template <typename InputType, typename OutputType, int TILE_M>
bool cudaCoreGemmTemplateCaller(Params const& params, cudaStream_t stream) {
    static_assert(TILE_M >= kMinTileM && TILE_M <= kMaxTileM, "TILE_M out of bounds.");
    if (params.m == TILE_M) {
        DEBUG_PRINT("Launching kernel for TILE_M = %d", TILE_M);
        cudaCoreGemmKernel<InputType, OutputType, TILE_M, kTileN, kBlockSize>(params, stream);
        return true;
    }
    if constexpr (TILE_M < kMaxTileM) {
        return cudaCoreGemmTemplateCaller<InputType, OutputType, TILE_M + 1>(params, stream);
    }
    return false;
}

template <typename InputType, typename OutputType>
bool cudaCoreGemmLauncher(Params const& params, cudaStream_t stream) {
    DEBUG_PRINT("Trying cudaCoreGemmLauncher for input=%d, output=%d, m=%d, n=%d, k=%d",
        params.inputType, params.outputType, params.m, params.n, params.k);
    return cudaCoreGemmTemplateCaller<InputType, OutputType, kMinTileM>(params, stream);
}

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream) {
    bool dispatched = false;

    if (params.n % 2 != 0) {
        DEBUG_PRINT("Error: n must be a multiple of 2.  n = %d", params.n);
        return false;
    }

    switch (params.inputType) {
        case nvinfer1::DataType::kFP8:
            if (params.k % kAlignmentInputFP8 != 0) {
                DEBUG_PRINT("Error: k must be a multiple of %d for FP8. k = %d", kAlignmentInputFP8, params.k);
                return false;
            }
            if (params.outputType == nvinfer1::DataType::kHALF) {
                dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, half>(params, stream);
            } else if (params.outputType == nvinfer1::DataType::kBF16) {
                dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, __nv_bfloat16>(params, stream);
            } else if (params.outputType == nvinfer1::DataType::kFLOAT) {
                dispatched = cudaCoreGemmLauncher<__nv_fp8_e4m3, float>(params, stream);
            } else {
                DEBUG_PRINT("Error: Unsupported output type for FP8: %d", params.outputType);
                return false;
            }
            break;

        case nvinfer1::DataType::kHALF:
            if (params.k % kAlignmentInputHalf != 0) {
                DEBUG_PRINT("Error: k must be a multiple of %d for HALF. k = %d", kAlignmentInputHalf, params.k);
                return false;
            }
            if (params.outputType == nvinfer1::DataType::kHALF) {
                dispatched = cudaCoreGemmLauncher<half, half>(params, stream);
            } else if (params.outputType == nvinfer1::DataType::kFLOAT) {
                dispatched = cudaCoreGemmLauncher<half, float>(params, stream);
            } else {
                DEBUG_PRINT("Error: Unsupported output type for HALF: %d", params.outputType);
                return false;
            }
            break;

        case nvinfer1::DataType::kBF16:
            if (params.k % kAlignmentInputBfloat16 != 0) {
                DEBUG_PRINT("Error: k must be a multiple of %d for BF16. k = %d", kAlignmentInputBfloat16, params.k);
                return false;
            }
            if (params.outputType == nvinfer1::DataType::kBF16) {
                dispatched = cudaCoreGemmLauncher<__nv_bfloat16, __nv_bfloat16>(params, stream);
            } else if (params.outputType == nvinfer1::DataType::kFLOAT) {
                dispatched = cudaCoreGemmLauncher<__nv_bfloat16, float>(params, stream);
            } else {
                DEBUG_PRINT("Error: Unsupported output type for BF16: %d", params.outputType);
                return false;
            }
            break;

        case nvinfer1::DataType::kFLOAT:
            if (params.k % kAlignmentInputFloat != 0) {
                DEBUG_PRINT("Error: k must be a multiple of %d for FLOAT. k = %d", kAlignmentInputFloat, params.k);
                return false;
            }
            if (params.outputType == nvinfer1::DataType::kFLOAT) {
                dispatched = cudaCoreGemmLauncher<float, float>(params, stream);
            } else {
                DEBUG_PRINT("Error: Unsupported output type for FLOAT: %d", params.outputType);
                return false;
            }
            break;

        default:
            DEBUG_PRINT("Error: Unsupported input type: %d", params.inputType);
            return false;
    }

    if (!dispatched) {
        DEBUG_PRINT(
            "sugesstify::kernels::cuda_core_gemm::cudaCoreGemmDispatcher failed to dispatch: inputType=%d, "
            "outputType=%d, "
            "m=%d, "
            "n=%d, k=%d",
            params.inputType, params.outputType, params.m, params.n, params.k);
    }
    return dispatched;
}

}
}
}