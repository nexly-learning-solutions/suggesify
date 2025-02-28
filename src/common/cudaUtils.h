#pragma once

#include "cudaBf16Wrapper.h"
#include "cudaDriverWrapper.h"
#include "cudaFp8Utils.h"
#include "logger.h"
#include "tllmException.h"
#include <algorithm>
#include <cinttypes>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#ifndef _WIN32
#include <sys/sysinfo.h>
#endif
#include <vector>
#ifdef _WIN32
#include <windows.h>
#undef ERROR
#endif

namespace suggestify::common
{

#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4)
{
    half x, y, z, w;
}

half4;


enum CublasDataType
{
    FLOAT_DATATYPE = 0,
    HALF_DATATYPE = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE = 3,
    FP8_DATATYPE = 4
};

enum TRTLLMCudaDataType
{
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8 = 4
};

enum class OperationType
{
    FP32,
    FP16,
    BF16,
    INT8,
    FP8
};

static char const* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static char const* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template <typename T>
void check(T result, char const* const func, char const* const file, int const line)
{
    if (result)
    {
        throw TllmException(
            file, line, fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(result)));
    }
}

template <typename T>
void checkEx(T result, std::initializer_list<T> const& validReturns, char const* const func, char const* const file,
    int const line)
{
    if (std::all_of(std::begin(validReturns), std::end(validReturns), [&result](T const& t) { return t != result; }))
    {
        throw TllmException(
            file, line, fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(result)));
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline std::optional<bool> isCudaLaunchBlocking()
{
    static bool firstCall = true;
    static std::optional<bool> result = std::nullopt;

    if (firstCall)
    {
        char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
        if (env != nullptr && std::string(env) == "1")
        {
            result = true;
        }
        else if (env != nullptr && std::string(env) == "0")
        {
            result = false;
        }
        firstCall = false;
    }

    return result;
}

inline bool doCheckError()
{
    auto const cudaLaunchBlocking = isCudaLaunchBlocking();
#ifndef NDEBUG
    bool const checkError = cudaLaunchBlocking.value_or(true);
#else
    bool const checkError = cudaLaunchBlocking.value_or(false);
#endif

    return checkError;
}

inline void syncAndCheck(char const* const file, int const line)
{
    if (doCheckError())
    {
        cudaDeviceSynchronize();
        check(cudaGetLastError(), "cudaGetLastError", file, line);
    }
}

#define sync_check_cuda_error() suggestify::common::syncAndCheck(__FILE__, __LINE__)

#define PRINT_FUNC_NAME_()                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cout << "[TensorRT-LLM][CALL] " << __FUNCTION__ << " " << std::endl;                                      \
    } while (0)

template<typename T> struct packed_type;
template <>          struct packed_type<float>         { using type = float; };
template <>          struct packed_type<half>          { using type = half2; };

#ifdef ENABLE_BF16
template<>
struct packed_type<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

#ifdef ENABLE_FP8
template<>
struct packed_type<__nv_fp8_e4m3> {
    using type = __nv_fp8x2_e4m3;
};
#endif

template<typename T> struct num_elems;
template <>          struct num_elems<float>           { static constexpr int value = 1; };
template <>          struct num_elems<float2>          { static constexpr int value = 2; };
template <>          struct num_elems<float4>          { static constexpr int value = 4; };
template <>          struct num_elems<half>            { static constexpr int value = 1; };
template <>          struct num_elems<half2>           { static constexpr int value = 2; };
#ifdef ENABLE_BF16
template <>          struct num_elems<__nv_bfloat16>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_bfloat162>  { static constexpr int value = 2; };
#endif
#ifdef ENABLE_FP8
template <>          struct num_elems<__nv_fp8_e4m3>   { static constexpr int value = 1; };
template <>          struct num_elems<__nv_fp8x2_e4m3>  { static constexpr int value = 2; };
#endif

template<typename T, int num> struct packed_as;
template<typename T>          struct packed_as<T, 1>              { using type = T; };
template<>                    struct packed_as<half,  2>          { using type = half2; };
template<>                    struct packed_as<float,  2>         { using type = float2; };
template<>                    struct packed_as<int8_t, 2>         { using type = int16_t; };
template<>                    struct packed_as<int32_t, 2>        { using type = int2; };
template<>                    struct packed_as<half2, 1>          { using type = half; };
template<>                    struct packed_as<float2, 1>         { using type = float; };
#ifdef ENABLE_BF16
template<> struct packed_as<__nv_bfloat16,  2> { using type = __nv_bfloat162; };
template<> struct packed_as<__nv_bfloat162, 1> { using type = __nv_bfloat16;  };
#endif
#ifdef ENABLE_FP8
template<> struct packed_as<__nv_fp8_e4m3,  2> { using type = __nv_fp8x2_e4m3; };
template<> struct packed_as<__nv_fp8x2_e4m3, 1> { using type = __nv_fp8_e4m3;  };
template<> struct packed_as<__nv_fp8_e5m2,  2> { using type = __nv_fp8x2_e5m2; };
template<> struct packed_as<__nv_fp8x2_e5m2, 1> { using type = __nv_fp8_e5m2;  };
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __device__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }

inline __device__ float2 operator*(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
inline __device__ float2 operator+(float2 a, float  b) { return make_float2(a.x + b, a.y + b); }
inline __device__ float2 operator-(float2 a, float  b) { return make_float2(a.x - b, a.y - b); }


template <typename T>
struct CudaDataType
{
};

template <>
struct CudaDataType<float>
{
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_32F;
};

template <>
struct CudaDataType<half>
{
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16F;
};

#ifdef ENABLE_BF16
template <>
struct CudaDataType<__nv_bfloat16>
{
    static constexpr cudaDataType_t value = cudaDataType::CUDA_R_16BF;
};
#endif

inline int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline int getDevice()
{
    int current_dev_id = 0;
    check_cuda_error(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}

inline int getDeviceCount()
{
    int count = 0;
    check_cuda_error(cudaGetDeviceCount(&count));
    return count;
}

template <typename T>
cudaMemoryType getPtrCudaMemoryType(T* ptr)
{
    cudaPointerAttributes attributes{};
    check_cuda_error(cudaPointerGetAttributes(&attributes, ptr));
    return attributes.type;
}

inline std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm)
{
    if (useUvm)
    {
        size_t freeSysMem = 0;
        size_t totalSysMem = 0;
#ifndef _WIN32
        struct sysinfo info
        {
        };

        sysinfo(&info);
        totalSysMem = info.totalram * info.mem_unit;
        freeSysMem = info.freeram * info.mem_unit;
#else
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(memInfo);
        GlobalMemoryStatusEx(&memInfo);
        totalSysMem = memInfo.ullTotalPhys;
        freeSysMem = memInfo.ullAvailPhys;
#endif

        LOG_INFO("Using UVM based system memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
            ((double) totalSysMem / 1e9), ((double) freeSysMem / 1e9));
        return {freeSysMem, totalSysMem};
    }

    size_t free = 0;
    size_t total = 0;
    check_cuda_error(cudaMemGetInfo(&free, &total));
    LOG_DEBUG("Using GPU memory for KV cache, total memory %0.2f GB, available memory %0.2f GB",
        ((double) total / 1e9), ((double) free / 1e9));
    return {free, total};
}

inline size_t getAllocationGranularity()
{
    auto const currentDevice = getDevice();
    ::CUmemAllocationProp prop = {};

    prop.type = ::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = currentDevice;
    prop.requestedHandleTypes = ::CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return granularity;
}

inline int getMultiProcessorCount()
{
    int device_id = 0;
    int multi_processor_count = 0;
    check_cuda_error(cudaGetDevice(&device_id));
    check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id));
    return multi_processor_count;
}

inline int getMaxSharedMemoryPerBlockOptin()
{
    int device_id = 0;
    int max_shared_memory_per_block = 0;
    check_cuda_error(cudaGetDevice(&device_id));
    check_cuda_error(
        cudaDeviceGetAttribute(&max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
    return max_shared_memory_per_block;
}

template <typename T1, typename T2>
inline size_t divUp(const T1& a, const T2& n)
{
    auto const tmp_a = static_cast<size_t>(a);
    auto const tmp_n = static_cast<size_t>(n);
    return (tmp_a + tmp_n - 1) / tmp_n;
}

inline int roundUp(int a, int n)
{
    return divUp(a, n) * n;
}

template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
    typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <typename T>
void printAbsMean(T const* buf, uint64_t size, cudaStream_t stream, std::string name = "")
{
    if (buf == nullptr)
    {
        LOG_WARNING("%s is an nullptr, skip!", name.c_str());
        return;
    }
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    T* h_tmp = new T[size];
    cudaMemcpyAsync(h_tmp, buf, sizeof(T) * size, cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    double sum = 0.0f;
    uint64_t zero_count = 0;
    float max_val = -1e10;
    bool find_inf = false;
    for (uint64_t i = 0; i < size; i++)
    {
        if (std::isinf((float) (h_tmp[i])))
        {
            find_inf = true;
            continue;
        }
        sum += abs((double) h_tmp[i]);
        if ((float) h_tmp[i] == 0.0f)
        {
            zero_count++;
        }
        max_val = max_val > abs(float(h_tmp[i])) ? max_val : abs(float(h_tmp[i]));
    }
    LOG_INFO("%20s size: %u, abs mean: %f, abs sum: %f, abs max: %f, find inf: %s", name.c_str(), size, sum / size,
        sum, max_val, find_inf ? "true" : "false");
    delete[] h_tmp;
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
}

template <typename T>
void printToStream(T const* result, int const size, FILE* strm)
{
    bool const split_rows = (strm == stdout);
    if (result == nullptr)
    {
        LOG_WARNING("It is an nullptr, skip! \n");
        return;
    }
    T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * size));
    check_cuda_error(cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i)
    {
        fprintf(strm, "%f, ", static_cast<float>(tmp[i]));
        if (split_rows && ((i + 1) % 10) == 0)
            fprintf(strm, "\n");
    }
    if (!split_rows || (size % 10) != 0)
    {
        fprintf(strm, "\n");
    }
    free(tmp);
}

template <typename T>
void printToScreen(T const* result, int const size)
{
    printToStream(result, size, stdout);
}

template <typename T>
void print2dToStream(T const* result, int const r, int const c, int const stride, FILE* strm)
{
    if (result == nullptr)
    {
        LOG_WARNING("It is an nullptr, skip! \n");
        return;
    }
    for (int ri = 0; ri < r; ++ri)
    {
        T const* ptr = result + ri * stride;
        printToStream(ptr, c, strm);
    }
    fprintf(strm, "\n");
}

template <typename T>
void print2dToScreen(T const* result, int const r, int const c, int const stride)
{
    print2dToStream(result, r, c, stride, stdout);
}

template <typename T>
void print2dToFile(std::string fname, T const* result, int const r, int const c, int const stride)
{
    FILE* fp = fopen(fname.c_str(), "wt");
    if (fp != nullptr)
    {
        print2dToStream(result, r, c, stride, fp);
        fclose(fp);
    }
}

inline void print_float_(float x)
{
    printf("%7.3f ", x);
}

inline void print_element_(float x)
{
    print_float_(x);
}

inline void print_element_(half x)
{
    print_float_((float) x);
}

#ifdef ENABLE_BF16
inline void print_element_(__nv_bfloat16 x)
{
    print_float_((float) x);
}
#endif

#ifdef ENABLE_FP8
inline void print_element_(__nv_fp8_e4m3 x)
{
    print_float_((float) x);
}
#endif

inline void print_element_(uint32_t ul)
{
    printf("%7" PRIu32, ul);
}

inline void print_element_(uint64_t ull)
{
    printf("%7" PRIu64, ull);
}

inline void print_element_(int32_t il)
{
    printf("%7" PRId32, il);
}

inline void print_element_(int64_t ill)
{
    printf("%7" PRId64, ill);
}

template <typename T>
inline void printMatrix(T const* ptr, int m, int k, int stride, bool is_device_ptr)
{
    T* tmp;
    if (is_device_ptr)
    {
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_cuda_error(cudaMemcpy(tmp, ptr, sizeof(T) * m * stride, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    else
    {
        tmp = const_cast<T*>(ptr);
    }

    for (int ii = -1; ii < m; ++ii)
    {
        if (ii >= 0)
        {
            printf("%07d ", ii);
        }
        else
        {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1)
        {
            if (ii >= 0)
            {
                print_element_(tmp[ii * stride + jj]);
            }
            else
            {
                printf("%7d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr)
    {
        free(tmp);
    }
}

template void printMatrix(float const* ptr, int m, int k, int stride, bool is_device_ptr);
template void printMatrix(half const* ptr, int m, int k, int stride, bool is_device_ptr);
#ifdef ENABLE_BF16
template void printMatrix(__nv_bfloat16 const* ptr, int m, int k, int stride, bool is_device_ptr);
#endif
#ifdef ENABLE_FP8
template void printMatrix(__nv_fp8_e4m3 const* ptr, int m, int k, int stride, bool is_device_ptr);
#endif
template void printMatrix(uint32_t const* ptr, int m, int k, int stride, bool is_device_ptr);
template void printMatrix(uint64_t const* ptr, int m, int k, int stride, bool is_device_ptr);
template void printMatrix(int const* ptr, int m, int k, int stride, bool is_device_ptr);

}

#define CUDA_CHECK(stat)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        suggestify::common::check((stat), #stat, __FILE__, __LINE__);                                                \
    } while (0)

#define CUDA_CHECK_FREE_RESOURCE(stat)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        suggestify::common::checkEx((stat), {cudaSuccess, cudaErrorCudartUnloading}, #stat, __FILE__, __LINE__);     \
    } while (0)
