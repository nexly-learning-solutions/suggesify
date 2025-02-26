
#pragma once

#include "../common/cublasMMWrapper.h"
#include "../common/workspace.h"
#include "../plugins/api/tllmPlugin.h"
#include "../plugins/common/checkMacrosPlugin.h"

#include <NvInferRuntime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

#include <cstring>
#include <map>
#include <memory>
#include <nvml.h>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

namespace suggestify::plugins
{

class BasePlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BasePluginV3 : public nvinfer1::IPluginV3,
                     public nvinfer1::IPluginV3OneCore,
                     public nvinfer1::IPluginV3OneBuild,
                     public nvinfer1::IPluginV3OneRuntime
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BaseCreator : public nvinfer1::IPluginCreator
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BaseCreatorV3 : public nvinfer1::IPluginCreatorV3One
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

template <typename T>
void write(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

template <typename T>
void read(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

inline size_t typeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBOOL: return 1UL;
    case nvinfer1::DataType::kFP8: return 1UL;
    case nvinfer1::DataType::kHALF: return 2UL;
    case nvinfer1::DataType::kBF16: return 2UL;
    case nvinfer1::DataType::kFLOAT: return 4UL;
    case nvinfer1::DataType::kINT8: return 1UL;
    case nvinfer1::DataType::kUINT8: return 1UL;
    case nvinfer1::DataType::kINT32: return 4UL;
    case nvinfer1::DataType::kINT64: return 8UL;
    }

    THROW("Unknown dtype %d", static_cast<int>(type));
    return 0;
}

inline cudaDataType_t trtToCublasDtype(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return CUDA_R_32F;
    case nvinfer1::DataType::kHALF: return CUDA_R_16F;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kBF16: return CUDA_R_16BF;
#endif
    default: THROW("Not supported data type for cuBLAS");
    }
}

template <typename T, typename Del = std::default_delete<T>>
class UniqPtrWNullCopy : public std::unique_ptr<T, Del>
{
public:
    using std::unique_ptr<T, Del>::unique_ptr;

    explicit UniqPtrWNullCopy(std::unique_ptr<T, Del>&& src)
        : std::unique_ptr<T, Del>::unique_ptr{std::move(src)}
    {
    }

    UniqPtrWNullCopy(UniqPtrWNullCopy const&)
        : std::unique_ptr<T, Del>::unique_ptr{}
    {
    }
};

void const* getCommSessionHandle();
}

inline bool isBuilding()
{
    auto constexpr key = "IS_BUILDING";
    auto const val = getenv(key);
    return val != nullptr && std::string(val) == "1";
}

#if ENABLE_MULTI_DEVICE
#define NCCLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess)                                                                                          \
        {                                                                                                              \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap();

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group);

#endif

std::shared_ptr<cublasHandle_t> getCublasHandle();
std::shared_ptr<cublasLtHandle_t> getCublasLtHandle();
std::shared_ptr<suggestify::common::CublasMMWrapper> getCublasMMWrapper(std::shared_ptr<cublasHandle_t> cublasHandle,
    std::shared_ptr<cublasLtHandle_t> cublasltHandle, cudaStream_t stream, void* workspace);

#ifndef DEBUG

#define PLUGIN_CHECK(status)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
            abort();                                                                                                   \
    } while (0)

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_BAD_PARAM;                                                                                   \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_FAILURE;                                                                                     \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)

#else

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__);                                      \
            return STATUS_BAD_PARAM;                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__);                                        \
            return STATUS_FAILURE;                                                                                     \
        }                                                                                                              \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));                        \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CHECK(status)                                                                                           \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status));                      \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#endif

class PluginFieldParser
{
public:
    PluginFieldParser(int32_t nbFields, nvinfer1::PluginField const* fields);
    PluginFieldParser(PluginFieldParser const&) = delete;
    PluginFieldParser& operator=(PluginFieldParser const&) = delete;
    ~PluginFieldParser();
    template <typename T>
    std::optional<T> getScalar(std::string_view const& name);
    template <typename T>
    std::optional<std::set<T>> getSet(std::string_view const& name);

private:
    nvinfer1::PluginField const* mFields;

    struct Record
    {
        Record(int32_t idx)
            : index{idx}
        {
        }

        int32_t const index;
        bool retrieved{false};
    };

    std::unordered_map<std::string_view, Record> mMap;
};

#define NVML_CHECK(cmd)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        nvmlReturn_t r = cmd;                                                                                          \
        if (r != NVML_SUCCESS)                                                                                         \
        {                                                                                                              \
            printf("Failed, NVML error %s:%d '%s'\n", __FILE__, __LINE__, nvmlErrorString(r));                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
