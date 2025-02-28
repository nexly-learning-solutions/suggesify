
#pragma once
#include <cuda_runtime_api.h>
#include <string>

#include "cudaDriverWrapper.h"

namespace suggestify
{
namespace kernels
{
namespace jit
{

class CubinObj
{
public:
    // Default constructor constructs an empty unusable CubinObj instance.
    CubinObj() = default;
    // Constructs from raw cubin content.
    explicit CubinObj(std::string const& content);
    // Deserializes from a serialization buffer.
    CubinObj(void const* buffer, size_t buffer_size);

    CubinObj(CubinObj const& other);
    CubinObj& operator=(CubinObj const& other);

    // CubinObj can be move-constructed/assigned.
    CubinObj(CubinObj&& other);
    CubinObj& operator=(CubinObj&& other);
    ~CubinObj();

    // Should be called at least once before calling launch().
    void initialize();
    void launch(dim3 gridDim, dim3 blockDim, CUstream hStream, void** kernelParams);

    // It is safe to call getSerializeSize()/serialize() before calling initialize().
    size_t getSerializationSize() const noexcept;
    void serialize(void* buffer, size_t buffer_size) const noexcept;

    bool isInitialized() const
    {
        return mInitialized;
    }

private:
    static constexpr char const* kFuncName = "kernel_mha";
    static constexpr char const* kSmemName = "smemSize";
    // Constructors should populate mContent.
    std::string mContent;

    // Fields below are undefined prior to initialize() call.
    bool mInitialized;
    std::shared_ptr<suggestify::common::CUDADriverWrapper> mDriver;
    CUmodule mModule;
    CUfunction mFunction;
    unsigned int mSharedMemBytes;
};

} // namespace jit
} // namespace kernels
} // namespace suggestify
