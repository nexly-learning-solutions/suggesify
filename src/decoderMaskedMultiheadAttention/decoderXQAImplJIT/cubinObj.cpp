
#include "cubinObj.h"

#include "serializationUtils.h"
#include "suggestify/common/assert.h"
#include "suggestify/common/cudaDriverWrapper.h"
#include "suggestify/common/cudaUtils.h"
#include <cuda_runtime_api.h>

namespace suggestify::kernels::jit
{

CubinObj::CubinObj(void const* buffer_, size_t buffer_size)
    : mInitialized(false)
{
    uint8_t const* buffer = static_cast<uint8_t const*>(buffer_);
    size_t remaining_buffer_size = buffer_size;
    uint32_t len = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);
    mContent.resize(len);
    TLLM_CHECK(len <= remaining_buffer_size);
    memcpy(mContent.data(), buffer, len);
}

CubinObj::CubinObj(std::string const& content)
    : mContent(content)
    , mInitialized(false)
{
}

CubinObj::CubinObj(CubinObj const& other)
{
    // Only uninitialized CubinObj can be copy-constructed.
    TLLM_CHECK(!other.mInitialized);

    this->mContent = other.mContent;
    this->mInitialized = false;
}

CubinObj& CubinObj::operator=(CubinObj const& other)
{
    if (this == &other)
    {
        return *this;
    }

    // Only uninitialized CubinObj can be copy-assigned.
    TLLM_CHECK(!other.mInitialized);

    this->mContent = other.mContent;
    this->mInitialized = false;

    return *this;
}

CubinObj::CubinObj(CubinObj&& other)
{
    this->mContent = std::move(other.mContent);
    if (other.mInitialized)
    {
        this->mInitialized = true;
        this->mDriver = std::move(other.mDriver);
        this->mModule = other.mModule;
        this->mFunction = other.mFunction;
        this->mSharedMemBytes = other.mSharedMemBytes;

        other.mInitialized = false;
    }
    else
    {
        this->mInitialized = false;
    }
}

CubinObj& CubinObj::operator=(CubinObj&& other)
{
    if (this == &other)
    {
        return *this;
    }

    this->mContent = std::move(other.mContent);
    if (other.mInitialized)
    {
        this->mInitialized = true;
        this->mDriver = std::move(other.mDriver);
        this->mModule = other.mModule;
        this->mFunction = other.mFunction;
        this->mSharedMemBytes = other.mSharedMemBytes;

        other.mInitialized = false;
    }
    else
    {
        this->mInitialized = false;
    }
    return *this;
}

size_t CubinObj::getSerializationSize() const noexcept
{
    size_t result = sizeof(uint32_t) + mContent.size();
    // Make result multiples of 4.
    result = (result + 3) & ~3;
    return result;
}

void CubinObj::serialize(void* buffer_, size_t buffer_size) const noexcept
{
    size_t remaining_buffer_size = buffer_size;
    uint8_t* buffer = static_cast<uint8_t*>(buffer_);
    uint32_t len = mContent.size();
    writeToBuffer<uint32_t>(len, buffer, remaining_buffer_size);
    TLLM_CHECK(len <= remaining_buffer_size);
    memcpy(buffer, mContent.c_str(), len);
}

void CubinObj::launch(dim3 gridDim, dim3 blockDim, CUstream hStream, void** kernelParams)
{
    TLLM_CHECK(mInitialized);
    TLLM_CU_CHECK(mDriver->cuLaunchKernel(mFunction, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
        blockDim.z, mSharedMemBytes, hStream, kernelParams, /*extra=*/nullptr));
}

void CubinObj::initialize()
{
    if (!mInitialized)
    {
        mDriver = suggestify::common::CUDADriverWrapper::getInstance();
        mModule = nullptr;
        TLLM_CU_CHECK(mDriver->cuModuleLoadData(&mModule, mContent.c_str()));
        TLLM_CHECK(mModule != nullptr);
        mFunction = nullptr;
        TLLM_CU_CHECK(mDriver->cuModuleGetFunction(&mFunction, mModule, kFuncName));
        TLLM_CHECK(mFunction != nullptr);

        // Populate mSharedMemBytes.
        CUdeviceptr shmem_dev_ptr = 0;
        TLLM_CU_CHECK(mDriver->cuModuleGetGlobal(&shmem_dev_ptr, nullptr, mModule, kSmemName));
        TLLM_CHECK(shmem_dev_ptr != 0);
        TLLM_CU_CHECK(mDriver->cuMemcpyDtoH(&mSharedMemBytes, shmem_dev_ptr, sizeof(unsigned int)));

        TLLM_CHECK(mSharedMemBytes > 0);

        /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */
        if (mSharedMemBytes >= 46 * 1024)
        {
            TLLM_CU_CHECK(mDriver->cuFuncSetAttribute(
                mFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, mSharedMemBytes));
        }

        sync_check_cuda_error();
        mInitialized = true;
    }
}

CubinObj::~CubinObj()
{
    if (mInitialized)
    {
        TLLM_CU_CHECK(mDriver->cuModuleUnload(mModule));
        mInitialized = false;
    }
}

} // namespace suggestify::kernels::jit
