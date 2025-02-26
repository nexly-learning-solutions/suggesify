#pragma once

#include "NvInferPlugin.h"
#include "../plugins/common/plugin.h"
#include "suggestify/runtime/cudaMemPool.h"
#include "suggestify/runtime/utils/debugUtils.h"
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace pluginInternal
{
class SideWorkspace
{
public:
    SideWorkspace(cudaStream_t stream)
        : mWorkspaceSize{0}
        , mWorkspacePtr{nullptr}
        , mStream{stream}
    {
    }

    ~SideWorkspace()
    {
        if (mWorkspacePtr)
        {
            TLLM_CUDA_CHECK(cudaFreeAsync(mWorkspacePtr, mStream));
        }
    }

    void* get(size_t workspaceSize)
    {
        if (mWorkspacePtr && mWorkspaceSize < workspaceSize)
        {
            TLLM_CUDA_CHECK(cudaFreeAsync(mWorkspacePtr, mStream));
            mWorkspacePtr = nullptr;
        }
        if (!mWorkspacePtr)
        {
            mWorkspaceSize = workspaceSize;
            auto pool_ptr
                = suggestify::runtime::CudaMemPool::getPrimaryPoolForDevice(suggestify::common::getDevice());
            TLLM_CUDA_CHECK(cudaMallocFromPoolAsync(&mWorkspacePtr, mWorkspaceSize, pool_ptr->getPool(), mStream));
        }
        return mWorkspacePtr;
    }

private:
    size_t mWorkspaceSize;
    void* mWorkspacePtr;
    cudaStream_t mStream;
};

class SideStream : public IPluginResource
{
public:
    SideStream(bool init = false)
        : mStream{}
        , mMainEvent{}
        , mSideEvent{}
        , mWorkspace{}
        , mInit{init}
    {
        if (init)
        {
            TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));
            TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&mMainEvent, cudaEventDisableTiming));
            TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&mSideEvent, cudaEventDisableTiming));
            mWorkspace = std::make_shared<SideWorkspace>(mStream);
        }
    }

    void free()
    {
        if (mInit)
        {
            mWorkspace = nullptr;
            TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
            TLLM_CUDA_CHECK(cudaStreamDestroy(mStream));
            TLLM_CUDA_CHECK(cudaEventDestroy(mMainEvent));
            TLLM_CUDA_CHECK(cudaEventDestroy(mSideEvent));
            mInit = false;
        }
    }

    int32_t release() noexcept override
    {
        try
        {
            free();
        }
        catch (std::exception const& e)
        {
            return -1;
        }
        return 0;
    }

    IPluginResource* clone() noexcept override
    {
        std::unique_ptr<SideStream> cloned{};
        try
        {
            if (!mInit)
            {
                cloned = std::make_unique<SideStream>( true);
            }
            else
            {
                return nullptr;
            }
        }
        catch (std::exception const& e)
        {
            return nullptr;
        }
        return cloned.release();
    }

    ~SideStream() override
    {
        free();
    }

    void* getWorkspacePtr(size_t workspaceSize)
    {
        return mWorkspace->get(workspaceSize);
    }

    cudaStream_t getStream() const
    {
        return mStream;
    }

    void waitMainStreamOnSideStream(cudaStream_t const stream) const
    {
        TLLM_CUDA_CHECK(cudaEventRecord(mMainEvent, stream));
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(mStream, mMainEvent));
    }

    void waitSideStreamOnMainStream(cudaStream_t const stream) const
    {
        TLLM_CUDA_CHECK(cudaEventRecord(mSideEvent, mStream));
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(stream, mSideEvent));
    }

    void stallMainStream(char const* name, cudaStream_t const stream, std::optional<int> delay = std::nullopt) const
    {
        suggestify::runtime::utils::stallStream(name, stream, delay);
    }

    void stallSideStream(char const* name, std::optional<int> delay = std::nullopt) const
    {
        suggestify::runtime::utils::stallStream(name, mStream, delay);
    }

    static char const* getResourceKey(int const stream_id)
    {
        std::string keyString = "side_stream_" + std::to_string(stream_id);
        return keyString.c_str();
    }

private:
    cudaStream_t mStream;
    cudaEvent_t mMainEvent;
    cudaEvent_t mSideEvent;
    std::shared_ptr<SideWorkspace> mWorkspace;
    bool mInit;
};

}
}

namespace suggestify::plugins
{

class CudaStreamPlugin : public BasePlugin
{
public:
    CudaStreamPlugin(int sideStreamId, int nbInputs, nvinfer1::DataType type);

    CudaStreamPlugin(void const* data, size_t length);

    CudaStreamPlugin(CudaStreamPlugin const&);

    void init();

    ~CudaStreamPlugin() override = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    const std::string mLayerName;
    int mSideStreamId;
    int mNbInputs;
    nvinfer1::DataType mType;
    nvinfer1::pluginInternal::SideStream* mSideStreamPtr;
};

class CudaStreamPluginCreator : public BaseCreator
{
public:
    CudaStreamPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}
