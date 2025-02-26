
#pragma once

#include "../common/assert.h"
#include "../executor/tensor.h"

#include <NvInferRuntime.h>
#include <filesystem>
#include <map>
#include <optional>

namespace suggestify::runtime
{

class RawEngine
{
public:
    enum Type
    {
        FilePath,
        AddressWithSize,
        HostMemory
    };

    explicit RawEngine(std::filesystem::path enginePath) noexcept
        : mType(FilePath)
        , mEnginePath(std::move(enginePath))
    {
    }

    explicit RawEngine(void const* engineAddr, std::size_t engineSize) noexcept
        : mType(AddressWithSize)
        , mEngineAddr(engineAddr)
        , mEngineSize(engineSize)
    {
    }

    explicit RawEngine(nvinfer1::IHostMemory const* engineBuffer) noexcept
        : mType(HostMemory)
        , mEngineBuffer(engineBuffer)
    {
    }

    [[nodiscard]] Type getType() const
    {
        return mType;
    }

    [[nodiscard]] std::filesystem::path getPath() const
    {
        CHECK(mEnginePath.has_value());
        return mEnginePath.value();
    }

    [[nodiscard]] std::optional<std::filesystem::path> getPathOpt() const
    {
        return mEnginePath;
    }

    void setPath(std::filesystem::path enginePath)
    {
        mEnginePath = std::move(enginePath);
    }

    [[nodiscard]] std::optional<std::map<std::string, suggestify::executor::Tensor>> const&
    getManagedWeightsMapOpt() const
    {
        return mManagedWeightsMap;
    }

    void setManagedWeightsMap(std::map<std::string, suggestify::executor::Tensor> managedWeightsMap)
    {
        mManagedWeightsMap = std::move(managedWeightsMap);
    }

    [[nodiscard]] void const* getAddress() const
    {
        CHECK(mType == AddressWithSize);
        return mEngineAddr;
    }

    [[nodiscard]] std::size_t getSize() const
    {
        CHECK(mType == AddressWithSize);
        return mEngineSize;
    }

    [[nodiscard]] nvinfer1::IHostMemory const* getHostMemory() const
    {
        CHECK(mType == HostMemory);
        return mEngineBuffer;
    }

private:
    Type mType;
    std::optional<std::filesystem::path> mEnginePath;

    struct
    {
        void const* mEngineAddr{};
        std::size_t mEngineSize{};
    };

    nvinfer1::IHostMemory const* mEngineBuffer{};
    std::optional<std::map<std::string, suggestify::executor::Tensor>> mManagedWeightsMap;
};

}
