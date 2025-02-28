
#pragma once

#include "common.h"
#include "modelConfig.h"
#include "runtimeDefaults.h"
#include "worldConfig.h"

#include <filesystem>
#include <istream>
#include <string>
#include <utility>

namespace suggestify::runtime
{

class GptJsonConfig
{
public:
    GptJsonConfig(std::string name, std::string version, std::string precision, SizeType32 tensorParallelism,
        SizeType32 pipelineParallelism, SizeType32 contextParallelism, SizeType32 gpusPerNode, ModelConfig modelConfig,
        std::optional<RuntimeDefaults> runtimeDefaults = std::nullopt)
        : mName(std::move(name))
        , mVersion(std::move(version))
        , mPrecision(std::move(precision))
        , mTensorParallelism{tensorParallelism}
        , mPipelineParallelism{pipelineParallelism}
        , mContextParallelism{contextParallelism}
        , mGpusPerNode{gpusPerNode}
        , mModelConfig(std::move(modelConfig))
        , mRuntimeDefaults(std::move(runtimeDefaults))
    {
    }

    static GptJsonConfig parse(std::string const& json);

    static GptJsonConfig parse(std::istream& json);

    static GptJsonConfig parse(std::filesystem::path const& path);

    [[nodiscard]] ModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] ModelConfig& getModelConfigMutable()
    {
        return mModelConfig;
    }

    [[nodiscard]] std::string const& getName() const
    {
        return mName;
    }

    [[nodiscard]] std::string const& getVersion() const
    {
        return mVersion;
    }

    [[nodiscard]] std::string const& getPrecision() const
    {
        return mPrecision;
    }

    [[nodiscard]] SizeType32 constexpr getTensorParallelism() const
    {
        return mTensorParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getPipelineParallelism() const
    {
        return mPipelineParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getContextParallelism() const
    {
        return mContextParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getGpusPerNode() const
    {
        return mGpusPerNode;
    }

    [[nodiscard]] SizeType32 constexpr getWorldSize() const
    {
        return mTensorParallelism * mPipelineParallelism * mContextParallelism;
    }

    [[nodiscard]] std::optional<RuntimeDefaults> getRuntimeDefaults() const
    {
        return mRuntimeDefaults;
    }

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig, std::string const& model) const;

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig) const
    {
        return engineFilename(worldConfig, getName());
    }

private:
    std::string const mName;
    std::string const mVersion;
    std::string const mPrecision;
    SizeType32 const mTensorParallelism;
    SizeType32 const mPipelineParallelism;
    SizeType32 const mContextParallelism;
    SizeType32 const mGpusPerNode;
    ModelConfig mModelConfig;
    std::optional<RuntimeDefaults> mRuntimeDefaults;
};

}
