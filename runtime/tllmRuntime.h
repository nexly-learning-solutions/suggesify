#pragma once

#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"
#include "layerProfiler.h"
#include "rawEngine.h"
#include "worldConfig.h"
#include <NvInferRuntime.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace suggestify::runtime
{
class TllmRuntime
{
public:
    using TensorMap = StringPtrMap<ITensor>;

    explicit TllmRuntime(RawEngine const& rawEngine, nvinfer1::ILogger* logger, float gpuWeightsPercent = 1.0f,
        bool useShapeInference = true);

    SizeType32 getNbContexts() const
    {
        return static_cast<SizeType32>(mContexts.size());
    }

    nvinfer1::IExecutionContext& getContext(SizeType32 contextIndex) const
    {
        return *mContexts.at(contextIndex);
    }

    SizeType32 getNbProfiles() const
    {
        return static_cast<SizeType32>(mEngine->getNbOptimizationProfiles());
    }

    [[nodiscard]] SizeType32 getOptProfileId(int numTokens, std::vector<SizeType32> const& splitPoints) const
    {
        if (getNbProfiles() == 1)
        {
            return 0;
        }
        auto const it = std::lower_bound(splitPoints.begin(), splitPoints.end(), numTokens);
        auto const optProfileId = std::distance(splitPoints.begin(), it);
        return optProfileId;
    }

    nvinfer1::IExecutionContext& addContext(std::int32_t profileIndex);

    void clearContexts();

    void setStaticInputTensors(TensorMap const& tensorMap);

    void setInputTensors(SizeType32 contextIndex, TensorMap const& tensorMap);

    void setOutputTensors(SizeType32 contextIndex, TensorMap& tensorMap);

    bool executeContext(SizeType32 contextIndex) const;

    CudaStream const& getStream() const;

    BufferManager::CudaStreamPtr getStreamPtr()
    {
        return mStream;
    }

    nvinfer1::ICudaEngine& getEngine()
    {
        return *mEngine;
    }

    nvinfer1::ICudaEngine const& getEngine() const
    {
        return *mEngine;
    }

    nvinfer1::IEngineInspector& getEngineInspector()
    {
        return *mEngineInspector;
    }

    nvinfer1::IEngineInspector const& getEngineInspector() const
    {
        return *mEngineInspector;
    }

    BufferManager& getBufferManager()
    {
        return mBufferManager;
    }

    BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

    void setLayerProfiler();
    bool hasLayerProfiler(SizeType32 contextId) const;
    std::string getLayerProfileInfo() const;
    void reportToProfiler(SizeType32 contextId);
    void loadManagedWeights(RawEngine const& rawEngine, int localRank);
    void printEngineInfo();
    void initializeUserBuffer(SizeType32 tpSize, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxSequenceLength, SizeType32 hiddenSize, std::optional<SizeType32> maxNumTokens);

    bool isUserBufferEnabled() const
    {
        return mUserBufferEnabled;
    }

private:
    void cacheTensorNames();

    void setInputTensorsImpl(SizeType32 contextIndex, TensorMap const& tensorMap, bool throwOnMiss);

    void setUserBufferTensors(SizeType32 contextIndex, TensorMap& tensorMap);

    static std::string shapeToString(nvinfer1::Dims64 const& dim)
    {
        std::string output("(");
        if (dim.nbDims == 0)
        {
            return output + ")";
        }
        for (int i = 0; i < dim.nbDims - 1; ++i)
        {
            output += std::to_string(dim.d[i]) + ", ";
        }
        output += std::to_string(dim.d[dim.nbDims - 1]) + ")";
        return output;
    }

    static std::string dataTypeToString(nvinfer1::DataType type)
    {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#endif
        switch (type)
        {
        case nvinfer1::DataType::kINT64: return "INT64";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kFLOAT: return "FP32";
        case nvinfer1::DataType::kBF16: return "BF16";
        case nvinfer1::DataType::kHALF: return "FP16";
        case nvinfer1::DataType::kBOOL: return "BOOL";
        case nvinfer1::DataType::kUINT8: return "UINT8";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kFP8: return "FP8";
        case nvinfer1::DataType::kINT4: return "INT4";
        default: return "UNKNOWN";
        }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        return "";
    }

    static std::string alignText(
        std::string const& text, int const width, bool const bCenter = true, char const blank = ' ')
    {
        int textLen = text.size();
        int padLeft = 0;
        int padRight = 0;
        padLeft = bCenter ? (width - textLen) / 2 : 0;
        padRight = width - padLeft - textLen;
        return std::string(padLeft, blank) + text + std::string(padRight, blank);
    }

    BufferManager::CudaStreamPtr mStream;
    BufferManager mBufferManager;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    BufferManager::IBufferPtr mEngineBuffer;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
    std::unique_ptr<ITensor> mDummyTensor;
    std::unique_ptr<nvinfer1::IEngineInspector> mEngineInspector;
    std::unique_ptr<LayerProfiler> mLayerProfiler;
    bool mUseShapeInference;
    TensorMap mManagedWeightsMap;
    std::vector<std::string> mInputTensorNames;
    std::vector<std::string> mOutputTensorNames;

    bool mUserBufferEnabled;
};
}
