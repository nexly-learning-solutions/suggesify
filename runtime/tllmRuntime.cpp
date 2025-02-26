#include "tllmRuntime.h"
#include "common.h"
#include "nlohmann/json.hpp"
#include "suggestify/common/assert.h"
#include "suggestify/common/logger.h"
#include "suggestify/common/mpiUtils.h"
#include "suggestify/common/nvtxUtils.h"
#include "suggestify/common/safetensors.h"
#include "suggestify/executor/tensor.h"
#include "../src/userbuffers/ub_interface.h"
#include "tllmLogger.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace suggestify::runtime;
using TensorMap = StringPtrMap<ITensor>;

namespace
{
static_assert(std::is_signed<SizeType32>::value, "SizeType32 must be signed");

nvinfer1::Dims shapeToDims(std::vector<std::size_t> const& shape)
{
    CHECK(shape.size() <= nvinfer1::Dims::MAX_DIMS);
    nvinfer1::Dims dims;
    auto constexpr dim_max = std::numeric_limits<ITensor::DimType64>::max();
    dims.nbDims = static_cast<std::int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        CHECK(shape[i] <= static_cast<std::size_t>(dim_max));
        dims.d[i] = static_cast<ITensor::DimType64>(shape[i]);
    }
    return dims;
}

std::vector<std::size_t> dimsToShape(nvinfer1::Dims const& dims)
{
    CHECK(dims.nbDims >= 0);
    std::vector<std::size_t> shape(dims.nbDims);
    for (std::int32_t i = 0; i < dims.nbDims; ++i)
    {
        CHECK(dims.d[i] >= 0);
        shape[i] = static_cast<std::size_t>(dims.d[i]);
    }
    return shape;
}

suggestify::runtime::TllmLogger defaultLogger{};

class StreamReader final : public nvinfer1::IStreamReader
{
public:
    StreamReader(std::filesystem::path fp)
    {
        mFile.open(fp.string(), std::ios::binary | std::ios::in);
        CHECK_WITH_INFO(mFile.good(), std::string("Error opening engine file: " + fp.string()));
    }

    virtual ~StreamReader()
    {
        if (mFile.is_open())
        {
            mFile.close();
        }
    }

    int64_t read(void* destination, int64_t nbBytes) final
    {
        if (!mFile.good())
        {
            return -1;
        }
        mFile.read(static_cast<char*>(destination), nbBytes);
        return mFile.gcount();
    }

    std::ifstream mFile;
};

void setWeightStreaming(nvinfer1::ICudaEngine& engine, float const gpuWeightsPercent)
{
    if (gpuWeightsPercent < 1)
    {
        int64_t streamableSize = engine.getStreamableWeightsSize();
        int64_t budget = gpuWeightsPercent * streamableSize;
        LOG_INFO("Set gpu weights percent to %f, which is %lld bytes. Valid range: %lld bytes - %lld bytes.",
            gpuWeightsPercent, budget, 0, streamableSize);
        engine.setWeightStreamingBudgetV2(budget);
    }
}

class LayerInfo
{
public:
    LayerInfo(std::optional<std::string> name, std::string type)
        : name(std::move(name))
        , type(std::move(type)){};
    std::optional<std::string> name;
    std::string type;
};

void assessLikelihoodOfRuntimeAllocation(
    nvinfer1::ICudaEngine const& engine, nvinfer1::IEngineInspector const& engineInspector)

{
    LOG_INFO("Inspecting the engine to identify potential runtime issues...");
    auto const profilingVerbosity = engine.getProfilingVerbosity();
    if (profilingVerbosity != nvinfer1::ProfilingVerbosity::kDETAILED)
    {
        LOG_INFO(
            "The profiling verbosity of the engine does not allow this analysis to proceed. Re-build the engine with "
            "'detailed' profiling verbosity to get more diagnostics.");
        return;
    }
    auto const* const layerTypeKey = "LayerType";
    auto const* const nameKey = "Name";
    auto const numLayers = engine.getNbLayers();
    LOG_INFO("Model has %i layers.", numLayers);
    std::vector<SizeType32> indexes(numLayers);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::vector<std::optional<LayerInfo>> layerInfos(numLayers);
    std::transform(indexes.cbegin(), indexes.cend(), layerInfos.begin(),
        [&](SizeType32 const idx)
        {
            auto const* const layerInfo
                = engineInspector.getLayerInformation(idx, nvinfer1::LayerInformationFormat::kJSON);

            auto const layerInfoCopy = std::string(layerInfo);
            auto const jsonLayerInfo = nlohmann::json::parse(layerInfoCopy);
            auto const layerJsonType = jsonLayerInfo.type();
            if (layerJsonType != nlohmann::detail::value_t::object)
            {
                return std::optional<LayerInfo>{};
            }
            if (!jsonLayerInfo.contains(layerTypeKey))
            {
                return std::optional<LayerInfo>{};
            }
            auto const& typeJson = jsonLayerInfo.at(layerTypeKey);
            if (typeJson.type() != nlohmann::detail::value_t::string)
            {
                return std::optional<LayerInfo>{};
            }
            std::optional<std::string> name{};
            if (jsonLayerInfo.contains(nameKey))
            {
                auto const& nameJson = jsonLayerInfo.at(nameKey);
                auto const nameJsonType = nameJson.type();
                if (nameJsonType == nlohmann::detail::value_t::string)
                {
                    name = nameJson.get<std::string>();
                }
            }
            return std::make_optional(LayerInfo{name, typeJson.get<std::string>()});
        });
    auto const layersWithInfoEnd = std::partition(
        layerInfos.begin(), layerInfos.end(), [](std::optional<LayerInfo> const& info) { return info.has_value(); });
    if (layersWithInfoEnd == layerInfos.begin())
    {
        LOG_INFO("Engine layer infos could not be parsed into useful information.");
        return;
    }
    auto const allocateLayersEnd = std::partition(layerInfos.begin(), layersWithInfoEnd,
        [](std::optional<LayerInfo> const& info) { return info.value().type == "allocate"; });
    auto numWarnings = 0;
    for (auto layerInfo = layerInfos.begin(); layerInfo != allocateLayersEnd; layerInfo++)
    {
        auto constexpr maxNumWarnings = 25;
        if (numWarnings < maxNumWarnings)
        {
            auto const layerName = layerInfo->value().name.value_or("");
            LOG_WARNING(
                "Layer '%s' has type '%s', which could lead to large runtime memory allocations. Performance "
                "might be degraded and / or you might run out of memory.",
                layerName.c_str(), layerInfo->value().type.c_str());
        }
        numWarnings++;
    }
    LOG_WARNING(
        "There were a total of %i layers with type 'allocate'. Some warnings might have been silenced to keep the "
        "output concise.",
        numWarnings);
}
}

TllmRuntime::TllmRuntime(
    RawEngine const& rawEngine, nvinfer1::ILogger* logger, float gpuWeightsPercent, bool useShapeInference)
    : mStream(std::make_shared<CudaStream>())
    , mBufferManager{mStream, true}
    , mRuntime{nvinfer1::createInferRuntime(static_cast<bool>(logger) ? *logger : defaultLogger)}
    , mUseShapeInference{useShapeInference}
    , mUserBufferEnabled{false}
{
    switch (rawEngine.getType())
    {
    case RawEngine::Type::FilePath:
    {
        auto reader = StreamReader(rawEngine.getPath());
        mEngine.reset(mRuntime->deserializeCudaEngine(reader));
        break;
    }
    case RawEngine::Type::AddressWithSize:
        mEngine.reset(mRuntime->deserializeCudaEngine(rawEngine.getAddress(), rawEngine.getSize()));
        break;
    case RawEngine::Type::HostMemory:
        mEngine.reset(
            mRuntime->deserializeCudaEngine(rawEngine.getHostMemory()->data(), rawEngine.getHostMemory()->size()));
        break;
    default: THROW("Unsupported raw engine type.");
    }

    CHECK_WITH_INFO(mEngine != nullptr, "Failed to deserialize cuda engine.");
    mEngineInspector.reset(mEngine->createEngineInspector());
    assessLikelihoodOfRuntimeAllocation(*mEngine, *mEngineInspector);
    setWeightStreaming(getEngine(), gpuWeightsPercent);
    auto const devMemorySize = mEngine->getDeviceMemorySizeV2();
    mEngineBuffer = mBufferManager.gpu(devMemorySize);
    LOG_INFO("[MemUsageChange] Allocated %.2f MiB for execution context memory.",
        static_cast<double>(devMemorySize) / 1048576.0);

    cacheTensorNames();
}

void TllmRuntime::cacheTensorNames()
{
    for (std::int32_t i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        auto const* const name = mEngine->getIOTensorName(i);
        if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            mInputTensorNames.emplace_back(name);
        }
        else if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            mOutputTensorNames.emplace_back(name);
        }
    }
}

nvinfer1::IExecutionContext& TllmRuntime::addContext(std::int32_t profileIndex)
{
    CHECK(0 <= profileIndex && profileIndex < mEngine->getNbOptimizationProfiles());
    mContexts.emplace_back(mEngine->createExecutionContextWithoutDeviceMemory());
    if (!mContexts.back())
    {
        if (mEngine->getStreamableWeightsSize() > 0)
        {
            THROW("Failed to allocate memory for weights. Please try reducing --gpu_weights_percent.");
        }
        else
        {
            THROW("Internal Error: Failed to create an execution context.");
        }
    }
    auto& context = *mContexts.back();
    context.setDeviceMemoryV2(mEngineBuffer->data(), static_cast<int64_t>(mEngineBuffer->getCapacity()));

    if (suggestify::common::Logger::getLogger()->isEnabled(suggestify::common::Logger::TRACE)
        && mContexts.size() == 1)
    {
        printEngineInfo();
    }

    context.setOptimizationProfileAsync(profileIndex, mStream->get());
    if (context.getNvtxVerbosity() == nvinfer1::ProfilingVerbosity::kDETAILED)
    {
        LOG_INFO(
            "The engine was built with kDETAILED profiling verbosity, which may result in small overheads at runtime.");
    }
    return context;
}

void TllmRuntime::printEngineInfo()
{
    auto& context = *(mContexts[0]);
    int const nIO = mEngine->getNbIOTensors();
    int const nOP = mEngine->getNbOptimizationProfiles();
    std::size_t mwn = 0;
    std::size_t mws = 0;

    std::vector<std::string> tensorNameList{};
    tensorNameList.reserve(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        tensorNameList.emplace_back(mEngine->getIOTensorName(i));
    }
    std::vector<std::map<std::string, std::string>> tiv(nIO);
    std::vector<std::vector<std::vector<nvinfer1::Dims64>>> topv(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        std::string name{tensorNameList[i]};
        char const* nameC{name.c_str()};
        mwn = std::max(mwn, name.size());
        tiv[i]["mode"] = mEngine->getTensorIOMode(nameC) == nvinfer1::TensorIOMode::kINPUT ? "I" : "O";
        tiv[i]["location"] = mEngine->getTensorLocation(nameC) == nvinfer1::TensorLocation::kDEVICE ? "GPU" : "CPU";
        tiv[i]["data_type"] = dataTypeToString(mEngine->getTensorDataType(nameC));
        tiv[i]["build_shape"] = shapeToString(mEngine->getTensorShape(nameC));
        mws = std::max(mws, tiv[i]["build_shape"].size());
        if (tiv[i]["mode"] == "I")
        {
            std::vector<std::vector<nvinfer1::Dims64>> topPerTensor(nOP);
            for (int k = 0; k < nOP; ++k)
            {
                if (tiv[i]["location"] == std::string("GPU"))
                {
                    std::vector<nvinfer1::Dims64> top(3);
                    top[0] = mEngine->getProfileShape(nameC, k, nvinfer1::OptProfileSelector::kMIN);
                    top[1] = mEngine->getProfileShape(nameC, k, nvinfer1::OptProfileSelector::kOPT);
                    top[2] = mEngine->getProfileShape(nameC, k, nvinfer1::OptProfileSelector::kMAX);
                    topPerTensor[k] = top;
                    mws = std::max(mws, shapeToString(top[2]).size());
                }
                else
                {
                    std::vector<nvinfer1::Dims64> top(3);
                    int const nDim = mEngine->getTensorShape(nameC).nbDims;
                    nvinfer1::Dims64 tensorShape{nDim, {-1}};
                    int const* pos = nullptr;
                    pos = mEngine->getProfileTensorValues(nameC, k, nvinfer1::OptProfileSelector::kMIN);
                    std::copy(pos, pos + nDim, tensorShape.d);
                    top[0] = tensorShape;
                    pos = mEngine->getProfileTensorValues(nameC, k, nvinfer1::OptProfileSelector::kOPT);
                    std::copy(pos, pos + nDim, tensorShape.d);
                    top[1] = tensorShape;
                    pos = mEngine->getProfileTensorValues(nameC, k, nvinfer1::OptProfileSelector::kMAX);
                    std::copy(pos, pos + nDim, tensorShape.d);
                    top[2] = tensorShape;
                    topPerTensor[k] = top;
                }
            }
            topv[i] = topPerTensor;
        }
        else
        {
            topv[i] = std::vector<std::vector<nvinfer1::Dims64>>(nOP);
        }
    }
    for (int k = 0; k < nOP; ++k)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int i = 0; i < nIO; ++i)
            {
                std::string name = tensorNameList[i];
                char const* nameC = name.c_str();
                if (tiv[i]["mode"] == "I")
                {
                    if (tiv[i]["location"] == std::string("GPU"))
                    {
                        context.setInputShape(nameC, topv[i][k][j]);
                    }
                    else
                    {
                        context.setInputTensorAddress(nameC, topv[i][k][j].d);
                    }
                }
                else
                {
                    CHECK_WITH_INFO(context.allInputDimensionsSpecified(), "Input dimensions not specified");
                    CHECK_WITH_INFO(context.allInputShapesSpecified(), "Input shapes not specified");
                    if (tiv[i]["location"] == std::string("GPU"))
                    {
                        topv[i][k].push_back(context.getTensorShape(nameC));
                    }
                    else
                    {
                        int const nDim = mEngine->getTensorShape(nameC).nbDims;
                        nvinfer1::Dims64 tensorShape{nDim, {}};
                        int const* pos = reinterpret_cast<int const*>(context.getTensorAddress(nameC));
                        std::copy(pos, pos + nDim, tensorShape.d);
                        topv[i][k].push_back(tensorShape);
                    }
                }
            }
        }
    }

    std::string info;
    LOG_TRACE("Information of engine input / output.");
    LOG_TRACE(std::string(mwn + mws + 24, '='));
    info = alignText("Name", mwn) + "|I/O|Location|DataType|" + alignText("Shape", mws) + "|";
    LOG_TRACE(info.c_str());
    LOG_TRACE(std::string(mwn + mws + 24, '-'));
    for (int i = 0; i < nIO; ++i)
    {
        info = alignText(tensorNameList[i], mwn, false) + "|";
        info += alignText(tiv[i]["mode"], 3) + "|";
        info += alignText(tiv[i]["location"], 8) + "|";
        info += alignText(tiv[i]["data_type"], 8) + "|";
        info += alignText(tiv[i]["build_shape"], mws) + "|";
        LOG_TRACE(info.c_str());
    }
    LOG_TRACE(std::string(mwn + mws + 24, '='));
    LOG_TRACE("Information of optimization profile.");
    for (int k = 0; k < nOP; ++k)
    {
        LOG_TRACE("Optimization Profile %d:", k);
        LOG_TRACE(std::string(mwn + mws * 3 + 4, '='));
        info = alignText("Name", mwn) + "|";
        info += alignText("Min", mws) + "|";
        info += alignText("Opt", mws) + "|";
        info += alignText("Max", mws) + "|";
        LOG_TRACE(info.c_str());
        LOG_TRACE(std::string(mwn + mws * 3 + 4, '-'));
        for (int i = 0; i < nIO; ++i)
        {
            auto const& top = topv[i][k];
            info = alignText(tensorNameList[i], mwn, false) + "|";
            info += alignText(shapeToString(top[0]), mws) + "|";
            info += alignText(shapeToString(top[1]), mws) + "|";
            info += alignText(shapeToString(top[2]), mws) + "|";
            LOG_TRACE(info.c_str());
        }
        LOG_TRACE(std::string(mwn + mws * 3 + 4, '='));
    }
}

void TllmRuntime::clearContexts()
{
    for (auto& context : mContexts)
    {
        context.reset();
    }
    mContexts.clear();
}

bool TllmRuntime::executeContext(SizeType32 contextIndex) const
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    auto res = context.enqueueV3(mStream->get());
    sync_check_cuda_error();
    return res;
}

void TllmRuntime::setInputTensorsImpl(SizeType32 contextIndex, TensorMap const& tensorMap, bool throwOnMiss)
{
    NVTX3_FUNC_RANGE();
    auto& context = getContext(contextIndex);
    for (auto const& name : mInputTensorNames)
    {
        auto const pos = tensorMap.find(name);
        if (pos == tensorMap.end())
        {
            if (throwOnMiss)
            {
                auto expectedShape = mEngine->getTensorShape(name.c_str());
                THROW("Input tensor '%s' not found; expected shape: %s", name.c_str(),
                    ITensor::toString(expectedShape).c_str());
            }
            else
            {
                continue;
            }
        }

        auto const& tensor = pos->second;
        auto const tensorDtype = tensor->getDataType();
        auto const engineDtype = mEngine->getTensorDataType(name.c_str());
        CHECK_WITH_INFO(tensorDtype == engineDtype
                || (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
            "%s: expected type %d, provided type %d", name.c_str(), static_cast<std::int32_t>(engineDtype),
            static_cast<std::int32_t>(tensorDtype));

        auto const tensorShape = tensor->getShape();
        auto const setInputShapeSuccess = context.setInputShape(name.c_str(), tensorShape);
        if (!setInputShapeSuccess)
        {
            auto const minShape
                = mEngine->getProfileShape(name.c_str(), contextIndex, nvinfer1::OptProfileSelector::kMIN);
            auto const maxShape
                = mEngine->getProfileShape(name.c_str(), contextIndex, nvinfer1::OptProfileSelector::kMAX);

            THROW("Tensor '%s' has invalid shape %s, expected in range min %s, max %s", name.c_str(),
                ITensor::toString(tensorShape).c_str(), ITensor::toString(minShape).c_str(),
                ITensor::toString(maxShape).c_str());
        }
        auto* const data = tensor->data();
        if (static_cast<bool>(data))
        {
            context.setInputTensorAddress(name.c_str(), data);
        }
        else
        {
            CHECK_WITH_INFO(tensor->getSize() == 0, std::string("Invalid data for tensor: ") + name);
            if (!mDummyTensor)
            {
                mDummyTensor = mBufferManager.gpu(ITensor::makeShape({1}));
            }
            context.setInputTensorAddress(name.c_str(), mDummyTensor->data());
        }
    }
}

void TllmRuntime::setStaticInputTensors(TensorMap const& tensorMap)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();

    CHECK_WITH_INFO(getNbContexts() > 0, "Contexts should be created before calling setStaticInputTensors");
    for (auto contextIndex = 0; contextIndex < getNbContexts(); ++contextIndex)
    {
        setInputTensorsImpl(contextIndex, tensorMap, false);
    }

    auto const begin = mInputTensorNames.begin();
    auto end = mInputTensorNames.end();
    for (auto const& [name, tensor] : tensorMap)
    {
        end = std::remove(begin, end, name);
    }
    mInputTensorNames.erase(end, mInputTensorNames.end());

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TllmRuntime::setInputTensors(SizeType32 contextIndex, TensorMap const& tensorMap)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();
    setInputTensorsImpl(contextIndex, tensorMap, true);

    auto& context = getContext(contextIndex);
    if (mUseShapeInference)
    {
        NVTX3_SCOPED_RANGE(infer_shapes);
        char const* missing = nullptr;
        auto const nbMissing = context.inferShapes(1, &missing);
        if (nbMissing > 0)
        {
            THROW("Input shape not specified: %s", missing);
        }
        else if (nbMissing < 0)
        {
            THROW("Invalid input shape");
        }
    }

    {
        NVTX3_SCOPED_RANGE(final_checks);
        CHECK_WITH_INFO(context.allInputDimensionsSpecified(), "Input dimensions not specified");
        CHECK_WITH_INFO(context.allInputShapesSpecified(), "Input shapes not specified");
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TllmRuntime::setOutputTensors(SizeType32 contextIndex, TensorMap& tensorMap)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();
    if (isUserBufferEnabled())
    {
        setUserBufferTensors(contextIndex, tensorMap);
    }

    auto& context = getContext(contextIndex);
    for (auto const& name : mOutputTensorNames)
    {
        auto const engineDtype = mEngine->getTensorDataType(name.c_str());
        auto const pos = tensorMap.find(name);
        if (pos != tensorMap.end())
        {
            auto const& tensor = pos->second;
            auto const tensorDtype = tensor->getDataType();
            CHECK_WITH_INFO(tensorDtype == engineDtype
                    || (tensorDtype == nvinfer1::DataType::kFP8 && engineDtype == nvinfer1::DataType::kHALF),
                "%s: expected type %d, provided type %d", name.c_str(), static_cast<std::int32_t>(engineDtype),
                static_cast<std::int32_t>(tensorDtype));

            if (mUseShapeInference)
            {
                auto const dims = context.getTensorShape(name.c_str());
                tensor->reshape(dims);
            }
            context.setTensorAddress(name.c_str(), tensor->data());
        }
        else if (mUseShapeInference)
        {
            auto const dims = context.getTensorShape(name.c_str());
            auto tensor = ITensor::SharedPtr(mBufferManager.gpu(dims, engineDtype));
            tensorMap.insert(pos, std::make_pair(name, tensor));
            context.setTensorAddress(name.c_str(), tensor->data());
        }
        else
        {
            THROW("Tensor %s is not found in tensorMap and shape inference is not allowed", name.c_str());
        }
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TllmRuntime::setUserBufferTensors(SizeType32 contextIndex, TensorMap& tensorMap)
{
    auto startsWith = [](std::string const& str, std::string const& prefix) -> bool
    { return str.size() > prefix.size() && str.compare(0, prefix.size(), prefix) == 0; };
    std::string prefix(suggestify::runtime::ub::tensor_prefix);
    auto& context = getContext(contextIndex);
    for (auto const& name : mOutputTensorNames)
    {
        auto const pos = tensorMap.find(name);
        if (pos != tensorMap.end() || !startsWith(name, prefix))
        {
            continue;
        }
        auto const engineDtype = mEngine->getTensorDataType(name.c_str());
        auto const dims = context.getTensorShape(name.c_str());
        void* ubBuffer = nullptr;
        if (name[prefix.size()] == '0')
        {
            ubBuffer = suggestify::runtime::ub::ub_get(0).addr;
        }
        else if (name[prefix.size()] == '1')
        {
            ubBuffer = suggestify::runtime::ub::ub_get(1).addr;
        }
        else
        {
            CHECK(false);
        }
        auto tensor = ITensor::SharedPtr(ITensor::wrap(ubBuffer, engineDtype, dims));
        tensorMap.insert(pos, std::make_pair(name, tensor));
        context.setTensorAddress(name.c_str(), ubBuffer);
    }
}

void TllmRuntime::initializeUserBuffer(SizeType32 tpSize, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    SizeType32 maxSequenceLength, SizeType32 hiddenSize, std::optional<SizeType32> maxNumTokens)
{
    auto startsWith = [](std::string const& str, std::string const& prefix) -> bool
    { return str.size() > prefix.size() && str.compare(0, prefix.size(), prefix) == 0; };
    std::string prefix(suggestify::runtime::ub::tensor_prefix);
    for (auto const& name : mOutputTensorNames)
    {
        if (startsWith(name, prefix))
        {
            mUserBufferEnabled = true;
            break;
        }
    }
    if (!mUserBufferEnabled)
    {
        return;
    }
    size_t realHiddenSize = hiddenSize * tpSize;
    size_t tokensNum = maxNumTokens.value_or(maxBatchSize * maxBeamWidth * maxSequenceLength);
    CHECK(tokensNum > 0);
    size_t maxMessageSize = tokensNum * realHiddenSize * sizeof(half);
    LOG_INFO("[UserBuffer] MaxBatchSize %d, maxBeamWidth %d, maxSequenceLength %d, maxNumTokens %d, select %lu",
        maxBatchSize, maxBeamWidth, maxSequenceLength, maxNumTokens.has_value() ? maxNumTokens.value() : 0, tokensNum);
    LOG_INFO("[UserBuffer] Allocated %.2f MiB for execution context memory.",
        static_cast<double>(maxMessageSize * 2) / 1048576.0);
    suggestify::runtime::ub::ub_initialize(tpSize);
    suggestify::runtime::ub::ub_allocate(0, maxMessageSize);
    suggestify::runtime::ub::ub_allocate(1, maxMessageSize);
}

CudaStream const& TllmRuntime::getStream() const
{
    return *mStream;
}

bool TllmRuntime::hasLayerProfiler(SizeType32 contextId) const
{
    return mContexts[contextId]->getProfiler() != nullptr;
}

void TllmRuntime::setLayerProfiler()
{
    mLayerProfiler = std::make_unique<LayerProfiler>();
    for (auto& context : mContexts)
    {
        context->setProfiler(mLayerProfiler.get());
        context->setEnqueueEmitsProfile(false);
    }
}

std::string TllmRuntime::getLayerProfileInfo() const
{
    CHECK(mLayerProfiler);
    return mLayerProfiler->getLayerProfile();
}

void TllmRuntime::reportToProfiler(SizeType32 contextId)
{
    mContexts[contextId]->reportToProfiler();
}

void TllmRuntime::loadManagedWeights(RawEngine const& rawEngine, int localRank)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_FUNC_RANGE();
    auto& engine = getEngine();
    auto& manager = getBufferManager();
    if (rawEngine.getManagedWeightsMapOpt().has_value())
    {
        LOG_DEBUG("Loading managed weights from raw engine");
        auto executorMap = rawEngine.getManagedWeightsMapOpt().value();
        for (auto const& [name, weight] : executorMap)
        {
            LOG_DEBUG("Loading managed weight: %s", name.c_str());
            auto iTensor = suggestify::executor::detail::toITensor(weight);
            auto weightsDevice = std::shared_ptr<ITensor>{manager.copyFrom(*iTensor, MemoryType::kGPU)};
            mManagedWeightsMap.insert(std::make_pair(name, weightsDevice));
        }
    }
    else
    {
        LOG_DEBUG("Loading managed weights from file");
        auto const enginePath = rawEngine.getPathOpt();
        CHECK_WITH_INFO(enginePath.has_value(), "Engine path is not set.");
        auto weightPath
            = enginePath->parent_path() / ("rank" + std::to_string(localRank) + "_managed_weights.safetensors");
        auto managed_weights = common::safetensors::ISafeTensor::open(weightPath.string().c_str());
        for (auto const& name : managed_weights->keys())
        {
            LOG_DEBUG("Loading managed weight: %s", name.c_str());
            auto const weight = managed_weights->getTensor(name.c_str());
            CHECK(weight->dtype() == engine.getTensorDataType(name.c_str()));
            auto weightsDevice
                = std::shared_ptr<ITensor>{manager.allocate(MemoryType::kGPU, weight->trtDims(), weight->dtype())};
            manager.copy(weight->data(), *weightsDevice, MemoryType::kCPU);
            mManagedWeightsMap.insert(std::make_pair(name, weightsDevice));
        }
    }
    setStaticInputTensors(mManagedWeightsMap);
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
