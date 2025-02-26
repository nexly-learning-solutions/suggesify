

#pragma once

#include "../batch_manager/kvCacheConfig.h"
#include "../executor/types.h"
#include "bufferManager.h"
#include "common.h"
#include "cudaEvent.h"
#include "generationInput.h"
#include "generationOutput.h"
#include "iTensor.h"
#include "modelConfig.h"
#include "rawEngine.h"
#include "samplingConfig.h"
#include "worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace suggestify::batch_manager
{
class TrtGptModelV1;
}

namespace suggestify::batch_manager::kv_cache_manager
{
class BaseKVCacheManager;
}

namespace suggestify::runtime
{

namespace utils
{
std::vector<uint8_t> loadEngine(std::string const& enginePath);
}

class AllReduceBuffers;
class IStatefulGptDecoder;
class NcclCommunicator;
class RuntimeBuffers;
class TllmRuntime;

class [[deprecated("Use the executor API instead.")]] GptSession
{
    using BaseKVCacheManager = batch_manager::kv_cache_manager::BaseKVCacheManager;
    using KvCacheConfig = batch_manager::kv_cache_manager::KvCacheConfig;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TokenGeneratedCallback = std::function<void(SizeType32 step, bool finished)>;

public:
    using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

    class Config
    {
    public:
        Config(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxSequenceLength,
            float gpuWeightsPercent = 1.0)
            : maxBatchSize{maxBatchSize}
            , maxBeamWidth{maxBeamWidth}
            , maxSequenceLength{maxSequenceLength}
            , gpuWeightsPercent{gpuWeightsPercent}
        {
        }

        SizeType32 maxBatchSize;
        SizeType32 maxBeamWidth;
        SizeType32 maxSequenceLength;
        float gpuWeightsPercent;
        bool decoderPerRequest{false};
        bool cudaGraphMode{false};
        KvCacheConfig kvCacheConfig{};
        std::optional<SizeType32> ctxMicroBatchSize = std::nullopt;
        std::optional<SizeType32> genMicroBatchSize = std::nullopt;
        std::optional<executor::DecodingMode> decodingMode = std::nullopt;
        bool normalizeLogProbs = true;
    };

    class GenerationProfiler
    {
    public:
        static constexpr unsigned int flags{cudaEventDefault};

        GenerationProfiler()
            : start(flags)
            , end(flags)
        {
        }

        CudaEvent const& getStart() const
        {
            return start;
        }

        CudaEvent const& getEnd() const
        {
            return end;
        }

        float getElapsedTimeMs()
        {
            start.synchronize();
            end.synchronize();

            float result;
            CUDA_CHECK(::cudaEventElapsedTime(&result, start.get(), end.get()));

            return result;
        }

    private:
        CudaEvent start;
        CudaEvent end;
    };

    GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        RawEngine const& rawEngine, LoggerPtr logger = nullptr);

    GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        void const* engineBuffer, std::size_t engineSize, LoggerPtr logger = nullptr)
        : GptSession(sessionConfig, modelConfig, worldConfig, RawEngine(engineBuffer, engineSize), std::move(logger))
    {
    }

    GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::vector<uint8_t> const& engineBuffer, LoggerPtr logger = nullptr)
        : GptSession(sessionConfig, modelConfig, worldConfig, RawEngine(engineBuffer.data(), engineBuffer.size()),
            std::move(logger))
    {
    }

    GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::string const& engineFile, LoggerPtr logger = nullptr);

    [[nodiscard]] nvinfer1::ILogger& getLogger() const;

    [[nodiscard]] BufferManager const& getBufferManager() const;
    [[nodiscard]] BufferManager::CudaStreamPtr getRuntimeStreamPtr() const;

    [[nodiscard]] ModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] WorldConfig const& getWorldConfig() const
    {
        return mWorldConfig;
    }

    [[nodiscard]] int getDevice() const noexcept
    {
        return mDevice;
    }

    [[nodiscard]] bool getNormalizeLogProbs() const noexcept
    {
        return mNormalizeLogProbs;
    }

    [[nodiscard]] nvinfer1::IEngineInspector& getEngineInspector() const;

    [[nodiscard]] nvinfer1::DataType getLogitDataType() const;

    [[nodiscard]] nvinfer1::DataType getTensorDataType(std::string const& name) const;

    [[nodiscard]] nvinfer1::Dims getTensorShape(std::string const& name) const;

    void generate(GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig,
        std::shared_ptr<GenerationProfiler> const generationProfiler = nullptr);

    void setLayerProfiler();

    [[nodiscard]] std::string getLayerProfileInfo() const;

private:
    [[nodiscard]] bool useCudaGraphs()
    {
        return !mCudaGraphInstances.empty();
    }

    void generateBatched(std::vector<GenerationOutput>& microBatchesOutputs,
        std::vector<GenerationInput> const& microBatchesInputs, SamplingConfig const& samplingConfig,
        TokenGeneratedCallback const& onTokenGenerated, std::shared_ptr<GenerationProfiler> const generationProfiler);

    void setup(Config const& sessionConfig);

    void createContexts();
    void createBuffers(SizeType32 numMicroBatches);
    void createDecoders(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, nvinfer1::DataType logitsType, bool decoderPerRequest,
        SizeType32 numMicroBatches, executor::DecodingMode const& decodingMode);
    void createKvCacheManager(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, KvCacheConfig const& config);
    void createCustomAllReduceWorkspace(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSequenceLength);

    void executeContextStep(std::vector<GenerationInput> const& generationBatchesInputs,
        std::vector<SizeType32> const& generationBatchesOffsets, BaseKVCacheManager const* kvCacheManager);
    SizeType32 executeGenerationStep(SizeType32 step, std::vector<GenerationInput> const& microBatchesInputs,
        std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType32> const& microBatchOffsets,
        BaseKVCacheManager* kvCacheManager, std::vector<bool>& microBatchesFinished);

    void decoderStepAsync(SizeType32 decoderStep, SizeType32 microBatchId);

    bool shouldStopSync(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 microBatchId);

    void finalize(SizeType32 microBatchId, SamplingConfig const& samplingConfig);

    void kvCacheAddSequences(SizeType32 beamWidth, SizeType32 microBatchId, SizeType32 firstBatchIdx);

    ITensor::SharedPtr initDecoder(ITensor& outputIds, GenerationInput const& inputs, GenerationOutput const& outputs,
        SamplingConfig const& samplingConfig, SizeType32 microBatchId) const;

    TokenGeneratedCallback createOnTokenGeneratedCallback(GenerationOutput& outputs);

    bool shouldUseKVCacheManager() const;

    class CudaGraphExecutor
    {
    public:
        CudaGraphExecutor() = default;

        ~CudaGraphExecutor()
        {
            try
            {
                clear();
            }
            catch (std::exception& e)
            {
                LOG_EXCEPTION(e);
            }
        }

        bool hasInstance()
        {
            return mInstance != nullptr;
        }

        void clear();
        void prepareNextGraph(TllmRuntime const& runtime, SizeType32 nextContextId);
        void launch(CudaStream const& stream);

    private:
        void create(cudaGraph_t const& graph);
        bool update(cudaGraph_t const& graph);
        void uploadToStream(CudaStream const& stream);

        cudaGraphExec_t mInstance;
    };

    class MicroBatchConfig
    {
    public:
        MicroBatchConfig()
            : numCtxBatches{1}
            , numGenBatches{1}
            , ctxBatchSize{0}
            , genBatchSize{0}
        {
        }

        explicit MicroBatchConfig(SizeType32 maxBatchSize, SizeType32 pipelineParallelism,
            std::optional<SizeType32> genMicroBatchSize, std::optional<SizeType32> ctxMicroBatchSize);

        constexpr SizeType32 numCtxPerGen() const
        {
            return numCtxBatches / numGenBatches;
        }

        constexpr SizeType32 getGenGraphId(SizeType32 flipFlopId, SizeType32 generationBatchId) const
        {
            return flipFlopId * numGenBatches + generationBatchId;
        }

        SizeType32 numCtxBatches;
        SizeType32 numGenBatches;
        SizeType32 ctxBatchSize;
        SizeType32 genBatchSize;
    };

    friend class batch_manager::TrtGptModelV1;

private:
    ModelConfig const mModelConfig;
    WorldConfig const mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<NcclCommunicator> mPipelineComm;
    std::shared_ptr<CudaStream> mCommStream;
    CudaEvent mCommEvent{};

    std::shared_ptr<AllReduceBuffers> mAllReduceBuffers;

    SizeType32 mDecoderMaxSequenceLength{};
    std::vector<SizeType32> mDecoderMaxAttentionWindowVec{};
    SizeType32 mDecoderMaxAttentionWindow{};
    SizeType32 mDecoderSinkTokenLength{};

    LoggerPtr mLogger;
    std::shared_ptr<TllmRuntime> mRuntime;
    std::shared_ptr<BaseKVCacheManager> mKvCacheManager;

    MicroBatchConfig mMicroBatchConfig;
    std::vector<std::shared_ptr<IStatefulGptDecoder>> mDecoders;
    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;
    std::vector<CudaEvent> mReceivedEvents;

    bool mCudaGraphMode{false};
    std::vector<CudaGraphExecutor> mCudaGraphInstances;

    bool mNormalizeLogProbs = true;
};

}
