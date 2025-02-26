
#include "runtimeBuffers.h"

#include "suggestify/batch_manager/kvCacheManager.h"
#include "suggestify/common/assert.h"
#include "runtimeKernels.h"
#include "tllmRuntime.h"
#include "utils/sessionUtils.h"

#include <algorithm>

using namespace suggestify::runtime;
namespace tc = suggestify::common;

void RuntimeBuffers::clear()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    contextLengthsHost = nullptr;
    contextLengthsDevice = nullptr;

    logits = nullptr;
    sequenceLengths = nullptr;
    lastTokenIds = nullptr;
    requestTypes = nullptr;

    cacheIndirectionDecoderInput = nullptr;
    cacheIndirectionDecoderOutput = nullptr;

    cumLogProbs = nullptr;
    logProbs = nullptr;

    hiddenStates = nullptr;

    allocated = false;
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::clearTensorMaps()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    for (auto& buffer : inputBuffers)
        buffer.clear();
    for (auto& buffer : outputBuffers)
        buffer.clear();
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::create(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = engine.getTensorDataType("logits");
        logits = manager.emptyTensor(MemoryType::kGPU, logitsType);
        originalLogitsPtr = logits;

        allGenerationLogits = manager.emptyTensor(MemoryType::kGPU, logitsType);
        if (modelConfig.computeGenerationLogits())
        {
            cacheGenerationFragmentPointerDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT64);
            cacheGenerationFragmentPointerHost
                = manager.emptyTensor(MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT64);

            generationLogitsFragments = std::make_shared<std::vector<TensorPtr>>();
        }
    }

    lastTokenIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    bool transformerBased = modelConfig.isTransformerBased();
    bool rnnBased = modelConfig.isRnnBased();

    contextLengthsHost = manager.emptyTensor(MemoryType::kPINNEDPOOL, nvinfer1::DataType::kINT32);
    if (transformerBased)
    {
        if (modelConfig.useGptAttentionPlugin())
        {
            requestTypes = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        }
        transformerBuffers.emplace(runtime, modelConfig, worldConfig);
    }
    if (rnnBased)
    {
        requestTypes = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        rnnStateBuffers.emplace(runtime, modelConfig, worldConfig);
    }

    cacheIndirectionDecoderInput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    cacheIndirectionDecoderOutput = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    nbFinished = BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::initFromInput(ITensor const& inputIds, TensorPtr const& inputLengths, bool inputPacked,
    SizeType32 beamWidth, std::vector<SizeType32> maxAttentionWindowVec, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, BufferManager& manager)
{
    contextLengthsDevice = inputLengths;
    contextLengthsHost->reshape(inputLengths->getShape());
    manager.copy(*contextLengthsDevice, *contextLengthsHost);
    manager.getStream().synchronize();

    generationConfig = GenerationConfig::fromInput(inputIds, *contextLengthsHost, inputPacked, beamWidth,
        maxAttentionWindowVec, maxAttentionWindow, sinkTokenLength, maxSequenceLength);
}

void RuntimeBuffers::reshape(ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxAttentionWindow = generationConfig.maxAttentionWindow;
    auto const maxSeqLength = generationConfig.maxSeqLength;
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    if (worldConfig.isLastPipelineParallelRank())
    {
        if (modelConfig.computeContextLogits())
        {
            if (!modelConfig.computeGenerationLogits())
            {
                allGenerationLogits->reshape(ITensor::makeShape({1, batchSize, beamWidth, vocabSizePadded}));
            }
        }
        else
        {
            if (modelConfig.computeGenerationLogits())
            {
                logits = originalLogitsPtr;
            }
            logits->reshape(ITensor::makeShape({batchSize, 1, vocabSizePadded}));
        }

        if (modelConfig.computeGenerationLogits())
        {
            allGenerationLogits->reshape(
                ITensor::makeShape({(maxSeqLength - maxInputLength), batchSize, beamWidth, vocabSizePadded}));

            cacheGenerationFragmentPointerDevice->reshape(
                ITensor::makeShape({batchSize, (maxSeqLength - maxInputLength)}));
            cacheGenerationFragmentPointerHost->reshape(
                ITensor::makeShape({batchSize, (maxSeqLength - maxInputLength)}));
        }
    }

    lastTokenIds->reshape(ITensor::makeShape({batchSize}));

    if (transformerBuffers)
    {
        if (modelConfig.useGptAttentionPlugin())
        {
            requestTypes->reshape(ITensor::makeShape({batchSize}));
        }
        transformerBuffers->reshape(generationConfig, modelConfig, worldConfig);
    }

    if (rnnStateBuffers)
    {
        requestTypes->reshape(ITensor::makeShape({batchSize}));
        rnnStateBuffers->reshape(generationConfig, modelConfig, worldConfig);
    }

    auto const cacheIndirShape = ITensor::makeShape({batchSize, beamWidth, maxAttentionWindow});
    cacheIndirectionDecoderInput->reshape(cacheIndirShape);
    cacheIndirectionDecoderOutput->reshape(cacheIndirShape);

    if (worldConfig.isPipelineParallel())
    {
        auto const maxNumTokens = std::max(beamWidth, maxInputLength);
        auto const hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();
        auto const hiddenStatesShape = ITensor::makeShape(
            {batchSize, maxNumTokens, hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    allocated = true;
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::reset(BufferManager& manager)
{
    clearTensorMaps();
    manager.setZero(*cacheIndirectionDecoderInput);
    manager.setZero(*cacheIndirectionDecoderOutput);

    if (transformerBuffers)
    {
        transformerBuffers->reset(manager);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->reset(manager);
    }
}

std::vector<RuntimeBuffers> RuntimeBuffers::split(
    SizeType32 contextBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    std::vector<RuntimeBuffers> bufferSlices;
    auto const generationBatchSize = generationConfig.batchSize;
    bufferSlices.reserve(tc::ceilDiv(generationBatchSize, contextBatchSize));
    if (contextBatchSize >= generationBatchSize)
    {
        bufferSlices.emplace_back(*this);
    }
    else
    {
        for (auto offset = 0; offset < generationBatchSize; offset += contextBatchSize)
        {
            auto const batchSize = std::min(contextBatchSize, generationBatchSize - offset);
            auto& buffers = bufferSlices.emplace_back();
            buffers.generationConfig = generationConfig;
            buffers.generationConfig.batchSize = batchSize;

            buffers.contextLengthsHost = ITensor::slice(contextLengthsHost, offset, batchSize);
            buffers.contextLengthsDevice = ITensor::slice(contextLengthsDevice, offset, batchSize);

            if (worldConfig.isLastPipelineParallelRank() && !modelConfig.computeContextLogits())
            {
                buffers.logits = ITensor::slice(logits, offset, batchSize);
            }

            buffers.lastTokenIds = ITensor::slice(lastTokenIds, offset, batchSize);

            if (transformerBuffers)
            {
                buffers.transformerBuffers
                    = transformerBuffers->sliceTo(generationConfig, modelConfig, offset, batchSize);
            }

            if (rnnStateBuffers)
            {
                buffers.rnnStateBuffers = rnnStateBuffers->sliceTo(offset, batchSize);
            }

            if (requestTypes != nullptr)
            {
                buffers.requestTypes = ITensor::slice(requestTypes, offset, batchSize);
            }
            if (worldConfig.isPipelineParallel())
            {
                CHECK_WITH_INFO(hiddenStates->getShape().nbDims == 3,
                    "Invalid shape for hiddenStates.");
                buffers.hiddenStates = ITensor::slice(hiddenStates, offset, batchSize);
            }

            buffers.cacheIndirectionDecoderOutput = ITensor::slice(cacheIndirectionDecoderOutput, offset, batchSize);

            if (modelConfig.usePromptTuning())
            {
                auto const& ptuningEnabled = promptTuningParams.promptTuningEnabled;
                buffers.promptTuningParams.promptTuningEnabled
                    = std::vector<bool>(ptuningEnabled.begin() + offset, ptuningEnabled.begin() + offset + batchSize);

                buffers.promptTuningParams.tasks = ITensor::slice(promptTuningParams.tasks, offset, batchSize);
            }
        }
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return bufferSlices;
}

void RuntimeBuffers::gatherLastTokenLogits(
    BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    CHECK_WITH_INFO(modelConfig.computeContextLogits(),
        "Gather last token logits is only necessary when context logits are computed");

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        TensorPtr tiledTensor = ITensor::slice(allGenerationLogits, 0, 1);
        tiledTensor->squeeze(0);
        kernels::gatherLastTokenLogits(*tiledTensor, *logits, *lastTokenIds, manager.getStream());
        manager.getStream().synchronize();

        std::swap(logits, tiledTensor);
        if (modelConfig.usePackedInput())
        {
            tiledTensor->reshape(
                ITensor::makeShape({generationConfig.inputLengthSum, vocabSizePadded}));
        }
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::postContextStep(std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;

    if (transformerBuffers)
    {
        transformerBuffers->postContextStep(this, contextBuffers, manager, modelConfig, worldConfig);
    }
    if (rnnStateBuffers)
    {
        rnnStateBuffers->postContextStep(this, contextBuffers, manager, modelConfig, worldConfig);
    }

    manager.copy(*contextLengthsDevice, *outputLengths);
    sequenceLengths = ITensor::view(outputLengths);
    sequenceLengths->reshape(ITensor::makeShape({batchSize * beamWidth}));
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth}));

    if (modelConfig.usePromptTuning())
    {
        std::vector<SizeType32> reqBeamWidths(batchSize, beamWidth);
        std::vector<SizeType32> reqPromptLengths;
        promptTuningTasksHost = manager.copyFrom(*promptTuningParams.tasks, MemoryType::kPINNEDPOOL);
        promptTuningParams.fillTasksTensor(promptTuningTasksHost, batchSize, 0, reqBeamWidths, reqPromptLengths,
            manager, modelConfig.usePackedInput());
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareContextStep(TensorPtr const& inputIds, TokenIdType const padId, BufferManager& manager,
    batch_manager::kv_cache_manager::BaseKVCacheManager const* kvCacheManager, SizeType32 firstBatchSlotIdx,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& stream = manager.getStream();

    sequenceLengths = contextLengthsDevice;

    if (transformerBuffers)
    {
        transformerBuffers->prepareContextStep(
            this, inputIds, padId, manager, kvCacheManager, firstBatchSlotIdx, modelConfig, worldConfig);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->prepareContextStep(this, manager);
    }

    if (modelConfig.usePackedInput())
    {
        kernels::invokeInclusiveSum(*lastTokenIds, *contextLengthsDevice, manager, stream);
    }
    else
    {
        manager.copy(*contextLengthsDevice, *lastTokenIds);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

RuntimeBuffers::TensorPtr RuntimeBuffers::prepareNextStep(SizeType32 const step, BufferManager& manager,
    batch_manager::kv_cache_manager::BaseKVCacheManager* kvCacheManager, SizeType32 firstBatchSlotIdx,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& stream = manager.getStream();
    SizeType32 const batchSize = generationConfig.batchSize;
    SizeType32 const beamWidth = generationConfig.beamWidth;

    auto const inputShape = [&modelConfig, batchSize, beamWidth]()
    {
        if (modelConfig.usePackedInput())
        {
            return ITensor::makeShape({batchSize * beamWidth});
        }
        else
        {
            return ITensor::makeShape({batchSize * beamWidth, 1});
        }
    }();

    auto nextInputIds = newTokens ? ITensor::view(newTokens, inputShape) : TensorPtr{};

    if (transformerBuffers)
    {
        transformerBuffers->prepareNextStep(
            this, step, manager, kvCacheManager, firstBatchSlotIdx, modelConfig, worldConfig);
    }

    kernels::invokeFill(*lastTokenIds, 1, stream);
    if (modelConfig.usePackedInput())
    {
        kernels::invokeInclusiveSum(*lastTokenIds, *lastTokenIds, manager, stream);
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return nextInputIds;
}

void RuntimeBuffers::getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType32 const step,
    TensorPtr const& inputIds, TensorPtr const& commPtrs, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig) const
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.clear();
    outputBuffers.clear();

    if (transformerBuffers)
    {
        transformerBuffers->getRuntimeBuffers(
            this, inputBuffers, outputBuffers, step, inputIds, modelConfig, worldConfig);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->getRuntimeBuffers(this, inputBuffers, outputBuffers, step, inputIds, modelConfig, worldConfig);
    }

    if (worldConfig.isTensorParallel())
    {
        inputBuffers.insert_or_assign("all_reduce_workspace", commPtrs);
    }

    if (modelConfig.usePromptTuning())
    {
        inputBuffers.insert_or_assign("prompt_embedding_table", promptTuningParams.embeddingTable);
        inputBuffers.insert_or_assign("tasks", promptTuningParams.tasks);
        inputBuffers.insert_or_assign("prompt_vocab_size", promptTuningParams.vocabSize);
    }

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
