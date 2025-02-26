
#pragma once

#include "bufferManager.h"
#include "modelConfig.h"
#include "worldConfig.h"

#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

namespace suggestify::runtime
{
class TllmRuntime;

namespace utils
{

int initDevice(WorldConfig const& worldConfig);

std::vector<uint8_t> loadEngine(std::string const& enginePath);

template <typename TInputContainer, typename TFunc>
auto transformVector(TInputContainer const& input, TFunc func)
    -> std::vector<std::remove_reference_t<decltype(func(input.front()))>>
{
    std::vector<std::remove_reference_t<decltype(func(input.front()))>> output{};
    output.reserve(input.size());
    std::transform(input.begin(), input.end(), std::back_inserter(output), func);
    return output;
}

std::vector<ITensor::SharedPtr> createBufferVector(TllmRuntime const& runtime, SizeType32 indexOffset,
    SizeType32 numBuffers, std::string const& prefix, MemoryType memType);

std::vector<ITensor::SharedPtr> createBufferVector(
    TllmRuntime const& runtime, SizeType32 numBuffers, MemoryType memType, nvinfer1::DataType dtype);

void reshapeBufferVector(std::vector<ITensor::SharedPtr>& vector, nvinfer1::Dims const& shape);

void assertNoVGQA(ModelConfig const& modelConfig, WorldConfig const& worldConfig);

std::vector<ITensor::SharedPtr> sliceBufferVector(
    std::vector<ITensor::SharedPtr> const& vector, SizeType32 offset, SizeType32 size);

void insertTensorVector(StringPtrMap<ITensor>& map, std::string const& key, std::vector<ITensor::SharedPtr> const& vec,
    SizeType32 indexOffset, std::vector<ModelConfig::LayerType> const& layerTypes, ModelConfig::LayerType type);

void insertTensorSlices(
    StringPtrMap<ITensor>& map, std::string const& key, ITensor::SharedPtr const& tensor, SizeType32 indexOffset);

void printTensorMap(std::ostream& stream, StringPtrMap<ITensor> const& map);

void setRawPointers(ITensor& pointers, ITensor::SharedPtr const& input, int32_t pointersSlot, int32_t inputSlot);

void setRawPointers(ITensor& pointers, ITensor::SharedPtr const& input);

void scatterBufferReplace(ITensor::SharedPtr& tensor, SizeType32 beamWidth, BufferManager& manager);

void tileBufferReplace(ITensor::SharedPtr& tensor, SizeType32 beamWidth, BufferManager& manager);

void tileCpuBufferReplace(ITensor::SharedPtr& tensor, SizeType32 beamWidth);

}
}
