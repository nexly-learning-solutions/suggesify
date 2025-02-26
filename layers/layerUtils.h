
#pragma once

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "suggestify/common/assert.h"
#include "decodingParams.h"
#include "bufferManager.h"
#include "common.h"
#include "iBuffer.h"

namespace suggestify::layers
{

struct FillBuffers
{
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorConstPtr = runtime::ITensor::UniqueConstPtr;
    using BufferConstPtr = runtime::IBuffer::SharedConstPtr;

    template <typename T>
    void operator()(std::optional<std::vector<T>> const& optParam, T const defaultValue, BufferPtr hostBuffer,
        BufferPtr deviceBuffer, BufferConstPtr batchSlots, std::pair<float, float> const& limits,
        std::string const& name) const
    {
        auto hostBufferRange = runtime::BufferRange<T>(*hostBuffer);
        for (size_t bi = 0; bi < batchSize; ++bi)
        {
            auto value = defaultValue;
            auto batchSlot = runtime::bufferCast<runtime::SizeType32>(*batchSlots)[bi];
            if (optParam)
            {
                if (optParam->size() == 1)
                {
                    value = optParam->front();
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(optParam->size() == batchSize, "Argument vector size mismatch.");
                    value = optParam.value()[bi];
                }
            }
            TLLM_CHECK_WITH_INFO(limits.first < static_cast<float>(value) && static_cast<float>(value) <= limits.second,
                "%s param (%f) is out of limits (%f, %f]", name.c_str(), static_cast<float>(value), limits.first,
                limits.second);
            hostBufferRange[batchSlot] = value;
        }

        if (batchSlots)
        {
            auto const hostSlice = runtime::IBuffer::slice(hostBuffer, 0, maxBatchSize);
            auto deviceSlice = runtime::IBuffer::slice(deviceBuffer, 0, maxBatchSize);
            mBufferManager->copy(*hostSlice, *deviceSlice);
        }
        else
        {
            auto const hostSlice = runtime::IBuffer::slice(hostBuffer, 0, batchSize);
            auto deviceSlice = runtime::IBuffer::slice(deviceBuffer, 0, batchSize);
            mBufferManager->copy(*hostSlice, *deviceSlice);
        }
    }

    runtime::SizeType32 batchSize;
    runtime::SizeType32 maxBatchSize;
    std::shared_ptr<runtime::BufferManager> mBufferManager;
};

template <typename T>
bool allOfBatchSlots(runtime::SizeType32 const* batchSlotsHost, T const* data, runtime::SizeType32 batchSize, T value)
{
    return std::all_of(
        batchSlotsHost, batchSlotsHost + batchSize, [&](runtime::SizeType32 b) { return data[b] == value; });
}

template <typename T>
T maxOfBatchSlots(runtime::SizeType32 const* batchSlotsHost, T const* data, runtime::SizeType32 batchSize)
{
    return std::transform_reduce(
        batchSlotsHost, batchSlotsHost + batchSize, std::numeric_limits<T>::lowest(),
        [](auto a, auto b) { return std::max(a, b); }, [&](auto i) { return data[i]; });
}

inline DecoderDomain getLocalDecoderDomain(
    std::shared_ptr<BaseDecodingInputs> baseInputs, DecoderDomain const& globalDecoderDomain)
{
    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    runtime::SizeType32 batchSize{baseInputs->localBatchSize};
    runtime::SizeType32 beamWidth{0};
    runtime::SizeType32 vocabSize{0};
    if (inputs->logits)
    {
        auto const& logitsShape = inputs->logits.value()->getShape();
        TLLM_CHECK(logitsShape.nbDims == 3 || logitsShape.nbDims == 4);
        beamWidth = inputs->logits.value()->getDimension<-2>();
        vocabSize = inputs->logits.value()->getDimension<-1>();
    }
    else if (inputs->logitsVec)
    {
        TLLM_CHECK(inputs->logitsVec->size());
        auto const& logitsShape = inputs->logitsVec.value()[0]->getShape();
        TLLM_CHECK(logitsShape.nbDims == 3 || logitsShape.nbDims == 4);
        beamWidth = inputs->logitsVec.value()[0]->getDimension<-2>();
        vocabSize = inputs->logitsVec.value()[0]->getDimension<-1>();
    }
    else if (inputs->batchSlots)
    {
        beamWidth = globalDecoderDomain.getBeamWidth();
        vocabSize = globalDecoderDomain.getVocabSize();
    }
    else
    {
        TLLM_THROW("Can't get local Decoder domain");
    }
    return {batchSize, beamWidth, vocabSize};
}

template <typename... T>
runtime::SizeType32 expandMatchElements(runtime::SizeType32 expandSize, std::vector<T>&... vector)
{
    std::array vectorSizes{vector.size()...};

    bool allSingle = true;
    for (auto size : vectorSizes)
    {
        if (size == expandSize)
        {
            allSingle = false;
        }
        else if (size != 1)
        {
            return 0;
        }
    }

    if (allSingle)
    {
        return 1;
    }

    (vector.resize(expandSize, vector.front()), ...);
    return expandSize;
}

}
