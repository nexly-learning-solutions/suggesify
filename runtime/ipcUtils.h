

#pragma once

#include "common.h"
#include "../kernels/customAllReduceKernels.h"
#include "bufferManager.h"
#include "iTensor.h"
#include "worldConfig.h"

namespace suggestify::runtime
{

class IpcMemory
{
public:
    using BufferPtr = IBuffer::SharedPtr;

    size_t static constexpr FLAGS_SIZE = (suggestify::kernels::MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);

    IpcMemory(
        std::size_t bufferSize, BufferManager const& manager, WorldConfig const& worldConfig, bool openIpc = true);
    ~IpcMemory();

    IpcMemory(IpcMemory const&) = delete;
    IpcMemory& operator=(IpcMemory const&) = delete;

    IpcMemory(IpcMemory&&) = default;
    IpcMemory& operator=(IpcMemory&&) = default;

    [[nodiscard]] std::vector<void*> const& getCommPtrs() const
    {
        return mCommPtrs;
    }

private:
    void allocateIpcMemory(std::size_t bufferSize, BufferManager const& manager, WorldConfig const& worldConfig);
    void destroyIpcMemory();

    SizeType32 mTpRank;
    std::vector<void*> mCommPtrs;
    BufferPtr mBuffer;
    bool mOpenIpc;
};

class AllReduceBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;

    AllReduceBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxSequenceLength,
        SizeType32 hiddenSize, BufferManager const& manager, WorldConfig const& worldConfig,
        bool const fakeBuffers = false);

    TensorPtr mAllReduceCommPtrs;
    std::vector<runtime::IpcMemory> mIpcMemoryHandles;
};

void lamportInitializeAll(void* buffer_0, void* buffer_1, void* buffer_2, size_t size);

}
