
#pragma once

#include "../runtime/cudaEvent.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace sugesstify::batch_manager
{

class ContextProgress
{
public:
    ContextProgress(int numLayers);

    void recordEvent(int layerIdx, cudaStream_t stream);

    void wait(int layerIdx);

    int getNumLayers() const
    {
        return mCudaEvents.size();
    }

    cudaEvent_t getEvent(int layerIdx)
    {
        return mCudaEvents.at(layerIdx).get();
    }

private:
    std::mutex mMutex;
    std::condition_variable mConditionVariable;
    std::unique_ptr<std::atomic_bool[]> mCudaEventsRecorded;
    std::vector<runtime::CudaEvent> mCudaEvents;
};

}
