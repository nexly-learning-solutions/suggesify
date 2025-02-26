
#include "workerPool.h"
#include "suggestify/common/cudaUtils.h"

namespace suggestify::runtime
{
WorkerPool::WorkerPool(std::size_t numWorkers, std::int32_t deviceId)
{
    for (std::size_t i = 0; i < numWorkers; ++i)
    {
        mWorkers.emplace_back(
            [this, deviceId]
            {
                if (deviceId >= 0)
                {
                    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                }
                else
                {
                    TLLM_LOG_WARNING("WorkerPool did not set cuda device");
                }

                while (true)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->mQueueMutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->mTasks.empty(); });
                        if (this->stop && this->mTasks.empty())
                        {
                            return;
                        }
                        task = std::move(this->mTasks.front());
                        this->mTasks.pop();
                    }

                    task();
                }
            });
    }
}

WorkerPool::~WorkerPool()
{
    {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : mWorkers)
    {
        worker.join();
    }
}
}
