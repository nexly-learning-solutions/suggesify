
#pragma once

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace suggestify::runtime
{

class WorkerPool
{
public:
    explicit WorkerPool(std::size_t numWorkers = 1, std::int32_t deviceId = -1);

    WorkerPool(WorkerPool const&) = delete;
    WorkerPool(WorkerPool&&) = delete;
    WorkerPool& operator=(WorkerPool const&) = delete;
    WorkerPool& operator=(WorkerPool&&) = delete;
    ~WorkerPool();

    template <class F>
    auto enqueue(F&& task) -> std::future<typename std::invoke_result<F>::type>
    {
        using returnType = typename std::invoke_result<F>::type;
        auto const taskPromise = std::make_shared<std::promise<returnType>>();
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            mTasks.push(
                [task = std::forward<F>(task), taskPromise]()
                {
                    try
                    {
                        if constexpr (std::is_void_v<returnType>)
                        {
                            task();
                            taskPromise->set_value();
                        }
                        else
                        {
                            taskPromise->set_value(task());
                        }
                    }
                    catch (...)
                    {
                        taskPromise->set_exception(std::current_exception());
                    }
                });
        }
        condition.notify_one();
        return taskPromise->get_future();
    }

private:
    std::vector<std::thread> mWorkers;
    std::queue<std::function<void()>> mTasks;

    std::mutex mQueueMutex;
    std::condition_variable condition;
    bool stop{};
};

}
