#pragma once

#include <deque>
#include <mutex>
#include <functional>
#include <future>
#include <thread>
#include <vector>
#include <condition_variable>

/// <summary>
/// A class representing a thread pool.
/// </summary>
class ThreadPool {
public:
    /// <summary>
    /// Constructs a ThreadPool with the default number of threads.
    /// </summary>
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

    /// <summary>
    /// Constructs a ThreadPool with the specified number of threads.
    /// </summary>
    /// <param name="numThreads">The number of threads to create in the pool.</param>
    ThreadPool(size_t numThreads) : stop(false) {
        // Create the specified number of threads.
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this]() {
                // Run a loop that continuously checks for tasks.
                while (true) {
                    std::function<void()> task;
                    {
                        // Lock the queue mutex.
                        std::unique_lock<std::mutex> lock(queueMutex);
                        // Wait on the condition variable until a task is available or the pool is stopped.
                        condition.wait(lock, [this]() { return stop || !tasks.empty(); });
                        // If the pool is stopped and there are no tasks, exit the loop.
                        if (stop && tasks.empty())
                            return;
                        // Get the next task from the queue.
                        task = std::move(tasks.front());
                        tasks.pop_front();
                    }
                    // Execute the task.
                    task();
                }
                });
        }
    }

    /// <summary>
    /// Destructor for the ThreadPool.
    /// </summary>
    ~ThreadPool() {
        // Lock the queue mutex.
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            // Set the stop flag to true.
            stop = true;
        }
        // Notify all waiting threads that the pool is stopped.
        condition.notify_all();
        // Join all threads in the pool.
        for (auto& thread : threads) {
            thread.join();
        }
    }

    /// <summary>
    /// Adds a new task to the thread pool.
    /// </summary>
    /// <typeparam name="Function">The type of the task function.</typeparam>
    /// <param name="func">The task function to add to the pool.</param>
    template <typename Function>
    void AddTask(Function&& func) {
        // Lock the queue mutex.
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            // Add the task to the queue.
            tasks.emplace_back([func]() { func(); });
        }
        // Notify one waiting thread that a task is available.
        condition.notify_one();
    }

    /// <summary>
    /// Enqueues a task to the thread pool and returns a future to its result.
    /// </summary>
    /// <typeparam name="F">The type of the task function.</typeparam>
    /// <typeparam name="Args">The types of the task arguments.</typeparam>
    /// <param name="f">The task function to enqueue.</param>
    /// <param name="args">The arguments to pass to the task function.</param>
    /// <returns>A future representing the result of the task.</returns>
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        // Define the return type of the task function.
        using return_type = typename std::invoke_result_t<F, Args...>;
        // Create a shared pointer to a packaged task that binds the task function and arguments.
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        // Get a future to the result of the task.
        std::future<return_type> result = task->get_future();
        // Lock the queue mutex.
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            // Add the task to the queue.
            tasks.emplace_back([task]() { (*task)(); });
        }
        // Notify one waiting thread that a task is available.
        condition.notify_one();
        // Return the future to the result.
        return result;
    }

    /// <summary>
    /// Executes the given function in the thread pool.
    /// </summary>
    /// <typeparam name="Function">The type of the function to execute.</typeparam>
    /// <param name="func">The function to execute.</param>
    template <typename Function>
    void Execute(Function&& func) {
        // Add the function as a task to the thread pool.
        AddTask(std::forward<Function>(func));
    }

    /// <summary>
    /// Shuts down the thread pool and waits for all tasks to complete.
    /// </summary>
    void shutdown() {
        // Lock the queue mutex.
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            // Set the stop flag to true.
            stop = true;
        }
        // Notify all waiting threads that the pool is stopped.
        condition.notify_all();
        // Join all threads in the pool.
        for (auto& thread : threads) {
            thread.join();
        }
    }

private:
    // Vector of threads in the pool.
    std::vector<std::thread> threads;
    // Queue of tasks waiting to be executed.
    std::deque<std::function<void()>> tasks;
    // Mutex for synchronizing access to the task queue.
    std::mutex queueMutex;
    // Condition variable for waiting on tasks.
    std::condition_variable condition;
    // Flag indicating whether the thread pool should stop.
    bool stop;
};
