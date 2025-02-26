#include "opUtils.h"
#include "mpiUtils.h"

#include "cuda.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <functional>
#include <mutex>
#include <thread>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if ENABLE_MULTI_DEVICE

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap()
{
    static std::unordered_map<nvinfer1::DataType, ncclDataType_t> dtypeMap = {{nvinfer1::DataType::kFLOAT, ncclFloat32},
        {nvinfer1::DataType::kHALF, ncclFloat16}, {nvinfer1::DataType::kBF16, ncclBfloat16}};
    return &dtypeMap;
}

namespace
{

ncclUniqueId getUniqueId(std::set<int> const& group) noexcept
{
    auto const rank = COMM_SESSION.getRank();
    LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
    ncclUniqueId id;
    if (rank == *group.begin())
    {
        NCCLCHECK(ncclGetUniqueId(&id));
        for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
        {
            COMM_SESSION.sendValue(id, *it, 0);
        }
    }
    else
    {
        COMM_SESSION.recvValue(id, *group.begin(), 0);
    }
    LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return id;
}
}

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group)
{
    auto const rank = COMM_SESSION.getRank();
    LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
    static std::map<std::set<int>, std::shared_ptr<ncclComm_t>> commMap;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream oss;
    int index = 0;
    for (auto const& rank : group)
    {
        if (index != 0)
        {
            oss << ",";
        }
        oss << rank;
        index++;
    }
    auto groupStr = oss.str();
    auto it = commMap.find(group);
    if (it != commMap.end())
    {
        auto ncclComm = it->second;
        LOG_TRACE("NCCL comm for group(%s) is cached for rank %d", groupStr.c_str(), rank);
        return ncclComm;
    }

    LOG_TRACE("Init NCCL comm for group(%s) for rank %d", groupStr.c_str(), rank);
    ncclUniqueId id = getUniqueId(group);
    int groupRank = 0;
    for (auto const& currentRank : group)
    {
        if (rank == currentRank)
            break;
        ++groupRank;
    }
    CHECK(groupRank < group.size());
    std::shared_ptr<ncclComm_t> ncclComm(new ncclComm_t,
        [](ncclComm_t* comm)
        {
            ncclCommDestroy(*comm);
            delete comm;
        });
    NCCLCHECK(ncclCommInitRank(ncclComm.get(), group.size(), id, groupRank));
    commMap[group] = ncclComm;
    LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return ncclComm;
}
#endif

void const* suggestify::common::getCommSessionHandle()
{
#if ENABLE_MULTI_DEVICE
    return &COMM_SESSION;
#else
    return nullptr;
#endif
}

namespace
{

inline CUcontext getCurrentCudaCtx()
{
    CUcontext ctx{};
    CUresult err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_ERROR_NOT_INITIALIZED || ctx == nullptr)
    {
        CUDA_CHECK(cudaFree(nullptr));
        err = cuCtxGetCurrent(&ctx);
    }
    CHECK(err == CUDA_SUCCESS);
    return ctx;
}

template <typename T>
class PerCudaCtxSingletonCreator
{
public:
    using CreatorFunc = std::function<std::unique_ptr<T>()>;
    using DeleterFunc = std::function<void(T*)>;

    PerCudaCtxSingletonCreator(CreatorFunc creator, DeleterFunc deleter)
        : mCreator{std::move(creator)}
        , mDeleter{std::move(deleter)}
    {
    }

    std::shared_ptr<T> operator()()
    {
        std::lock_guard<std::mutex> lk{mMutex};
        CUcontext ctx{getCurrentCudaCtx()};
        std::shared_ptr<T> result = mObservers[ctx].lock();
        if (result == nullptr)
        {
            result = std::shared_ptr<T>{mCreator().release(),
                [this, ctx](T* obj)
                {
                    if (obj == nullptr)
                    {
                        return;
                    }
                    mDeleter(obj);

                    std::shared_ptr<T> observedObjHolder;
                    std::lock_guard<std::mutex> lk{mMutex};
                    observedObjHolder = mObservers.at(ctx).lock();
                    if (observedObjHolder == nullptr)
                    {
                        mObservers.erase(ctx);
                    }
                }};
            mObservers.at(ctx) = result;
        }
        return result;
    }

private:
    CreatorFunc mCreator;
    DeleterFunc mDeleter;
    mutable std::mutex mMutex;
    std::unordered_map<CUcontext, std::weak_ptr<T>> mObservers;
};

template <typename T>
class PerThreadSingletonCreator
{
public:
    using CreatorFunc = std::function<std::unique_ptr<T>()>;
    using DeleterFunc = std::function<void(T*)>;

    PerThreadSingletonCreator(CreatorFunc creator, DeleterFunc deleter)
        : mCreator{std::move(creator)}
        , mDeleter{std::move(deleter)}
    {
    }

    std::shared_ptr<T> operator()()
    {
        std::lock_guard<std::mutex> lk{mMutex};

        std::thread::id thread = std::this_thread::get_id();
        std::shared_ptr<T> result = mObservers[thread].lock();

        if (result == nullptr)
        {
            result = std::shared_ptr<T>{mCreator().release(),
                [this, thread](T* obj)
                {
                    if (obj == nullptr)
                    {
                        return;
                    }
                    mDeleter(obj);

                    std::shared_ptr<T> observedObjHolder;
                    std::lock_guard<std::mutex> lk{mMutex};
                    observedObjHolder = mObservers.at(thread).lock();
                    if (observedObjHolder == nullptr)
                    {
                        mObservers.erase(thread);
                    }
                }};
            mObservers.at(thread) = result;
        }
        return result;
    }

private:
    CreatorFunc mCreator;
    DeleterFunc mDeleter;
    mutable std::mutex mMutex;
    std::unordered_map<std::thread::id, std::weak_ptr<T>> mObservers;
};

}

std::shared_ptr<cublasHandle_t> getCublasHandle()
{
    static PerThreadSingletonCreator<cublasHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasHandle_t>(new cublasHandle_t);
            CUDA_CHECK(cublasCreate(handle.get()));
            return handle;
        },
        [](cublasHandle_t* handle)
        {
            CUDA_CHECK(cublasDestroy(*handle));
            delete handle;
        });
    return creator();
}

std::shared_ptr<cublasLtHandle_t> getCublasLtHandle()
{
    static PerThreadSingletonCreator<cublasLtHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasLtHandle_t>(new cublasLtHandle_t);
            CUDA_CHECK(cublasLtCreate(handle.get()));
            return handle;
        },
        [](cublasLtHandle_t* handle)
        {
            CUDA_CHECK(cublasLtDestroy(*handle));
            delete handle;
        });
    return creator();
}

std::shared_ptr<suggestify::common::CublasMMWrapper> getCublasMMWrapper(std::shared_ptr<cublasHandle_t> cublasHandle,
    std::shared_ptr<cublasLtHandle_t> cublasltHandle, cudaStream_t stream, void* workspace)
{
    static PerThreadSingletonCreator<suggestify::common::CublasMMWrapper> creator(
        [cublasHandle, cublasltHandle, stream, workspace]() -> auto
        {
            auto wrapper = std::unique_ptr<suggestify::common::CublasMMWrapper>(
                new suggestify::common::CublasMMWrapper(cublasHandle, cublasltHandle, stream, workspace));
            return wrapper;
        },
        [](suggestify::common::CublasMMWrapper* wrapper) { delete wrapper; });
    return creator();
}
