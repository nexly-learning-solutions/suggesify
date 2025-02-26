#include "../plugins/common/plugin.h"

#include "suggestify/common/mpiUtils.h"

#include "checkMacrosPlugin.h"
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
    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
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
    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return id;
}
}

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group)
{
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
    static std::map<std::set<int>, std::weak_ptr<ncclComm_t>> commMap;
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
        auto ncclComm = it->second.lock();
        if (ncclComm)
        {
            TLLM_LOG_TRACE("NCCL comm for group(%s) is cached for rank %d", groupStr.c_str(), rank);
            return ncclComm;
        }
    }

    TLLM_LOG_TRACE("Init NCCL comm for group(%s) for rank %d", groupStr.c_str(), rank);
    ncclUniqueId id = getUniqueId(group);
    int groupRank = 0;
    for (auto const& currentRank : group)
    {
        if (rank == currentRank)
            break;
        ++groupRank;
    }
    TLLM_CHECK(groupRank < group.size());
    std::shared_ptr<ncclComm_t> ncclComm(new ncclComm_t,
        [](ncclComm_t* comm)
        {
            ncclCommDestroy(*comm);
            delete comm;
        });
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    NCCLCHECK(ncclCommInitRank(ncclComm.get(), group.size(), id, groupRank));
    commMap[group] = ncclComm;
    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return ncclComm;
}
#endif

void const* suggestify::plugins::getCommSessionHandle()
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
        TLLM_CUDA_CHECK(cudaFree(nullptr));
        err = cuCtxGetCurrent(&ctx);
    }
    TLLM_CHECK(err == CUDA_SUCCESS);
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
            TLLM_CUDA_CHECK(cublasCreate(handle.get()));
            return handle;
        },
        [](cublasHandle_t* handle)
        {
            TLLM_CUDA_CHECK(cublasDestroy(*handle));
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
            TLLM_CUDA_CHECK(cublasLtCreate(handle.get()));
            return handle;
        },
        [](cublasLtHandle_t* handle)
        {
            TLLM_CUDA_CHECK(cublasLtDestroy(*handle));
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

PluginFieldParser::PluginFieldParser(int32_t nbFields, nvinfer1::PluginField const* fields)
    : mFields{fields}
{
    for (int32_t i = 0; i < nbFields; i++)
    {
        mMap.emplace(fields[i].name, PluginFieldParser::Record{i});
    }
}

PluginFieldParser::~PluginFieldParser()
{
    for (auto const& [name, record] : mMap)
    {
        if (!record.retrieved)
        {
            std::stringstream ss;
            ss << "unused plugin field with name: " << name;
            suggestify::plugins::logError(ss.str().c_str(), __FILE__, FN_NAME, __LINE__);
        }
    }
}

template <typename T>
nvinfer1::PluginFieldType toFieldType();
#define SPECIALIZE_TO_FIELD_TYPE(T, type)                                                                              \
    template <>                                                                                                        \
    nvinfer1::PluginFieldType toFieldType<T>()                                                                         \
    {                                                                                                                  \
        return nvinfer1::PluginFieldType::type;                                                                        \
    }
SPECIALIZE_TO_FIELD_TYPE(half, kFLOAT16)
SPECIALIZE_TO_FIELD_TYPE(float, kFLOAT32)
SPECIALIZE_TO_FIELD_TYPE(double, kFLOAT64)
SPECIALIZE_TO_FIELD_TYPE(int8_t, kINT8)
SPECIALIZE_TO_FIELD_TYPE(int16_t, kINT16)
SPECIALIZE_TO_FIELD_TYPE(int32_t, kINT32)
SPECIALIZE_TO_FIELD_TYPE(char, kCHAR)
SPECIALIZE_TO_FIELD_TYPE(nvinfer1::Dims, kDIMS)
SPECIALIZE_TO_FIELD_TYPE(void, kUNKNOWN)
#undef SPECIALIZE_TO_FIELD_TYPE

template <typename T>
std::optional<T> PluginFieldParser::getScalar(std::string_view const& name)
{
    auto const iter = mMap.find(name);
    if (iter == mMap.end())
    {
        return std::nullopt;
    }
    auto& record = mMap.at(name);
    auto const& f = mFields[record.index];
    TLLM_CHECK(toFieldType<T>() == f.type && f.length == 1);
    record.retrieved = true;
    return std::optional{*static_cast<T const*>(f.data)};
}

#define INSTANTIATE_PluginFieldParser_getScalar(T)                                                                     \
    template std::optional<T> PluginFieldParser::getScalar(std::string_view const&)
INSTANTIATE_PluginFieldParser_getScalar(half);
INSTANTIATE_PluginFieldParser_getScalar(float);
INSTANTIATE_PluginFieldParser_getScalar(double);
INSTANTIATE_PluginFieldParser_getScalar(int8_t);
INSTANTIATE_PluginFieldParser_getScalar(int16_t);
INSTANTIATE_PluginFieldParser_getScalar(int32_t);
INSTANTIATE_PluginFieldParser_getScalar(char);
INSTANTIATE_PluginFieldParser_getScalar(nvinfer1::Dims);
#undef INSTANTIATE_PluginFieldParser_getScalar

template <typename T>
std::optional<std::set<T>> PluginFieldParser::getSet(std::string_view const& name)
{
    auto const iter = mMap.find(name);
    if (iter == mMap.end())
    {
        return std::nullopt;
    }
    auto& record = mMap.at(name);
    auto const& f = mFields[record.index];
    TLLM_CHECK(toFieldType<T>() == f.type);
    std::set<T> group;
    auto const* r = static_cast<T const*>(f.data);
    for (int j = 0; j < f.length; ++j)
    {
        group.insert(*r);
        ++r;
    }

    record.retrieved = true;
    return std::optional{group};
}

#define INSTANTIATE_PluginFieldParser_getVector(T)                                                                     \
    template std::optional<std::set<T>> PluginFieldParser::getSet(std::string_view const&)
INSTANTIATE_PluginFieldParser_getVector(int32_t);
#undef INSTANTIATE_PluginFieldParser_getVector
