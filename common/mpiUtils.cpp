
#include <numeric>
#include <unordered_set>

#include "mpiUtils.h"

#include "assert.h"
#include "logger.h"
#include "suggestify/runtime/common.h"
#include "suggestify/runtime/iBuffer.h"

#include <csignal>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <type_traits>
#ifndef _WIN32
#include <unistd.h>
#endif

static_assert(std::is_same<suggestify::runtime::SizeType32, std::int32_t>::value);

namespace suggestify::mpi
{

MPI_Datatype getMpiDtype(MpiType dtype)
{
#if ENABLE_MULTI_DEVICE
    static std::unordered_map<MpiType, MPI_Datatype> const dtype_map{
        {MpiType::kBYTE, MPI_BYTE},
        {MpiType::kHALF, MPI_UINT16_T},
        {MpiType::kFLOAT, MPI_FLOAT},
        {MpiType::kDOUBLE, MPI_DOUBLE},
        {MpiType::kBOOL, MPI_C_BOOL},
        {MpiType::kINT8, MPI_INT8_T},
        {MpiType::kUINT8, MPI_UINT8_T},
        {MpiType::kINT32, MPI_INT32_T},
        {MpiType::kUINT32, MPI_UINT32_T},
        {MpiType::kINT64, MPI_INT64_T},
        {MpiType::kUINT64, MPI_UINT64_T},
        {MpiType::kFP8, MPI_UINT8_T},
        {MpiType::kBF16, MPI_UINT16_T},
        {MpiType::kCHAR, MPI_CHAR},
    };
    return dtype_map.at(dtype);
#else
    THROW("Multi device support is disabled.");
#endif
}

MPI_Op getMpiOp(MpiOp op)
{
#if ENABLE_MULTI_DEVICE
    static std::unordered_map<MpiOp, MPI_Op> const op_map{
        {MpiOp::NULLOP, MPI_OP_NULL},
        {MpiOp::MAX, MPI_MAX},
        {MpiOp::MIN, MPI_MIN},
        {MpiOp::SUM, MPI_SUM},
        {MpiOp::PROD, MPI_PROD},
        {MpiOp::LAND, MPI_LAND},
        {MpiOp::BAND, MPI_BAND},
        {MpiOp::LOR, MPI_LOR},
        {MpiOp::BOR, MPI_BOR},
        {MpiOp::LXOR, MPI_LXOR},
        {MpiOp::BXOR, MPI_BXOR},
        {MpiOp::MINLOC, MPI_MINLOC},
        {MpiOp::MAXLOC, MPI_MAXLOC},
        {MpiOp::REPLACE, MPI_REPLACE},
    };
    return op_map.at(op);
#else
    THROW("Multi device support is disabled.");
#endif
}

namespace
{

bool mpiInitialized = false;
std::recursive_mutex mpiMutex;

MpiComm initLocalSession()
{
#if ENABLE_MULTI_DEVICE
    MPI_Comm localComm = nullptr;
    MPI_Comm_split_type(COMM_SESSION, OMPI_COMM_TYPE_HOST, COMM_SESSION.getRank(), MPI_INFO_NULL, &localComm);
    MpiComm localSession{localComm, false};
#else
    MpiComm localSession{COMM_SESSION, false};
#endif
    return localSession;
}

}

std::vector<int> getWorldRanks(MpiComm const& comm)
{
#if ENABLE_MULTI_DEVICE
    MPI_Group group = nullptr;
    MPI_Group worldGroup = nullptr;

    MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
    MPICHECK(MPI_Comm_group(comm, &group));

    int groupSize = 0;
    MPICHECK(MPI_Group_size(group, &groupSize));
    std::vector<int> ranks(groupSize);
    std::vector<int> worldRanks(groupSize);
    std::iota(ranks.begin(), ranks.end(), 0);

    MPICHECK(MPI_Group_translate_ranks(group, groupSize, ranks.data(), worldGroup, worldRanks.data()));
    MPICHECK(MPI_Group_free(&group));
    MPICHECK(MPI_Group_free(&worldGroup));
#else
    std::vector<int> worldRanks{0};
#endif
    return worldRanks;
}

void initialize(MpiThreadSupport threadMode, bool forwardAbortToParent)
{
    if (mpiInitialized)
    {
        return;
    }
    std::lock_guard<std::recursive_mutex> lk(mpiMutex);
    if (mpiInitialized)
    {
        return;
    }
#if ENABLE_MULTI_DEVICE
    int initialized = 0;
    MPI_CHECK(MPI_Initialized(&initialized));
    if (!initialized)
    {
        LOG_INFO("Initializing MPI with thread mode %d", threadMode);
        int providedMode = 0;
        auto requiredMode = static_cast<int>(threadMode);
        MPICHECK(MPI_Init_thread(nullptr, nullptr, requiredMode, &providedMode));
        CHECK_WITH_INFO(providedMode >= requiredMode, "MPI_Init_thread failed");
        std::atexit([]() { MPI_Finalize(); });

        for (int sig : {SIGABRT, SIGSEGV})
        {
            __sighandler_t previousHandler = nullptr;
            if (forwardAbortToParent)
            {
                previousHandler = std::signal(sig,
                    [](int signal)
                    {
#ifndef _WIN32
                        pid_t parentProcessId = getppid();
                        kill(parentProcessId, SIGKILL);
#endif
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    });
            }
            else
            {
                previousHandler = std::signal(sig, [](int signal) { MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); });
            }
            CHECK_WITH_INFO(previousHandler != SIG_ERR, "Signal handler setup failed");
        }

        MpiComm::localSession();
        LOG_INFO("Initialized MPI");
    }
#endif
    mpiInitialized = true;
}

void MpiComm::barrier() const
{
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Barrier(mComm));
#else
    THROW("Multi device support is disabled.");
#endif
}

#if ENABLE_MULTI_DEVICE
template <typename TMpiFunc, typename TBase, typename... TArgs,
    typename = std::enable_if_t<std::is_same_v<void, std::remove_const_t<TBase>>>>
size_t invokeChunked(TMpiFunc func, TBase* buffer, size_t size, MPI_Datatype dtype, TArgs... args)
{
    constexpr auto maxP1 = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
    if (LIKELY(size < maxP1))
    {
        MPICHECK(func(buffer, size, dtype, args...));
        return 1;
    }

    constexpr size_t alignment = 256;
    int elementSize = 1;
    MPICHECK(MPI_Type_size(dtype, &elementSize));
    elementSize = std::min<int>(elementSize, alignment);

    auto const step = maxP1 - (alignment / elementSize);

    using TCast = std::conditional_t<std::is_const_v<TBase>, uint8_t const, uint8_t>;
    size_t count = 0;
    while (size != 0)
    {
        auto currentStep = static_cast<int>(std::min(size, step));
        MPICHECK(func(buffer, currentStep, dtype, args...));
        size -= currentStep;
        size_t diff = static_cast<size_t>(currentStep) * elementSize;
        buffer = static_cast<TCast*>(buffer) + diff;
        ++count;
    }

    return count;
}
#endif

std::shared_ptr<MpiRequest> MpiComm::bcastAsync(void* buffer, size_t size, MpiType dtype, int root) const
{
    std::shared_ptr<MpiRequest> r = std::make_shared<MpiRequest>();
#if ENABLE_MULTI_DEVICE
    invokeChunked(MPI_Ibcast, buffer, size, getMpiDtype(dtype), root, mComm, &r->mRequest);
#else
    THROW("Multi device support is disabled.");
#endif
    return r;
}

std::shared_ptr<MpiRequest> MpiComm::bcastAsync(runtime::IBuffer& buf, int root) const
{
    CHECK(buf.getMemoryType() != runtime::MemoryType::kGPU);
    return bcastAsync(buf.data(), buf.getSizeInBytes(), MpiType::kBYTE, root);
}

void MpiComm::bcast(void* buffer, size_t size, MpiType dtype, int root) const
{
#if ENABLE_MULTI_DEVICE
    invokeChunked(MPI_Bcast, buffer, size, getMpiDtype(dtype), root, mComm);
#else
    THROW("Multi device support is disabled.");
#endif
}

void MpiComm::bcast(runtime::IBuffer& buf, int root) const
{
    bcast(buf.data(), buf.getSizeInBytes(), MpiType::kBYTE, root);
}

std::shared_ptr<MpiRequest> MpiComm::sendAsync(void const* buffer, size_t size, MpiType dtype, int dest, int tag) const
{
    LOG_DEBUG("start MPI_Isend with size %d", size);
    std::shared_ptr<MpiRequest> r = std::make_shared<MpiRequest>();
#if ENABLE_MULTI_DEVICE
    invokeChunked(MPI_Isend, buffer, size, getMpiDtype(dtype), dest, tag, mComm, &r->mRequest);
#else
    THROW("Multi device support is disabled.");
#endif
    LOG_DEBUG("end MPI_Isend with size %d", size);
    return r;
}

std::shared_ptr<MpiRequest> MpiComm::sendAsync(runtime::IBuffer const& buf, int dest, int tag) const
{
    return sendAsync(buf.data(), buf.getSizeInBytes(), MpiType::kBYTE, dest, tag);
}

void MpiComm::send(void const* buffer, size_t size, MpiType dtype, int dest, int tag) const
{
    LOG_DEBUG("start MPI_Send with size %d", size);
#if ENABLE_MULTI_DEVICE
    invokeChunked(MPI_Send, buffer, size, getMpiDtype(dtype), dest, tag, mComm);
#else
    THROW("Multi device support is disabled.");
#endif
    LOG_DEBUG("end MPI_Send with size %d", size);
}

void MpiComm::send(runtime::IBuffer const& buf, int dest, int tag) const
{
    send(buf.data(), buf.getSizeInBytes(), MpiType::kBYTE, dest, tag);
}

MPI_Status MpiComm::recv(void* buffer, size_t size, MpiType dtype, int source, int tag) const
{
    LOG_DEBUG("start MPI_Recv with size %d", size);
    MPI_Status status{};
#if ENABLE_MULTI_DEVICE
    invokeChunked(MPI_Recv, buffer, size, getMpiDtype(dtype), source, tag, mComm, &status);
#else
    THROW("Multi device support is disabled.");
#endif
    LOG_DEBUG("end MPI_Recv with size %d", size);
    return status;
}

MPI_Status MpiComm::recv(runtime::IBuffer& buf, int source, int tag) const
{
    return recv(buf.data(), buf.getSizeInBytes(), MpiType::kBYTE, source, tag);
}

MpiComm MpiComm::split(int color, int key) const
{
    MPI_Comm splitComm = nullptr;
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Comm_split(mComm, color, key, &splitComm));
#else
    THROW("Multi device support is disabled.");
#endif
    return MpiComm{splitComm, true};
}

void MpiComm::allreduce(void const* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op) const
{
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Allreduce(sendbuf, recvbuf, count, getMpiDtype(dtype), getMpiOp(op), mComm));
#else
    THROW("Multi device support is disabled.");
#endif
}

void MpiComm::allgather(void const* sendbuf, void* recvbuf, int count, MpiType dtype) const
{
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Allgather(sendbuf, count, getMpiDtype(dtype), recvbuf, count, getMpiDtype(dtype), mComm));
#else
    THROW("Multi device support is disabled.");
#endif
}

void MpiComm::allgatherv(void const* sendbuf, int sendcount, MpiType sendtype, void* recvbuf,
    std::vector<int> const& recvcounts, std::vector<int> const& displs, MpiType recvtype) const
{
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Allgatherv(sendbuf, sendcount, getMpiDtype(sendtype), recvbuf, recvcounts.data(), displs.data(),
        getMpiDtype(recvtype), mComm));

#else
    THROW("Multi device support is disabled.");
#endif
}

void MpiComm::mprobe(int source, int tag, MPI_Message* msg, MPI_Status* status) const
{
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Mprobe(source, tag, mComm, msg, status));
#else
    THROW("Multi device support is disabled.");
#endif
}

bool MpiComm::improbe(int source, int tag, MPI_Message* msg, MPI_Status* status) const
{
#if ENABLE_MULTI_DEVICE
    int flag{0};
    MPICHECK(MPI_Improbe(source, tag, mComm, &flag, msg, status));
    return flag != 0;
#else
    THROW("Multi device support is disabled.");
    return false;
#endif
}

bool MpiComm::iprobe(int source, int tag, MPI_Status* status) const
{
#if ENABLE_MULTI_DEVICE
    int flag{0};
    MPICHECK(MPI_Iprobe(source, tag, mComm, &flag, status));
    return flag != 0;
#else
    THROW("Multi device support is disabled.");
    return false;
#endif
}

void MpiComm::recvPoll(int source, int tag, int periodMs) const
{
    MPI_Status status;
    while (!iprobe(source, tag, &status))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(periodMs));
    }
}

int MpiComm::getRank() const
{
    int rank = 0;
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Comm_rank(mComm, &rank));
#endif
    return rank;
}

int MpiComm::getSize() const
{
    int world_size = 1;
#if ENABLE_MULTI_DEVICE
    MPICHECK(MPI_Comm_size(mComm, &world_size));
#endif
    return world_size;
}

MpiComm const& MpiComm::world()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    static MpiComm commWorld{MPI_COMM_WORLD, false};
    initialize();
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return commWorld;
}

MpiComm& MpiComm::mutableSession()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    static MpiComm commSession{MPI_COMM_WORLD, false};
    initialize();
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return commSession;
}

MpiComm& MpiComm::mutableLocalSession()
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    static MpiComm localSession = initLocalSession();
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return localSession;
}

void MpiComm::refreshLocalSession()
{
#if ENABLE_MULTI_DEVICE
    static std::mutex mutex;
    std::unique_lock lock(mutex);
    auto initSessionRanks = getWorldRanks(MpiComm::session());
    auto localSessionRanks = getWorldRanks(MpiComm::localSession());

    std::vector<int> intersectionRanks;
    std::unordered_set<int> localSessionRanksSet(localSessionRanks.begin(), localSessionRanks.end());
    for (auto rank : initSessionRanks)
    {
        if (localSessionRanksSet.find(rank) != localSessionRanksSet.end())
        {
            intersectionRanks.push_back(rank);
        }
    }

    MPI_Group worldGroup = nullptr;
    MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
    MPI_Group localGroup = nullptr;
    MPICHECK(MPI_Group_incl(worldGroup, intersectionRanks.size(), intersectionRanks.data(), &localGroup));
    MPI_Comm localComm = nullptr;
    MPICHECK(MPI_Comm_create_group(MPI_COMM_WORLD, localGroup, intersectionRanks.front(), &localComm));
    MpiComm::mutableLocalSession().mFreeComm = true;
    MpiComm::mutableLocalSession() = MpiComm{localComm, false};
    LOG_INFO("Refreshed the MPI local session");
#endif
}

MpiComm::MpiComm(MPI_Comm g, bool freeComm)
    : mComm{g}
    , mFreeComm{freeComm}
{
    CHECK(mComm != MPI_COMM_NULL);
}

MpiComm::~MpiComm() noexcept
{
#if ENABLE_MULTI_DEVICE
    if (mFreeComm && mComm)
    {
        if (MPI_Comm_free(&mComm) != MPI_SUCCESS)
        {
            LOG_ERROR("MPI_Comm_free failed");
        }
    }
#endif
}

MpiComm::MpiComm(MpiComm&& comm) noexcept
    : mComm{comm.mComm}
    , mFreeComm{comm.mFreeComm}
{
    comm.mFreeComm = false;
}

MpiComm& MpiComm::operator=(MpiComm&& comm) noexcept
{
    this->~MpiComm();
    mComm = comm.mComm;
    mFreeComm = comm.mFreeComm;
    comm.mFreeComm = false;
    return *this;
}

MpiWaitThread::MpiWaitThread(std::string name, std::function<void()> funcWait, std::function<void()> funcSetup)
    : mName{name.c_str()}
    , mFuncWait{funcWait}
    , mFuncSetup{funcSetup}
{
    LOG_TRACE("%s: %s start", mName.c_str(), __PRETTY_FUNCTION__);
    mThread = std::make_unique<std::thread>(&MpiWaitThread::sideThread, this);
    LOG_TRACE("%s: %s stop", mName.c_str(), __PRETTY_FUNCTION__);
}

MpiWaitThread::~MpiWaitThread()
{
    LOG_TRACE("%s: %s start", mName.c_str(), __PRETTY_FUNCTION__);
    waitStop();
    mShouldExit.store(true);
    notifyStart();
    mThread->join();
    mThread.reset(nullptr);
    LOG_TRACE("%s: %s stop", mName.c_str(), __PRETTY_FUNCTION__);
}

void MpiWaitThread::sideThread()
{
    if (mFuncSetup)
    {
        mFuncSetup();
    }
    while (!mShouldExit.load())
    {
        notifyStop();
        waitStart();
        mFuncWait();
    }
}

void MpiWaitThread::waitStart()
{
    LOG_TRACE("%s: %s start", mName.c_str(), __PRETTY_FUNCTION__);
    std::unique_lock<std::mutex> lock(mMutex);
    mCondVar.wait(lock, [this] { return mRunning; });
    LOG_TRACE("%s: %s stop", mName.c_str(), __PRETTY_FUNCTION__);
}

void MpiWaitThread::waitStop()
{
    LOG_TRACE("%s: %s start", mName.c_str(), __PRETTY_FUNCTION__);
    std::unique_lock<std::mutex> lock(mMutex);
    mCondVar.wait(lock, [this] { return !mRunning; });
    LOG_TRACE("%s: %s stop", mName.c_str(), __PRETTY_FUNCTION__);
}

void MpiWaitThread::notifyStart()
{
    LOG_TRACE("%s: %s start", mName.c_str(), __PRETTY_FUNCTION__);
    std::lock_guard<std::mutex> lock(mMutex);
    mRunning = true;
    mCondVar.notify_one();
    LOG_TRACE("%s: %s stop", mName.c_str(), __PRETTY_FUNCTION__);
}

void MpiWaitThread::notifyStop()
{
    LOG_TRACE("%s: %s start", mName.c_str(), __PRETTY_FUNCTION__);
    std::lock_guard<std::mutex> lock(mMutex);
    mRunning = false;
    mCondVar.notify_one();
    LOG_TRACE("%s: %s stop", mName.c_str(), __PRETTY_FUNCTION__);
}

}
