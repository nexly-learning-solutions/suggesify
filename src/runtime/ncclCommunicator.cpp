
#include "ncclCommunicator.h"

#include "../common/logger.h"
#include "utils/multiDeviceUtils.h"

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

using namespace suggestify::runtime;

namespace
{
#if ENABLE_MULTI_DEVICE

ncclDataType_t toNcclType(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: return ncclFloat32;
    case nvinfer1::DataType::kHALF: return ncclHalf;
    case nvinfer1::DataType::kINT8: return ncclInt8;
    case nvinfer1::DataType::kINT32: return ncclInt32;
    case nvinfer1::DataType::kUINT8: return ncclUint8;
    case nvinfer1::DataType::kINT64: return ncclInt64;
    case nvinfer1::DataType::kFP8: return ncclUint8;
#if ENABLE_BF16
    case nvinfer1::DataType::kBF16: return ncclBfloat16;
#endif
    default: THROW("Unsupported data type: %d", static_cast<int>(dataType));
    }
}
#endif
}

void NcclCommunicator::send(
    void const* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    NCCL_CHECK(ncclSend(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
#else
    THROW("Multi device support is disabled.");
#endif
}

void NcclCommunicator::receive(
    void* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const
{
#if ENABLE_MULTI_DEVICE
    NCCL_CHECK(ncclRecv(sendbuff, count, toNcclType(dataType), peer, mComm, stream.get()));
#else
    THROW("Multi device support is disabled.");
#endif
}

ncclComm_t NcclCommunicator::createComm(int worldSize, int rank, mpi::MpiComm const& mpiComm)
{
#if ENABLE_MULTI_DEVICE

    ncclUniqueId id;
    if (rank == 0)
    {
        ncclGetUniqueId(&id);
    }
    mpiComm.bcastValue(id, 0);
    ncclComm_t comm;
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    NCCL_CHECK(ncclCommInitRank(&comm, worldSize, id, rank));
    return comm;
#else
    return nullptr;
#endif
}

NcclCommunicator::~NcclCommunicator()
{
#if ENABLE_MULTI_DEVICE
    if (mComm && ncclCommDestroy(mComm) != ncclSuccess)
    {
        LOG_WARNING("Failed to destroy NCCL communicator.");
    }
#endif
}
