
#pragma once

#include "../common/mpiUtils.h"
#include "cudaStream.h"
#include "iBuffer.h"
#include "worldConfig.h"

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

namespace suggestify::runtime
{

class NcclCommunicator
{
public:
    explicit NcclCommunicator(ncclComm_t comm)
        : mComm{comm} {};

    explicit NcclCommunicator(int worldSize, int rank, mpi::MpiComm const& mpiComm = COMM_SESSION)
        : mComm{createComm(worldSize, rank, mpiComm)} {};

    explicit NcclCommunicator(WorldConfig const& worldConfig, mpi::MpiComm const& mpiComm = COMM_SESSION)
        : NcclCommunicator{worldConfig.getSize(), worldConfig.getRank(), mpiComm} {};

    ~NcclCommunicator();

    NcclCommunicator(NcclCommunicator const&) = delete;
    NcclCommunicator& operator=(NcclCommunicator const&) = delete;

    void send(IBuffer const& buf, int peer, CudaStream const& stream) const
    {
        send(buf.data(), buf.getSize(), buf.getDataType(), peer, stream);
    }

    void receive(IBuffer& buf, int peer, CudaStream const& stream) const
    {
        receive(buf.data(), buf.getSize(), buf.getDataType(), peer, stream);
    }

private:
    void send(
        void const* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const;

    void receive(void* sendbuff, size_t count, nvinfer1::DataType dataType, int peer, CudaStream const& stream) const;

    static ncclComm_t createComm(int worldSize, int rank, mpi::MpiComm const& mpiComm);

    ncclComm_t mComm;
};

}
