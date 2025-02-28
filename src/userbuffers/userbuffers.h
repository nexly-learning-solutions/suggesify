#pragma once
#include "cuda_runtime.h"
#include "../common/cudaUtils.h"
#include "../common/dataType.h"
#include "../common/envUtils.h"
#include "../common/mpiUtils.h"
#include <cuda.h>
#if defined(__aarch64__) || defined(_M_ARM64)
#define MNNVL
#endif

#define MAX_REGIONS 16
#define MAX_SMS 32
#define MAX_OPS 32
#define MAX_PEERS 8192
#define MAX_REQUESTS 1024
#define LAUNCH_GPU 1
#define LAUNCH_CPU 2
#define MAX_NVLINK 32

#define UB_MEM_UC_CONTIG 1
#define UB_MEM_MC_CREATED 2
#define UB_MEM_ALLOCATED 4

#define REG0_OPFLAGS (MAX_PEERS * 2)
#define REG0_RECV (REG0_OPFLAGS * userbuffers_op_types)
#define REG0_SINGLENODE (2 * MAX_NVLINK * MAX_SMS + MAX_OPS)
#define REG0_OFFSET(comm) ((2 * MAX_REGIONS) * MAX_NVLINK + REG0_SINGLENODE * 2 + MAX_PEERS)
#define REG0_ONESHOT_MAX 32 * 1024
#define REG0_ONESHOT_BUFFER (MAX_NVLINK * REG0_ONESHOT_MAX)
#define REG0_COMMBUFFER (REG0_ONESHOT_BUFFER * 2)
#define REG0_FLAGS (REG0_RECV + MAX_PEERS * MAX_REGIONS * 3)

namespace sugesstify::runtime::ub
{
enum req_type
{
    userbuffers_allreduceop_sharp,
    userbuffers_sendop,
    userbuffers_allreduceop_nonsharp,
    userbuffers_allreduceop_nonsharp2,
    userbuffers_alltoall,
    userbuffers_op_types
};

struct communicator
{
    int myrank, nranks;
    int nvrank, nvsize;
    int free_region;

    int launch_mode;

    void* gpu_ptrs;
    int sms, threads;
    int use_rr_kernel;
    int cga_size;
    int push, use_ce;

    void* mem_ptr[MAX_REGIONS];
    void** peer_ptr[MAX_REGIONS];

    int memflags[MAX_REGIONS];

    CUmemGenericAllocationHandle* uchandles[MAX_REGIONS];
    void* ucbase_ptr[MAX_REGIONS];
    size_t mem_size[MAX_REGIONS];

    void* mc_ptr[MAX_REGIONS];
    void* mc_baseptr;
    CUmemGenericAllocationHandle mc_handle;
    size_t mc_offset, mc_maxsize;
    int use_mc;

    int ar_nvsize, ar_firstgpu,
        ar_nvrank;
    int ar2_nvsize, ar2_firstgpu, ar2_nvrank;
    int pipe_id;
    int sm_arch;
    int oneshot, pdl_launch;

    MPI_Comm comm_world,
        comm_inter,
        comm_intra;
    int ibnvsize;

    int *send_id, *recv_id;
    int mydev;
};
typedef struct communicator communicator;

int create_communicator(communicator** comm);

int create_communicator_grouped(communicator** comm, int pipegpus, int pipenodes);
int create_communicator_grouped2(communicator** comm, int pipegpus, int pipenodes, int tensorgpus, int tensornodes);

int pipe_rank(communicator* comm,
    int step);

int register_user_buffer_collective(void** gpubuff, size_t bytes, communicator* comm, bool alloc = false);

void destroy_communicator(communicator* comm);
}

namespace sugesstify::kernels::ub
{
using namespace sugesstify::runtime::ub;
void allreduce2_userbuff_inplace_impl(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream = 0);

int allgather2_userbuff_residual_impl(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream);

int allreduce2_userbuff_inplace_rmsnorm_quant_impl(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream);
}
