#include "ub_interface.h"
#if ENABLE_MULTI_DEVICE
namespace sugesstify::runtime::ub
{
void ub_initialize(int tp)
{
    UserBufferAllocator::Instance().initialize(tp);
}

bool ub_is_initialized()
{
    return UserBufferAllocator::Instance().is_initialized();
}

void* ub_allocate(int idx, size_t bytes)
{
    return UserBufferAllocator::Instance().allocate(idx, bytes);
}

void ub_deallocate(void* addr)
{
    UserBufferAllocator::Instance().deallocate(addr);
}

UBBuffer ub_get(int idx)
{
    return UserBufferAllocator::Instance().get(idx);
}

communicator* ub_comm()
{
    return UserBufferAllocator::Instance().comm();
}

bool ub_supported()
{
    return true;
}
};

namespace sugesstify::kernels::ub
{
using namespace sugesstify::runtime::ub;

void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    allreduce2_userbuff_inplace_impl(handler, offset, elements, dataType, comm, stream);
}

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    return allgather2_userbuff_residual_impl(handler, offset, elements, hidden_size, residual, dataType, comm, stream);
}

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream)
{
    return allreduce2_userbuff_inplace_rmsnorm_quant_impl(handler, offset, out_handler, out_offset, elements,
        hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, dataType, comm, stream);
}
}
#else
namespace sugesstify::runtime::ub
{
void ub_initialize(int tp) {}

bool ub_is_initialized()
{
    return false;
}

void* ub_allocate(int idx, size_t bytes)
{
    return nullptr;
}

void ub_deallocate(void* addr) {}

UBBuffer ub_get(int idx)
{
    return UBBuffer();
}

communicator* ub_comm()
{
    return nullptr;
}

bool ub_supported()
{
    return false;
}
};

namespace sugesstify::kernels::ub
{
using namespace sugesstify::runtime::ub;

void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
}

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    return 0;
}

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream)
{
    return 0;
}
}
#endif
