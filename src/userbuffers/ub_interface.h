#pragma once
#include "cuda_runtime.h"
#include "../common/cudaUtils.h"
#include "../common/dataType.h"
#include "ub_allocator.h"

namespace sugesstify::runtime::ub
{
void ub_initialize(int tp);
bool ub_is_initialized();
void* ub_allocate(int idx, size_t bytes);
void ub_deallocate(void* addr);
UBBuffer ub_get(int idx);
communicator* ub_comm();
bool ub_supported();
};

namespace sugesstify::kernels::ub
{
using namespace sugesstify::runtime::ub;
void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream = 0);

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream);

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream);
}
