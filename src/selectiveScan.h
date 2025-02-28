
#pragma once

#include "../common/assert.h"
#include "../common/cudaDriverWrapper.h"
#include "../common/cudaUtils.h"

namespace sugesstify
{
namespace kernels
{

struct SSMParamsBase
{
    int batch, dim, dstate, dt_rank, nheads, ngroups, chunk_size;
    int max_seqlen;
    int num_tokens;
    bool remove_padding;
    bool delta_softplus;
    bool is_mamab2;

    void* __restrict__ A_ptr;
    void* __restrict__ BC_ptr;
    void* __restrict__ D_ptr;
    void* __restrict__ u_ptr;
    void* __restrict__ delta_ptr;
    void* __restrict__ delta_bias_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ x_ptr;
    void* __restrict__ z_ptr;
    void* __restrict__ Os_ptr;
    void* __restrict__ St_ptr;
    void* __restrict__ dc_ptr;
    void* __restrict__ dA_ptr;
    void* __restrict__ CB_ptr;
    void* __restrict__ desc_ptr;
    int const* __restrict__ last_token_ids_ptr;
    int const* __restrict__ slot_mapping_ptr;
};


template <typename input_t, typename weight_t>
void invokeSelectiveScan(SSMParamsBase& params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void invokeChunkScan(SSMParamsBase& params, cudaStream_t stream, sugesstify::common::CUDADriverWrapper* driver);

template <typename input_t, typename weight_t>
void invokeSelectiveScanUpdate(SSMParamsBase& params, cudaStream_t stream);
}
}
