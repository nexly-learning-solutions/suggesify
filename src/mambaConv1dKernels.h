
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"

namespace suggestify
{
namespace kernels
{

struct MambaConv1dParamsBase
{
    int batch, dim, max_seqlen, dconv, pre_stride, post_stride;
    bool remove_padding;
    bool apply_silu;
    void* __restrict__ in_ptr;
    void* state_in_ptr;
    void* state_out_ptr;
    void* __restrict__ weight_ptr;
    void* __restrict__ bias_ptr;
    void* __restrict__ out_ptr;
    int const* __restrict__ last_token_ids_ptr;
    int const* __restrict__ state_slot_mapping_ptr;
};


template <typename input_t>
void invokeMambaConv1dContext(MambaConv1dParamsBase& params, cudaStream_t stream);

template <typename input_t>
void invokeMambaConv1dGeneration(MambaConv1dParamsBase& params, cudaStream_t stream);

}
}
