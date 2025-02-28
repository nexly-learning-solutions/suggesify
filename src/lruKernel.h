
#pragma once

#include "../common/assert.h"

namespace suggestify
{
namespace kernels
{

struct lruParams
{
    int batch, width;
    int max_seqlen;
    int block_size;
    bool remove_padding;

    void* __restrict__ A_ptr;
    void* __restrict__ x_ptr;
    void* __restrict__ y_ptr;
    void* __restrict__ y_bias_ptr;
    void* __restrict__ gate_ptr;
    void* __restrict__ gate_bias_ptr;
    void* __restrict__ gate_x_ptr;
    void* __restrict__ gate_x_bias_ptr;
    void* __restrict__ gate_a_ptr;
    void* __restrict__ gate_a_bias_ptr;
    void* __restrict__ state_ptr;
    void* __restrict__ out_ptr;
    int const* __restrict__ last_token_ids_ptr;
    int const* __restrict__ slot_mapping_ptr;
};


template <typename T>
void invokeRGLRU(lruParams& params, cudaStream_t stream);

template <typename T>
void invokeRGLRUUpdate(lruParams& params, cudaStream_t stream);

}
}
