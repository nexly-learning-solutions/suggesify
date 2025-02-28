

#include "../common/cudaTypeUtils.cuh"
#include "../src/lookupKernels.h"

using namespace suggestify::common;

namespace suggestify
{
namespace kernels
{
template <typename Tout, typename Tin, typename Idx>
__global__ void lookup_kernel(Tout* output, Idx const* input, Tin const* weight, int64_t const token_num,
    Idx const offset, Idx const size, Idx const n_embed, Tout const* perTokenScales)
{
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < token_num * n_embed;
         index += blockDim.x * gridDim.x)
    {
        int64_t const word_index = input[index / n_embed] - offset;
        Idx const col_index = index % n_embed;
        Tout embedding;
        if (word_index < 0 || word_index >= size)
        {
            embedding = Tout(0.f);
        }
        else
        {
            embedding = (Tout) weight[word_index * n_embed + col_index];
            if (perTokenScales != nullptr)
            {
                embedding *= perTokenScales[word_index];
            }
        }
        output[index] = embedding;
    } // end for index
}

template <typename Tout, typename Tin, typename Idx>
void invokeLookUp(Tout* out, Idx const* input, Tin const* weight, int64_t const token_num, Idx const offset,
    Idx const size, Idx const n_embed, Tout const* perTokenScales, cudaStream_t stream)
{
    int64_t constexpr max_block_num = 65536;
    Idx constexpr max_block_size = 512;
    dim3 grid(min(token_num, max_block_num));
    dim3 block(min(n_embed, max_block_size));
    lookup_kernel<Tout, Tin, Idx>
        <<<grid, block, 0, stream>>>(out, input, weight, token_num, offset, size, n_embed, perTokenScales);
}

#define INSTANTIATE_LOOK_UP(Tout, Tin, Idx)                                                                            \
    template void invokeLookUp<Tout, Tin, Idx>(Tout * out, Idx const* input, Tin const* weight,                        \
        int64_t const token_num, Idx const offset, Idx const size, Idx const n_embed, Tout const* perTokenScales,      \
        cudaStream_t stream)

INSTANTIATE_LOOK_UP(float, float, int);
INSTANTIATE_LOOK_UP(float, int8_t, int);
INSTANTIATE_LOOK_UP(half, half, int);
INSTANTIATE_LOOK_UP(half, int8_t, int);

#ifdef ENABLE_BF16
INSTANTIATE_LOOK_UP(__nv_bfloat16, __nv_bfloat16, int);
INSTANTIATE_LOOK_UP(__nv_bfloat16, int8_t, int);
#endif

} // namespace kernels
} // namespace suggestify
