#pragma once

#include "../src/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include <cuda_runtime_api.h>

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag,
    HopperGroupedGemmInput::EpilogueFusion FUSION, typename TileShape, typename ClusterShape, bool BIAS>
void sm90_generic_moe_gemm_kernelLauncher(HopperGroupedGemmInput hopper_input, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* kernel_occupancy, size_t* workspace_size);

}
}
}
