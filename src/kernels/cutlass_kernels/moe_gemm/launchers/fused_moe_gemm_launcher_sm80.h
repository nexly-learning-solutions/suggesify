
namespace suggestify::kernels::cutlass_kernels
{
template <typename ElementType_, typename CutlassWeightType_, int MaxTileM_, int TileN_, int TileK_, int Stages_,
    typename EpilogueTag>
void sm80_generic_fused_moe_gemm_kernelLauncher(ElementType_ const* A, CutlassWeightType_ const* B,
    ElementType_ const* biases, bool bias_is_broadcast, ElementType_* C, int64_t const* total_tokens_including_expert,
    int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream,
    int* kernel_occupancy);
}
