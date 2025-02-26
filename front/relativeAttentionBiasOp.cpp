
#include "suggestify/kernels/buildRelativeAttentionBiasKernel.h"
#include "thUtils.h"

namespace th = torch;
namespace tl = suggestify;
namespace tk = suggestify::kernels;

namespace torch_ext
{

template <typename T>
void handleInvokeRelativeAttentionBias(th::Tensor& relative_attention_bias, th::Tensor& relative_attention_bias_table,
    int64_t const num_head, int64_t const max_seq_len, int64_t const num_bucket, bool const is_bidirectional,
    int64_t const max_distance, cudaStream_t stream)
{

    T* relative_attention_bias_ptr = get_ptr<T>(relative_attention_bias);
    T const* relative_attention_bias_table_ptr = get_ptr<T>(relative_attention_bias_table);

    tk::invokeBuildRelativeAttentionBias<T>(relative_attention_bias_ptr, relative_attention_bias_table_ptr, num_head,
        (max_seq_len + 1), num_bucket, is_bidirectional, max_distance, stream);
}

void buildRelativeAttentionBias(
    th::Tensor& relative_attention_bias,
    th::Tensor& relative_attention_bias_table,
    int64_t const num_head, int64_t const max_seq_len, int64_t const num_bucket, bool const is_bidirectional,
    int64_t const max_distance)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    switch (relative_attention_bias_table.scalar_type())
    {
    case at::ScalarType::Float:
        handleInvokeRelativeAttentionBias<float>(relative_attention_bias, relative_attention_bias_table, num_head,
            max_seq_len, num_bucket, is_bidirectional, max_distance, stream);
        break;
    case at::ScalarType::Half:
        handleInvokeRelativeAttentionBias<half>(relative_attention_bias, relative_attention_bias_table, num_head,
            max_seq_len, num_bucket, is_bidirectional, max_distance, stream);
        break;
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
        handleInvokeRelativeAttentionBias<__nv_bfloat16>(relative_attention_bias, relative_attention_bias_table,
            num_head, max_seq_len, num_bucket, is_bidirectional, max_distance, stream);
        break;
#endif
    default: throw std::runtime_error("Unimplemented scalar type");
    }

    sync_check_cuda_error();
}

}

static auto relative_attention_bias
    = torch::RegisterOperators("suggestify::relative_attention_bias", &torch_ext::buildRelativeAttentionBias);
