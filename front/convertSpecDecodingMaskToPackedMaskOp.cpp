
#include "suggestify/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "thUtils.h"

namespace th = torch;

namespace torch_ext
{
void convertSpecDecodingMaskToPackedMask(torch::Tensor specDecodingGenerationLengthsTensor,
    torch::Tensor specDecodingMaskTensor, int64_t maxSpecDecodingTokens, torch::Tensor specDecodingPackedMaskTensor,
    torch::optional<int64_t> stream_ptr = torch::nullopt)
{
    CHECK_WITH_INFO(
        at::cuda::is_available(), "convert_spec_decoding_mask_to_packed_mask should be called with cuda enabled.");
    cudaStream_t stream;
    if (stream_ptr.has_value())
    {
        stream = reinterpret_cast<cudaStream_t>(stream_ptr.value());
    }
    else
    {
        stream = at::cuda::getCurrentCUDAStream();
    }
    CHECK_WITH_INFO(specDecodingGenerationLengthsTensor.dim() == 1
            && specDecodingGenerationLengthsTensor.scalar_type() == torch::kInt,
        "spec_decoding_generation_lengths tensor should be 1D int tensor.");

    CHECK_WITH_INFO(specDecodingMaskTensor.dim() == 3 && specDecodingMaskTensor.scalar_type() == torch::kBool,
        "spec_decoding_mask tensor should be 3D bool tensor.");

    CHECK_WITH_INFO(
        specDecodingPackedMaskTensor.dim() == 2 && specDecodingPackedMaskTensor.scalar_type() == torch::kInt,
        "spec_decoding_packed_mask tensor should be 2D int tensor.");

    int batchSize = specDecodingGenerationLengthsTensor.size(0);

    int64_t scanTempMemoryBytes = suggestify::kernels::speculative_decoding::invokeScanGenerationLengths(
        nullptr, 0, nullptr, nullptr, batchSize, stream);
    int64_t reduceMaxTempMemoryBytes = suggestify::kernels::speculative_decoding::invokeReduceMaxGenerationLengths(
        nullptr, 0, nullptr, nullptr, batchSize, stream);

    torch::Tensor scanTempMemoryStorage = torch::empty(
        {
            scanTempMemoryBytes,
        },
        torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    torch::Tensor reduceMaxTempMemoryStorage = torch::empty(
        {
            reduceMaxTempMemoryBytes,
        },
        torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    torch::Tensor scanedSpecDecodingGenerationLengths = torch::empty(
        {
            batchSize,
        },
        torch::dtype(torch::kInt).device(torch::kCUDA).requires_grad(false));
    torch::Tensor maxSpecDecodingGenerationLengths = torch::empty(
        {
            1,
        },
        torch::dtype(torch::kInt).device(torch::kCUDA).requires_grad(false));

    suggestify::kernels::speculative_decoding::invokeScanReduceGenerationLengths(batchSize,
        specDecodingGenerationLengthsTensor.data_ptr<int>(),
        reinterpret_cast<void*>(scanTempMemoryStorage.data_ptr<int8_t>()), scanTempMemoryBytes,
        scanedSpecDecodingGenerationLengths.data_ptr<int>(),
        reinterpret_cast<void*>(reduceMaxTempMemoryStorage.data_ptr<int8_t>()), reduceMaxTempMemoryBytes,
        maxSpecDecodingGenerationLengths.data_ptr<int>(), stream);

    int hostMaxSpecDecodingGenerationLengths;
    cudaMemcpyAsync(&hostMaxSpecDecodingGenerationLengths, maxSpecDecodingGenerationLengths.data_ptr<int>(),
        sizeof(int), cudaMemcpyDeviceToHost, stream);
    suggestify::kernels::speculative_decoding::invokeConvertMaskToPackedMask(batchSize,
        scanedSpecDecodingGenerationLengths.data_ptr<int>(), maxSpecDecodingGenerationLengths.data_ptr<int>(),
        specDecodingMaskTensor.data_ptr<bool>(), nullptr, maxSpecDecodingTokens, maxSpecDecodingTokens + 1,
        specDecodingPackedMaskTensor.data_ptr<int>(), stream);
}

}

static auto convert_spec_decoding_mask_to_packed_mask = torch::RegisterOperators(
    "suggestify::convert_spec_decoding_mask_to_packed_mask", &torch_ext::convertSpecDecodingMaskToPackedMask);
