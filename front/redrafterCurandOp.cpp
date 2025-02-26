
#include "suggestify/common/cudaUtils.h"
#include "../src/decodingCommon.h"
#include "../src/speculativeDecoding/explicitDraftTokensKernels.h"
#include "suggestify/layers/defaultDecodingParams.h"
#include "../common.h"
#include "thUtils.h"

#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

#include <cstdint>

namespace th = torch;
namespace tr = suggestify::runtime;
namespace tk = suggestify::kernels;
namespace tksd = suggestify::kernels::speculative_decoding;

namespace torch_ext
{

namespace
{
void initializeDeviceCurandStates(
    uint64_t batchSize, th::Tensor& curandState, th::optional<th::Tensor>& randomSeeds, cudaStream_t stream)
{
    auto* curandStatePtr = get_ptr<curandState_t>(curandState);
    tr::SizeType32* batchSlotsPtr = nullptr;

    if (randomSeeds.has_value())
    {
        if (batchSize > 1 && randomSeeds->size(0) == 1)
        {
            CHECK_WITH_INFO(randomSeeds->device().is_cpu(), "Random seed tensor expected on host.");
            auto const randomSeed = get_val<uint64_t>(randomSeeds.value(), 0);
            tk::invokeCurandInitialize(curandStatePtr, batchSlotsPtr, batchSize, randomSeed, stream);
        }
        else
        {
            CHECK_WITH_INFO(
                randomSeeds->dim() == 1 && randomSeeds->size(0) == batchSize, "Random seed tensor size mismatch.");
            CHECK_WITH_INFO(randomSeeds->device().is_cuda(), "Random seed tensor expected on device.");

            auto* randomSeedsPtr = get_ptr<uint64_t>(randomSeeds.value());
            tk::invokeCurandBatchInitialize(curandStatePtr, batchSlotsPtr, batchSize, randomSeedsPtr, stream);
        }
    }
    else
    {
        tk::invokeCurandInitialize(
            curandStatePtr, batchSlotsPtr, batchSize, suggestify::layers::DefaultDecodingParams::getSeed(), stream);
    }
    sync_check_cuda_error();
}
}

void prepareRandomTensors(th::Tensor& curandState,
    th::Tensor& randDataSample,
    th::Tensor& randDataValidation,
    th::optional<th::Tensor> randomSeeds,
    int64_t const batchSize,
    int64_t const numPaths,
    int64_t const draftLength,
    bool const initialize
)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto const scalarType = randDataSample.scalar_type();
    CHECK_TYPE(randDataValidation, scalarType);

    CHECK_WITH_INFO(
        randDataSample.dim() == 1 && randDataSample.size(0) == batchSize, "Random sample tensor size mismatch.");
    CHECK_WITH_INFO(randDataValidation.dim() == 3 && randDataValidation.size(0) == batchSize
            && randDataValidation.size(1) == numPaths && randDataValidation.size(2) == draftLength,
        "Random validation tensor size mismatch.");

    CHECK_WITH_INFO(
        curandState.dim() == 2 && curandState.size(0) == batchSize && curandState.size(1) == sizeof(curandState_t),
        "Curand state tensor shpe mismatch."
        "(got (%lu, %lu), need (%lu, %lu)).",
        curandState.size(0), curandState.size(1), batchSize, sizeof(curandState_t));

    if (initialize)
    {
        initializeDeviceCurandStates(batchSize, curandState, randomSeeds, stream);
    }

    switch (scalarType)
    {
    case at::ScalarType::Float:
    {
        tksd::FillRandDataExplicitDraftTokensParams<float> params;
        params.batchSize = static_cast<tr::SizeType32>(batchSize);
        params.numPaths = static_cast<tr::SizeType32>(numPaths);
        params.draftLength = static_cast<tr::SizeType32>(draftLength);
        params.randDataSample = get_ptr<float>(randDataSample);
        params.randDataVerification = get_ptr<float>(randDataValidation);
        params.curandState = get_ptr<curandState_t>(curandState);
        params.batchSlots = nullptr;
        params.skipVerification = initialize;

        tksd::invokeFillRandData(params, stream);
    }
    break;
    case at::ScalarType::Half:
    {
        tksd::FillRandDataExplicitDraftTokensParams<half> params;
        params.batchSize = static_cast<tr::SizeType32>(batchSize);
        params.numPaths = static_cast<tr::SizeType32>(numPaths);
        params.draftLength = static_cast<tr::SizeType32>(draftLength);
        params.randDataSample = get_ptr<half>(randDataSample);
        params.randDataVerification = get_ptr<half>(randDataValidation);
        params.curandState = get_ptr<curandState_t>(curandState);
        params.batchSlots = nullptr;
        params.skipVerification = initialize;

        tksd::invokeFillRandData(params, stream);
    }
    break;
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        tksd::FillRandDataExplicitDraftTokensParams<__nv_bfloat16> params;
        params.batchSize = static_cast<tr::SizeType32>(batchSize);
        params.numPaths = static_cast<tr::SizeType32>(numPaths);
        params.draftLength = static_cast<tr::SizeType32>(draftLength);
        params.randDataSample = get_ptr<__nv_bfloat16>(randDataSample);
        params.randDataVerification = get_ptr<__nv_bfloat16>(randDataValidation);
        params.curandState = get_ptr<curandState_t>(curandState);
        params.batchSlots = nullptr;
        params.skipVerification = initialize;

        tksd::invokeFillRandData(params, stream);
    }
    break;
#endif
    default: throw std::runtime_error("Unsupported tensor type.");
    }
    sync_check_cuda_error();
}

}

static auto redrafter_prepare_random_tensors
    = torch::RegisterOperators("suggestify::redrafter_prepare_random_tensors", &torch_ext::prepareRandomTensors);
