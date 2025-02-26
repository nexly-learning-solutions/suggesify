
#include "../common/mpiUtils.h"
#include "../common/opUtils.h"
#include "../torchUtils.h"

#include <NvInferRuntime.h>
#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <set>
#include <string>
#include <torch/extension.h>
#include <vector>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

namespace torch_ext
{
#if ENABLE_MULTI_DEVICE

namespace
{

class AllgatherOp
{
public:
    AllgatherOp(std::set<int> group, nvinfer1::DataType type)
        : mGroup(std::move(group))
        , mType(type)
    {
    }

    ~AllgatherOp() = default;

    int initialize() noexcept
    {
        LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

    torch::Tensor run(torch::Tensor input) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        std::vector<int64_t> outputShape = input.sizes().vec();
        outputShape.insert(outputShape.begin(), mGroup.size());
        auto output = torch::empty(outputShape, input.options());
        size_t size = input.numel();
        CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        NCCLCHECK(ncclAllGather(
            input.data_ptr(), output.mutable_data_ptr(), size, (*getDtypeMap())[mType], *mNcclComm, stream));
        return output;
    }

private:
    std::set<int> mGroup;
    nvinfer1::DataType mType;
    std::shared_ptr<ncclComm_t> mNcclComm;
};

}

#endif

torch::Tensor allgather(torch::Tensor input, torch::List<int64_t> group_)
{
#if ENABLE_MULTI_DEVICE
    auto const type = suggestify::runtime::TorchUtils::dataType(input.scalar_type());
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllgatherOp op(group, type);
    op.initialize();
    auto output = op.run(input);
    return output;
#else
    return input;
#endif
}

}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("allgather(Tensor input, int[] group) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("allgather", &torch_ext::allgather);
}
