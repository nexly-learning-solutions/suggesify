
#include "ncclCommunicatorOp.h"

#include "suggestify/runtime/iBuffer.h"

namespace tr = suggestify::runtime;

namespace torch_ext
{

NcclCommunicatorOp::NcclCommunicatorOp(int64_t tpSize, int64_t ppSize, int64_t rank)
    : mRank(static_cast<int32_t>(rank))
{
    mPipelineComm = std::make_shared<suggestify::runtime::NcclCommunicator>(tpSize * ppSize, rank);
}

void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
{
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    suggestify::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->send(*tr::IBuffer::wrap(ptr, size), static_cast<int>(toRank), cudaStream);
}

void NcclCommunicatorOp::recv(th::Tensor& tensor, int64_t fromRank) const
{
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    suggestify::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->receive(*tr::IBuffer::wrap(ptr, size), static_cast<int>(fromRank), cudaStream);
}

}

static auto trtllmNcclCommunicator = torch::jit::class_<torch_ext::NcclCommunicatorOp>("trtllm", "NcclCommunicatorOp")
                                         .def(torch::jit::init<int64_t, int64_t, int64_t>())
                                         .def("send", &torch_ext::NcclCommunicatorOp::send)
                                         .def("recv", &torch_ext::NcclCommunicatorOp::recv);
