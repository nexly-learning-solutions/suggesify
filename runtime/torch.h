
#pragma once

#include "../common/assert.h"
#include "../common/logger.h"
#include "cudaStream.h"
#include "iTensor.h"
#include "torchUtils.h"

#include <ATen/ATen.h>

#include <stdexcept>

namespace suggestify::runtime
{

class Torch
{
public:
    static at::Tensor tensor(ITensor::SharedPtr tensor)
    {
        auto const tensorOptions = at::device(TorchUtils::device((*tensor).data()))
                                       .pinned_memory((*tensor).getMemoryType() == MemoryType::kPINNEDPOOL)
                                       .dtype(TorchUtils::dataType((*tensor).getDataType()))
                                       .layout(at::kStrided);
        return at::for_blob(tensor->data(), TorchUtils::shape(tensor->getShape()))
            .options(tensorOptions)
            .deleter(
                [ptr = std::move(tensor)](void* data) mutable
                {
                    try
                    {
                        CHECK(data == ptr->data());
                        ptr.reset();
                    }
                    catch (std::exception const& e)
                    {
                        LOG_EXCEPTION(e);
                    }
                })
            .make_tensor();
    }

    static at::Tensor buffer(IBuffer::SharedPtr buffer)
    {
        auto const shape = ITensor::makeShape({static_cast<runtime::SizeType32>(buffer->getSize())});
        return tensor(ITensor::view(std::move(buffer), shape));
    }

    static void setCurrentStream(runtime::CudaStream& cudaStream)
    {
        at::cuda::setCurrentCUDAStream(TorchUtils::stream(cudaStream));
    }

private:
    Torch() = default;
};

}
