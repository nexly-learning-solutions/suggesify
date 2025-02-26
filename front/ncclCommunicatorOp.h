#pragma once

#include "../ncclCommunicator.h"
#include "thUtils.h"
#include <memory>

namespace th = torch;

namespace torch_ext
{

class NcclCommunicatorOp : public th::jit::CustomClassHolder
{
public:
    NcclCommunicatorOp(int64_t tpSize, int64_t ppSize, int64_t rank);

    void send(th::Tensor tensor, int64_t toRank) const;
    void recv(th::Tensor& tensor, int64_t fromRank) const;

private:
    int32_t mRank;
    std::shared_ptr<suggestify::runtime::NcclCommunicator> mPipelineComm;
};

}
