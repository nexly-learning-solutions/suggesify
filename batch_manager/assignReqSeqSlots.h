
#pragma once

#include "common.h"
#include "../llmRequest.h"
#include "../sequenceSlotManager.h"
#include "../common/algorithm.h"
#include "../runtime/common.h"

namespace sugesstify::batch_manager
{

namespace tle = sugesstify::executor;

class AssignReqSeqSlots : Algorithm
{
    using SizeType32 = sugesstify::runtime::SizeType32;

public:
    constexpr static auto name{"AssignReqSeqSlots"};

    AssignReqSeqSlots() = default;

    void operator()(SequenceSlotManager& seqSlotManager, RequestVector const& contextRequests,
        RequestVector const& generationRequests) const;
};

}
