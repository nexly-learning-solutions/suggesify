
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace sugesstify::batch_manager
{


class SequenceSlotManager
{
public:
    using SlotIdType = int32_t;
    using SequenceIdType = std::uint64_t;

    SequenceSlotManager(SlotIdType maxNumSlots, uint64_t maxSequenceIdleMicroseconds);

    std::optional<SlotIdType> getSequenceSlot(bool const& startFlag, SequenceIdType const& sequenceId);

    void freeSequenceSlot(SequenceIdType sequenceId);

    void freeIdleSequenceSlots();

private:
    SlotIdType mMaxNumSlots;
    std::chrono::microseconds mMaxSequenceIdleMicroseconds;

    std::unordered_map<SequenceIdType, SlotIdType> mSequenceIdToSlot;
    std::queue<SlotIdType> mAvailableSlots;
    std::vector<std::chrono::steady_clock::time_point> mLastTimepoint;
};

}
