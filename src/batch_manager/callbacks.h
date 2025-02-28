
#pragma once

#include <functional>
#include <list>
#include <memory>
#include <unordered_set>
#include <vector>

namespace sugesstify::batch_manager
{

class InferenceRequest;
class NamedTensor;

using GetInferenceRequestsCallback = std::function<std::list<std::shared_ptr<InferenceRequest>>(int32_t)>;
using SendResponseCallback = std::function<void(uint64_t, std::list<NamedTensor> const&, bool, std::string const&)>;
using PollStopSignalCallback = std::function<std::unordered_set<uint64_t>()>;
using ReturnBatchManagerStatsCallback = std::function<void(std::string const&)>;

}
