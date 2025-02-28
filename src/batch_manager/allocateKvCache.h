
#pragma once

#include "common.h"
#include "../kvCacheManager.h"
#include "../common/algorithm.h"
#include "../common/optionalRef.h"
#include "../runtime/common.h"

namespace sugesstify::batch_manager
{

namespace tle = sugesstify::executor;

class AllocateKvCache : Algorithm
{
    using BaseKVCacheManager = sugesstify::batch_manager::kv_cache_manager::BaseKVCacheManager;

    template <typename T>
    using OptionalRef = sugesstify::common::OptionalRef<T>;

public:
    constexpr static auto name{"AllocateKvCache"};

    using SizeType32 = sugesstify::runtime::SizeType32;

    AllocateKvCache() = default;

    void operator()(BaseKVCacheManager& kvCacheManager, RequestVector& contextRequests,
        RequestVector const& generationRequests, runtime::ModelConfig const& modelConfig,
        OptionalRef<BaseKVCacheManager> crossKvCacheManager = std::nullopt) const;
};

}
