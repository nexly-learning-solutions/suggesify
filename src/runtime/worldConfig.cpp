
#include "worldConfig.h"

#include "../common/assert.h"
#include "../common/cudaUtils.h"
#include "../common/logger.h"
#include "../common/mpiUtils.h"
#include "../common/stringUtils.h"

#include <algorithm>
#include <numeric>
#include <set>

using namespace suggestify::runtime;
namespace tc = suggestify::common;

WorldConfig::WorldConfig(SizeType32 tensorParallelism, SizeType32 pipelineParallelism, SizeType32 contextParallelism,
    SizeType32 rank, SizeType32 gpusPerNode, std::optional<std::vector<SizeType32>> const& deviceIds)
    : mTensorParallelism{tensorParallelism}
    , mPipelineParallelism{pipelineParallelism}
    , mContextParallelism{contextParallelism}
    , mRank{rank}
    , mGpusPerNode{gpusPerNode}
    , mDeviceIds{deviceIds.value_or(std::vector<SizeType32>(mGpusPerNode))}
{
#if ENABLE_MULTI_DEVICE
    auto const numDevices = mDeviceIds.size();
    CHECK(numDevices > 0);

    if (!deviceIds.has_value())
    {
        mDeviceIds.resize(mGpusPerNode);
        std::iota(mDeviceIds.begin(), mDeviceIds.end(), 0);
    }
    else
    {
        CHECK_WITH_INFO(static_cast<SizeType32>(numDevices) <= mGpusPerNode,
            "Number of device IDs %zu is greater than GPUs per node %d", numDevices, mGpusPerNode);

        CHECK(*std::max_element(mDeviceIds.begin(), mDeviceIds.end()) < mGpusPerNode);
        CHECK(*std::min_element(mDeviceIds.begin(), mDeviceIds.end()) >= 0);

        std::set<SizeType32> const deviceIdSet(mDeviceIds.begin(), mDeviceIds.end());
        CHECK_WITH_INFO(
            deviceIdSet.size() == numDevices, "Device IDs are not unique %zu != %zu", deviceIdSet.size(), numDevices);

        if (std::adjacent_find(deviceIdSet.begin(), deviceIdSet.end(), [](auto x, auto y) { return y - x != 1; })
            != deviceIdSet.end())
        {
            LOG_WARNING("The user specified device IDs are not contiguous!");
        }
        LOG_INFO("Using user-specified devices: %s", tc::arr2str(mDeviceIds.data(), numDevices).c_str());
    }

    CHECK(mTensorParallelism > 0);
    CHECK(mPipelineParallelism > 0);
#else
    mRank = 0;
    mGpusPerNode = 1;
    mTensorParallelism = 1;
    mPipelineParallelism = 1;
#endif
}

bool WorldConfig::validMpiConfig() const
{
    return COMM_SESSION.getSize() == getSize();
}

WorldConfig WorldConfig::mpi(SizeType32 gpusPerNode, std::optional<SizeType32> tensorParallelism,
    std::optional<SizeType32> pipelineParallelism, std::optional<SizeType32> contextParallelism,
    std::optional<std::vector<SizeType32>> const& deviceIds)
{
#if ENABLE_MULTI_DEVICE
    auto& comm = COMM_SESSION;
    auto const mpiSize = comm.getSize();
    auto const mpiRank = comm.getRank();
    auto const mpiLocalSize = LOCAL_COMM_SESSION.getSize();
    LOG_INFO("MPI size: %d, MPI local size: %d, rank: %d", mpiSize, mpiLocalSize, mpiRank);
    auto const pp = pipelineParallelism.value_or(1);
    auto const cp = contextParallelism.value_or(1);
    auto const tp = tensorParallelism.value_or(mpiSize / pp / cp);
    LOG_DEBUG("TP: %d, PP: %d, CP: %d, gpusPerNode: %d", tp, pp, cp, gpusPerNode);
    CHECK_WITH_INFO(
        mpiSize == tp * pp * cp, "MPI size %d != TP size %d * PP size %d * CP Size %d", mpiSize, tp, pp, cp);
    SizeType32 deviceCount{0};
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if ((mpiSize < gpusPerNode && deviceCount < mpiSize) || (mpiSize >= gpusPerNode && deviceCount < gpusPerNode))
    {
        CHECK_WITH_INFO(deviceCount == 1,
            "Detect %d GPUs, the GPU number is incompatible with %d gpusPerNode when MPI size is %d", deviceCount,
            gpusPerNode, mpiSize);
        LOG_WARNING("gpusPerNode is %d but only detect single GPU, will set gpusPerNode to 1", gpusPerNode);
        if (std::getenv("CUDA_VISIBLE_DEVICES") != nullptr || std::getenv("NVIDIA_VISIBLE_DEVICES") != nullptr)
        {
            std::ostringstream oss;
            if (std::getenv("CUDA_VISIBLE_DEVICES") != nullptr)
            {
                oss << " CUDA_VISIBLE_DEVICES=" << std::getenv("CUDA_VISIBLE_DEVICES");
            }
            if (std::getenv("NVIDIA_VISIBLE_DEVICES") != nullptr)
            {
                oss << " NVIDIA_VISIBLE_DEVICES=" << std::getenv("NVIDIA_VISIBLE_DEVICES");
            }
            std::string envStr = oss.str();
            LOG_WARNING(
                "Detect%s, please provide the full device list instead of limiting to single device, "
                "otherwise allreduce performance may be sub-optimal "
                "since custom allreduce kernel relies on P2P access to peer devices.",
                envStr.c_str());
        }
        gpusPerNode = 1;
    }

    return WorldConfig{tp, pp, cp, mpiRank, gpusPerNode, deviceIds};
#else
    return WorldConfig();
#endif
}

std::vector<SizeType32> WorldConfig::getPipelineParallelGroup() const
{
    auto const pp = getPipelineParallelism();
    auto const tp = getTensorParallelism();
    auto const cp = getContextParallelism();
    auto const worldSize = getSize();
    std::vector<SizeType32> group;
    group.reserve(pp);
    for (SizeType32 idx = getTensorParallelRank() * cp + getContextParallelRank(); idx < worldSize; idx += tp * cp)
    {
        group.push_back(idx);
    }
    return group;
}

std::vector<SizeType32> WorldConfig::getTensorParallelGroup() const
{
    auto const tp = getTensorParallelism();
    auto const rank = getRank();
    auto const tpRank = getTensorParallelRank();
    std::vector<SizeType32> group;
    group.reserve(tp);
    for (SizeType32 idx = 0; idx < tp; idx++)
    {
        group.push_back(rank - tpRank + idx);
    }
    return group;
}

std::vector<SizeType32> WorldConfig::getContextParallelGroup() const
{
    auto const cp = getContextParallelism();
    auto const tp = getTensorParallelism();
    auto const pp = getPipelineParallelism();
    auto const rank = getRank();
    std::vector<SizeType32> group;
    group.reserve(cp);
    for (SizeType32 idx = 0; idx < cp; idx++)
    {
        group.push_back(rank + cp % (tp * pp));
    }
    return group;
}
