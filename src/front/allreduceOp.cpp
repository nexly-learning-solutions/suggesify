
#include "../common/customAllReduceUtils.h"
#include "../common/dataType.h"
#include "../common/mpiUtils.h"
#include "../common/opUtils.h"
#include "../src/customAllReduceKernels.h"
#include "../torchUtils.h"
#include "thUtils.h"
#include <nvml.h>
#include <torch/extension.h>
#include <unordered_set>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif

using suggestify::kernels::AllReduceFusionOp;
using suggestify::kernels::AllReduceStrategyType;
using suggestify::kernels::AllReduceStrategyConfig;

namespace torch_ext
{

#if ENABLE_MULTI_DEVICE

namespace
{

class NvmlManager
{
public:
    NvmlManager()
    {
        NVML_CHECK(nvmlInit());
    }

    ~NvmlManager()
    {
        NVML_CHECK(nvmlShutdown());
    }
};

std::set<int> getLocalGroup(std::set<int> const& group)
{
    auto const myRank = COMM_SESSION.getRank();
    auto const myLocalRank = LOCAL_COMM_SESSION.getRank();
    auto const localSize = LOCAL_COMM_SESSION.getSize();

    std::vector<int32_t> ranks(localSize, 0);
    std::vector<int32_t> localRanks(localSize, 0);
    if (group.size() >= localSize)
    {
        LOCAL_COMM_SESSION.allgather(&myRank, ranks.data(), 1, suggestify::mpi::MpiType::kINT32);
        LOCAL_COMM_SESSION.allgather(&myLocalRank, localRanks.data(), 1, suggestify::mpi::MpiType::kINT32);
    }
    else
    {
        if (myRank == *group.begin())
        {
            ranks.clear();
            int rank;
            ranks.push_back(myRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.recvValue(rank, *it, 0);
                ranks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(ranks.data(), localSize, suggestify::mpi::MpiType::kINT32, *it, 0);
            }

            localRanks.clear();
            localRanks.push_back(myLocalRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.recvValue(rank, *it, 0);
                localRanks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(localRanks.data(), localSize, suggestify::mpi::MpiType::kINT32, *it, 0);
            }
        }
        else
        {
            LOCAL_COMM_SESSION.sendValue(myRank, *group.begin(), 0);
            LOCAL_COMM_SESSION.recv(ranks.data(), localSize, suggestify::mpi::MpiType::kINT32, *group.begin(), 0);

            LOCAL_COMM_SESSION.sendValue(myLocalRank, *group.begin(), 0);
            LOCAL_COMM_SESSION.recv(
                localRanks.data(), localSize, suggestify::mpi::MpiType::kINT32, *group.begin(), 0);
        }
    }

    std::set<int> localGroup;
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        auto rank = ranks[i];
        if (group.find(rank) != group.end())
        {
            localGroup.insert(localRanks[i]);
        }
    }
    return localGroup;
}

class AllreduceOp
{
public:
    AllreduceOp(std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy,
        AllReduceStrategyConfig config, AllReduceFusionOp op, float eps, bool affine, bool bias)
        : mGroup(std::move(group))
        , mType(type)
        , mStrategy(strategy)
        , mConfig(config)
        , mOp(op)
        , mEps(eps)
        , mAffine(affine)
        , mBias(bias)
    {
    }

    ~AllreduceOp() = default;

    std::tuple<torch::Tensor, torch::Tensor> run(
        torch::Tensor input, torch::optional<torch::Tensor> workspace, torch::TensorList reduce_fusion_inputs) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        auto output = torch::empty_like(input);
        auto finalOutput = torch::empty_like(input);
        size_t size = input.numel();
        auto const sizePerElem = suggestify::common::getDTypeSize(mType);

        AllReduceStrategyType runtimeStrategy;

        static char* forceNcclAllReduceStrategyChar = std::getenv("FORCE_NCCL_ALL_REDUCE_STRATEGY");
        bool forceNcclAllReduceStrategy = (forceNcclAllReduceStrategyChar != nullptr);
        if (forceNcclAllReduceStrategy || mStrategy == AllReduceStrategyType::NCCL)
        {
            runtimeStrategy = AllReduceStrategyType::NCCL;
        }
        else
        {
            runtimeStrategy = selectImplementation(size, mGroup.size(), mType);
        }

        auto const rank = COMM_SESSION.getRank();
        switch (runtimeStrategy)
        {
        case AllReduceStrategyType::NCCL:
        {
            LOG_DEBUG("AllReducePlugin strategy for rank %d: NCCL", rank);
            break;
        }
        case AllReduceStrategyType::ONESHOT:
        {
            LOG_DEBUG("AllReducePlugin strategy for rank %d: ONESHOT", rank);
            break;
        }
        case AllReduceStrategyType::TWOSHOT:
        {
            LOG_DEBUG("AllReducePlugin strategy for rank %d: TWOSHOT", rank);
            break;
        }
        default: break;
        }

        if (runtimeStrategy == AllReduceStrategyType::NCCL)
        {
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                NCCLCHECK(ncclAllReduce(input.data_ptr(), output.mutable_data_ptr(), size, (*getDtypeMap())[mType],
                    ncclSum, *mNcclComm, stream));
                suggestify::kernels::AllReduceParams params;
                int fusion_ptr_idx = 0;
                params.fusion_params.bias_buffer = mBias ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.fusion_params.residual_buffer = reduce_fusion_inputs[fusion_ptr_idx++].data_ptr();
                params.fusion_params.weight_buffer
                    = mAffine ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.local_output_buffer_ptr = finalOutput.mutable_data_ptr();
                params.elts_total = size;
                params.fusion_params.hidden_size = input.size(-1);
                params.fusion_params.eps = mEps;
                params.fusion_params.intermediate_buffer = output.mutable_data_ptr();
                suggestify::kernels::residualRmsNorm(params, mType, stream, mOp);
            }
            else
            {
                NCCLCHECK(ncclAllReduce(input.data_ptr(), output.mutable_data_ptr(), size, (*getDtypeMap())[mType],
                    ncclSum, *mNcclComm, stream));
            }
        }
        else
        {
            auto const tpSize = mGroup.size();
            int tpRank = 0;
            for (auto const& currentRank : mGroup)
            {
                if (rank == currentRank)
                    break;
                ++tpRank;
            }

            int token_num = size / input.size(-1);
            int hidden_size = input.size(-1);
            auto workspace_ptr = workspace.value().mutable_data_ptr();
            auto params = suggestify::kernels::AllReduceParams::deserialize(
                reinterpret_cast<int64_t*>(workspace_ptr), tpSize, tpRank, mType, token_num, hidden_size, mOp);

            params.local_input_buffer_ptr = input.data_ptr();
            params.elts_total = size;
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                int fusion_ptr_idx = 0;
                params.local_output_buffer_ptr = finalOutput.mutable_data_ptr();
                params.fusion_params.bias_buffer = mBias ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.fusion_params.residual_buffer = reduce_fusion_inputs[fusion_ptr_idx++].data_ptr();
                params.fusion_params.weight_buffer
                    = mAffine ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.fusion_params.hidden_size = hidden_size;
                params.fusion_params.eps = mEps;
                params.fusion_params.intermediate_buffer = output.mutable_data_ptr();
                for (int i = 0; i < tpSize; ++i)
                {
                    params.fusion_params.lamport_peer_comm_buffer_ptrs[i]
                        = reinterpret_cast<void**>(workspace_ptr)[tpSize * 4 + i];
                    params.fusion_params.lamport_peer_comm_buffer_ptrs[i + suggestify::kernels::MAX_RANKS_PER_NODE]
                        = reinterpret_cast<void**>(workspace_ptr)[tpSize * 5 + i];
                    params.fusion_params
                        .lamport_peer_comm_buffer_ptrs[i + suggestify::kernels::MAX_RANKS_PER_NODE * 2]
                        = reinterpret_cast<void**>(workspace_ptr)[tpSize * 6 + i];
                }
            }
            else
            {
                params.local_output_buffer_ptr = output.mutable_data_ptr();
            }
            suggestify::kernels::customAllReduce(params, mType, runtimeStrategy, mConfig, mOp, stream);
        }

        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            return std::make_tuple(finalOutput, output);
        }
        else
        {
            return std::make_tuple(output, output);
        }
    }

    int initialize() noexcept
    {
        LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        if (mStrategy != AllReduceStrategyType::NCCL)
        {
            initGroupTopology();
        }

        LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

private:
    void initGroupTopology() noexcept
    {
        static std::map<std::set<int>, std::tuple<bool, bool>> cache;
        if (cache.find(mGroup) != cache.end())
        {
            auto [isNVLINKSupported, isP2PSupported] = cache[mGroup];
            mIsNVLINKSupported = isNVLINKSupported;
            mIsP2PSupported = isP2PSupported;
            return;
        }
        setGroupTopology();
        cache[mGroup] = {mIsNVLINKSupported, mIsP2PSupported};
    }

    void setGroupTopology() noexcept
    {
        auto const rank = COMM_SESSION.getRank();
        LOG_INFO("Detecting local TP group for rank %d", rank);
        std::set<int> localGroup = getLocalGroup(mGroup);
        if (mGroup.size() != localGroup.size())
        {
            mIsP2PSupported = false;
            mIsNVLINKSupported = false;
            LOG_INFO("Found inter-node TP group for rank %d", rank);
            return;
        }
        LOG_INFO("TP group is intra-node for rank %d", rank);

        NvmlManager nvmlManager;
        std::unordered_set<int> visitedDevice;
        mIsP2PSupported = true;
        mIsNVLINKSupported = true;

        for (int firstDeviceId : localGroup)
        {
            for (int secondDeviceId : localGroup)
            {
                if (firstDeviceId == secondDeviceId || visitedDevice.find(secondDeviceId) != visitedDevice.end())
                {
                    continue;
                }

                int canAccessPeer = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, firstDeviceId, secondDeviceId));

                if (!canAccessPeer)
                {
                    mIsP2PSupported = false;
                    mIsNVLINKSupported = false;

                    return;
                }

                nvmlDevice_t firstDevice;
                NVML_CHECK(nvmlDeviceGetHandleByIndex(firstDeviceId, &firstDevice));

                bool isNVLINK = false;

                for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++)
                {
                    nvmlPciInfo_t remotePciInfo;
                    if (nvmlDeviceGetNvLinkRemotePciInfo_v2(firstDevice, link, &remotePciInfo) != NVML_SUCCESS)
                    {
                        continue;
                    }

                    nvmlDevice_t remoteDevice;
                    auto const result = nvmlDeviceGetHandleByPciBusId_v2(remotePciInfo.busId, &remoteDevice);

                    if (result == NVML_SUCCESS)
                    {
                        unsigned int remoteDeviceId;
                        NVML_CHECK(nvmlDeviceGetIndex(remoteDevice, &remoteDeviceId));

                        if (remoteDeviceId == secondDeviceId)
                        {
                            isNVLINK = true;
                        }
                    }
                    else if (result == NVML_ERROR_NOT_FOUND)
                    {
                        nvmlDevice_t secondDevice;
                        NVML_CHECK(nvmlDeviceGetHandleByIndex(secondDeviceId, &secondDevice));

                        for (unsigned int secondLink = 0; secondLink < NVML_NVLINK_MAX_LINKS; secondLink++)
                        {
                            nvmlPciInfo_t secondRemotePciInfo;
                            if (nvmlDeviceGetNvLinkRemotePciInfo_v2(secondDevice, secondLink, &secondRemotePciInfo)
                                != NVML_SUCCESS)
                            {
                                continue;
                            }

                            if (strcmp(remotePciInfo.busId, secondRemotePciInfo.busId) == 0)
                            {
                                isNVLINK = true;
                                break;
                            }
                        }
                    }
                    else
                    {
                        NVML_CHECK(result);
                    }

                    if (isNVLINK)
                    {
                        break;
                    }
                }

                mIsNVLINKSupported &= isNVLINK;
            }
            visitedDevice.insert(firstDeviceId);
        }
    }

    AllReduceStrategyType selectImplementation(size_t messageSize, int worldSize, nvinfer1::DataType type) noexcept
    {
        bool const isAuto = (mStrategy == AllReduceStrategyType::AUTO);

        if (!mIsP2PSupported)
        {
            if (!isAuto)
            {
                LOG_WARNING("Since Peer to Peer not supported, fallback to AllReduceStrategy: NCCL");
            }
            return AllReduceStrategyType::NCCL;
        }

        if (isAuto && !mIsNVLINKSupported)
        {
            return AllReduceStrategyType::NCCL;
        }

        auto const maxWorkspaceSize = suggestify::utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(worldSize);

        AllReduceStrategyType strat = AllReduceStrategyType::NCCL;
        auto const messageSizeBytes = messageSize * suggestify::common::getDTypeSize(type);

        if (messageSizeBytes <= maxWorkspaceSize)
        {
            if (!isAuto)
            {
                strat = mStrategy;
            }
            else if (worldSize <= 2)
            {
                strat = AllReduceStrategyType::ONESHOT;
            }
            else if (worldSize <= 4)
            {
                if (messageSizeBytes < 1 * 1000 * 1000)
                {
                    strat = AllReduceStrategyType::ONESHOT;
                }
                else
                {
                    strat = AllReduceStrategyType::NCCL;
                }
            }
            else
            {
                if (messageSizeBytes < 500 * 1000)
                {
                    strat = AllReduceStrategyType::ONESHOT;
                }
                else
                {
                    strat = AllReduceStrategyType::NCCL;
                }
            }

            if (!suggestify::kernels::configurationSupported(strat, messageSize, worldSize, type))
            {
                if (!isAuto)
                {
                    LOG_WARNING("Since not alignment, fallback to AllReduceStrategy: NCCL");
                }
                strat = AllReduceStrategyType::NCCL;
            }
        }
        else
        {
            if (!isAuto)
            {
                LOG_WARNING("Since messageSize > maxWorkspace, fallback to AllReduceStrategy: NCCL");
            }
            strat = AllReduceStrategyType::NCCL;
        }

        return strat;
    }

private:
    std::set<int> mGroup;
    bool mIsNVLINKSupported;
    bool mIsP2PSupported;
    nvinfer1::DataType mType;
    AllReduceStrategyType mStrategy;
    AllReduceStrategyConfig mConfig;
    AllReduceFusionOp mOp;
    float mEps;
    std::shared_ptr<ncclComm_t> mNcclComm;
    bool mAffine;
    bool mBias;
};

}

#endif

std::tuple<torch::Tensor, torch::Tensor> allreduce(torch::Tensor input, torch::optional<torch::Tensor> workspace,
    torch::TensorList reduce_fusion_inputs, torch::List<int64_t> group_, int64_t const strategy_, int64_t const config_,
    int64_t const fusion_op_, double const eps_, bool const affine_, bool const bias_)
{
#if ENABLE_MULTI_DEVICE
    auto const dtype = suggestify::runtime::TorchUtils::dataType(input.scalar_type());
    auto const strategy = static_cast<AllReduceStrategyType>(int8_t(strategy_));
    auto const config = static_cast<AllReduceStrategyConfig>(int8_t(config_));
    auto const fusion_op = static_cast<AllReduceFusionOp>(int8_t(fusion_op_));
    float const eps = eps_;
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllreduceOp op(group, dtype, strategy, config, fusion_op, eps, affine_, bias_);
    op.initialize();
    auto output = op.run(input, workspace, reduce_fusion_inputs);
    return output;
#else
    return std::make_tuple(input, input);
#endif
}

}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "allreduce(Tensor input, Tensor? workspace, Tensor[] reduce_fusion_inputs, int[] group, int "
        "strategy, int config, int op, float eps, bool affine, bool bias) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("allreduce", &torch_ext::allreduce);
}
