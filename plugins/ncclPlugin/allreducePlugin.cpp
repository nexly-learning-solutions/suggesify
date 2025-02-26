#include "allreducePlugin.h"

#include "../common/customAllReduceUtils.h"
#include "../common/dataType.h"
#include "../common/mpiUtils.h"
#include "../src/customAllReduceKernels.h"
#include "../src/userbuffers/ub_interface.h"
#include <nccl.h>
#include <unordered_set>

using namespace nvinfer1;
using suggestify::plugins::AllreducePluginCreator;
using suggestify::plugins::AllreducePlugin;
using suggestify::kernels::AllReduceFusionOp;
using suggestify::kernels::AllReduceStrategyType;
using suggestify::kernels::AllReduceStrategyConfig;

static char const* ALLREDUCE_PLUGIN_VERSION{"1"};
static char const* ALLREDUCE_PLUGIN_NAME{"AllReduce"};
PluginFieldCollection AllreducePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> AllreducePluginCreator::mPluginAttributes;

AllreducePlugin::AllreducePlugin(std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy,
    AllReduceStrategyConfig config, AllReduceFusionOp op, int32_t counter, float eps, int8_t affine, int8_t bias,
    int8_t scale)
    : mGroup(std::move(group))
    , mType(type)
    , mStrategy(strategy)
    , mConfig(config)
    , mOp(op)
    , mEps(eps)
    , mAffine(affine)
    , mBias(bias)
    , mScale(scale)
{
    check();
}

AllreducePlugin::AllreducePlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mStrategy);
    read(d, mConfig);
    read(d, mOp);
    read(d, mEps);
    read(d, mAffine);
    read(d, mBias);
    read(d, mScale);
    mGroup.clear();
    int groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mGroup.insert(groupItem);
    }
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
    check();
}

void AllreducePlugin::check() noexcept
{
    if (mStrategy != AllReduceStrategyType::UB)
    {
        TLLM_CHECK(mOp != AllReduceFusionOp::LAST_PROCESS_FOR_UB);
    }
}

nvinfer1::IPluginV2DynamicExt* AllreducePlugin::clone() const noexcept
{
    auto* plugin = new AllreducePlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AllreducePlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool AllreducePlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    int base_inputs = 0;
    if (mStrategy == AllReduceStrategyType::NCCL || mStrategy == AllReduceStrategyType::UB)
    {
        base_inputs = 1;
    }
    else
    {
        base_inputs = 2;
    }
    int fusion_op_extra_inputs = 0;
    int scale_idx = 0;
    if (mOp != AllReduceFusionOp::NONE)
    {
        ++fusion_op_extra_inputs;
        if (mAffine)
        {
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
                ++fusion_op_extra_inputs;
            ++fusion_op_extra_inputs;
        }
        if (mBias)
        {
            ++fusion_op_extra_inputs;
        }
        if (mScale)
        {
            scale_idx = base_inputs + fusion_op_extra_inputs;
            ++fusion_op_extra_inputs;
        }
    }

    TLLM_CHECK(nbInputs == (base_inputs + fusion_op_extra_inputs));

    if (mStrategy != AllReduceStrategyType::NCCL && mStrategy != AllReduceStrategyType::UB && pos == 1)
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT64) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    if (mOp != AllReduceFusionOp::NONE && mScale && pos == scale_idx)
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    if (mStrategy == AllReduceStrategyType::UB && mOp != AllReduceFusionOp::NONE && mScale && pos == nbInputs)
    {
        return (inOut[pos].type == nvinfer1::DataType::kFP8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void AllreducePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t AllreducePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

AllReduceStrategyType AllreducePlugin::selectImplementation(
    size_t messageSize, int worldSize, nvinfer1::DataType type) noexcept
{
    bool const isAuto = (mStrategy == AllReduceStrategyType::AUTO);

    if (!mIsP2PSupported)
    {
        if (!isAuto)
        {
            TLLM_LOG_INFO("Since Peer to Peer not supported, fallback to AllReduceStrategy: NCCL");
        }
        return AllReduceStrategyType::NCCL;
    }

    if (isAuto && !mIsNVLINKSupported)
    {
        return AllReduceStrategyType::NCCL;
    }

    auto const maxWorkspaceSize = utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(worldSize);

    AllReduceStrategyType strat = AllReduceStrategyType::NCCL;
    auto const messageSizeBytes = messageSize * common::getDTypeSize(type);

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

        if (!kernels::configurationSupported(strat, messageSize, worldSize, type))
        {
            if (!isAuto)
            {
                TLLM_LOG_WARNING("Since not alignment, fallback to AllReduceStrategy: NCCL");
            }
            strat = AllReduceStrategyType::NCCL;
        }
    }
    else
    {
        if (!isAuto)
        {
            TLLM_LOG_WARNING("Since messageSize > maxWorkspace, fallback to AllReduceStrategy: NCCL");
        }
        strat = AllReduceStrategyType::NCCL;
    }

    return strat;
}

int AllreducePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    size_t size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }
    auto const sizePerElem = common::getDTypeSize(mType);

    kernels::AllReduceStrategyType runtimeStrategy;

    static char* forceNcclAllReduceStrategyChar = std::getenv("FORCE_NCCL_ALL_REDUCE_STRATEGY");
    bool forceNcclAllReduceStrategy = (forceNcclAllReduceStrategyChar != nullptr);
    if (forceNcclAllReduceStrategy || mStrategy == AllReduceStrategyType::NCCL)
    {
        runtimeStrategy = AllReduceStrategyType::NCCL;
    }
    else if (mStrategy == AllReduceStrategyType::UB)
    {
        runtimeStrategy = AllReduceStrategyType::UB;
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
        TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: NCCL", rank);
        break;
    }
    case AllReduceStrategyType::ONESHOT:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: ONESHOT", rank);
        break;
    }
    case AllReduceStrategyType::TWOSHOT:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: TWOSHOT", rank);
        break;
    }
    case AllReduceStrategyType::UB:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: UB", rank);
        break;
    }
    default: break;
    }

    if (runtimeStrategy == AllReduceStrategyType::NCCL)
    {
        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM || mOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
        {
            NCCLCHECK(ncclAllReduce(inputs[0], outputs[1], size, (*getDtypeMap())[mType], ncclSum, *mNcclComm, stream));
            suggestify::kernels::AllReduceParams params;
            int fusion_ptr_idx = 0;
            if (mStrategy == AllReduceStrategyType::NCCL)
            {
                fusion_ptr_idx = 1;
            }
            else
            {
                fusion_ptr_idx = 2;
            }
            params.fusion_params.bias_buffer = mBias ? inputs[fusion_ptr_idx++] : nullptr;
            params.fusion_params.residual_buffer = inputs[fusion_ptr_idx++];
            params.fusion_params.weight_buffer = mAffine ? inputs[fusion_ptr_idx++] : nullptr;
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
            {
                params.fusion_params.weight_buffer_pre_residual_norm = mAffine ? inputs[fusion_ptr_idx++] : nullptr;
            }
            params.local_output_buffer_ptr = outputs[0];
            params.elts_total = size;
            params.fusion_params.hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
            params.fusion_params.eps = mEps;
            params.fusion_params.intermediate_buffer = outputs[1];
            TLLM_LOG_DEBUG("residualRmsNorm called");
            suggestify::kernels::residualRmsNorm(params, mType, stream, mOp);
        }
        else
        {
            NCCLCHECK(ncclAllReduce(inputs[0], outputs[0], size, (*getDtypeMap())[mType], ncclSum, *mNcclComm, stream));
        }
    }
    else if (runtimeStrategy == AllReduceStrategyType::UB)
    {
        TLLM_CHECK(!mBias);

        auto const tpSize = mGroup.size();
        size_t dtype_size = suggestify::common::getDTypeSize(mType);
        int hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

        TLLM_CHECK_WITH_INFO(suggestify::runtime::ub::ub_is_initialized(), "UserBuffer has not been initialized!");
        auto ub_buffer0 = suggestify::runtime::ub::ub_get(0);
        auto ub_buffer1 = suggestify::runtime::ub::ub_get(1);
        TLLM_CHECK(inputs[0] == ub_buffer0.addr);
        auto ub_comm = suggestify::runtime::ub::ub_comm();
        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            TLLM_CHECK(mAffine);
            TLLM_CHECK(mScale);
            TLLM_CHECK(outputs[0] == ub_buffer1.addr);
            void* residual = const_cast<void*>(inputs[1]);
            void* gamma = const_cast<void*>(inputs[2]);
            float* scale = const_cast<float*>(reinterpret_cast<float const*>(inputs[3]));
            suggestify::kernels::ub::allreduce2_userbuff_inplace_rmsnorm_quant_launcher(ub_buffer0.handle, 0,
                ub_buffer1.handle, 0, size, hidden_size, nullptr, gamma, mEps, scale, residual, outputs[1], mType,
                ub_comm, stream);
        }
        else if (mOp == AllReduceFusionOp::LAST_PROCESS_FOR_UB)
        {
            TLLM_CHECK(outputs[1] == ub_buffer1.addr);
            void* residual = const_cast<void*>(inputs[1]);
            suggestify::kernels::ub::allreduce2_userbuff_inplace_launcher(
                ub_buffer0.handle, 0, size, mType, ub_comm, stream);
            suggestify::kernels::ub::allgather2_userbuff_residual_launcher(
                ub_buffer1.handle, 0, size, hidden_size, residual, mType, ub_comm, stream);
            TLLM_CUDA_CHECK(
                cudaMemcpyAsync(outputs[0], ub_buffer0.addr, size * dtype_size, cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            suggestify::kernels::ub::allreduce2_userbuff_inplace_launcher(
                ub_buffer0.handle, 0, size, mType, ub_comm, stream);
            TLLM_CUDA_CHECK(
                cudaMemcpyAsync(outputs[0], ub_buffer0.addr, size * dtype_size, cudaMemcpyDeviceToDevice, stream));
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

        int token_num = size / inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
        int hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
        auto params = suggestify::kernels::AllReduceParams::deserialize(
            reinterpret_cast<int64_t*>(const_cast<void*>(inputs[1])), tpSize, tpRank, mType, token_num, hidden_size,
            mOp);

        params.local_output_buffer_ptr = outputs[0];
        params.local_input_buffer_ptr = inputs[0];
        params.elts_total = size;

        int fusion_ptr_idx = 2;
        params.fusion_params.bias_buffer = mBias ? inputs[fusion_ptr_idx++] : nullptr;
        params.fusion_params.residual_buffer = inputs[fusion_ptr_idx++];
        params.fusion_params.weight_buffer = mAffine ? inputs[fusion_ptr_idx++] : nullptr;
        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
            params.fusion_params.weight_buffer_pre_residual_norm = mAffine ? inputs[fusion_ptr_idx++] : nullptr;
        params.fusion_params.hidden_size = hidden_size;
        params.fusion_params.eps = mEps;
        params.fusion_params.intermediate_buffer = outputs[1];
        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            for (int i = 0; i < tpSize; ++i)
            {
                params.fusion_params.lamport_peer_comm_buffer_ptrs[i]
                    = reinterpret_cast<void**>(const_cast<void*>(inputs[1]))[tpSize * 4 + i];
                params.fusion_params.lamport_peer_comm_buffer_ptrs[i + suggestify::kernels::MAX_RANKS_PER_NODE]
                    = reinterpret_cast<void**>(const_cast<void*>(inputs[1]))[tpSize * 5 + i];
                params.fusion_params.lamport_peer_comm_buffer_ptrs[i + suggestify::kernels::MAX_RANKS_PER_NODE * 2]
                    = reinterpret_cast<void**>(const_cast<void*>(inputs[1]))[tpSize * 6 + i];
            }
        }
        TLLM_LOG_DEBUG("customAllReduce called");
        suggestify::kernels::customAllReduce(params, mType, runtimeStrategy, mConfig, mOp, stream);
    }

    return 0;
}

nvinfer1::DataType AllreducePlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    int fusion_op_extra_output = (mOp != AllReduceFusionOp::NONE ? 1 : 0);
    assert(index <= fusion_op_extra_output);
    if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM && mStrategy == AllReduceStrategyType::UB && mScale && index == 0)
    {
        return nvinfer1::DataType::kFP8;
    }
    return inputTypes[0];
}


char const* AllreducePlugin::getPluginType() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

char const* AllreducePlugin::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

int AllreducePlugin::getNbOutputs() const noexcept
{
    return (mOp != AllReduceFusionOp::NONE ? 2 : 1);
}

bool AllreducePlugin::isCustomAllReduceSupported(int ranks_per_node) const noexcept
{
    constexpr bool isCudaVersionSupported =
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        true;
#else
        false;
#endif

    return isCudaVersionSupported && (ranks_per_node % 2 == 0) && (ranks_per_node <= kernels::MAX_RANKS_PER_NODE)
        && (ranks_per_node > 0);
}

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
                COMM_SESSION.recvValue(rank, *it, 0);
                ranks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                COMM_SESSION.send(ranks.data(), localSize, suggestify::mpi::MpiType::kINT32, *it, 0);
            }

            localRanks.clear();
            localRanks.push_back(myLocalRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                COMM_SESSION.recvValue(rank, *it, 0);
                localRanks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                COMM_SESSION.send(localRanks.data(), localSize, suggestify::mpi::MpiType::kINT32, *it, 0);
            }
        }
        else
        {
            COMM_SESSION.sendValue(myRank, *group.begin(), 0);
            COMM_SESSION.recv(ranks.data(), localSize, suggestify::mpi::MpiType::kINT32, *group.begin(), 0);

            COMM_SESSION.sendValue(myLocalRank, *group.begin(), 0);
            COMM_SESSION.recv(localRanks.data(), localSize, suggestify::mpi::MpiType::kINT32, *group.begin(), 0);
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

void AllreducePlugin::initGroupTopology() noexcept
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

void AllreducePlugin::setGroupTopology() noexcept
{
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_INFO("Detecting local TP group for rank %d", rank);
    std::set<int> localGroup = getLocalGroup(mGroup);
    if (mGroup.size() != localGroup.size())
    {
        mIsP2PSupported = false;
        mIsNVLINKSupported = false;
        TLLM_LOG_INFO("Found inter-node TP group for rank %d", rank);
        return;
    }
    TLLM_LOG_INFO("TP group is intra-node for rank %d", rank);

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
            TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, firstDeviceId, secondDeviceId));

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

int AllreducePlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    mNcclComm = getComm(mGroup);
    if (mStrategy != AllReduceStrategyType::NCCL)
    {
        initGroupTopology();
    }

    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    return 0;
}

void AllreducePlugin::terminate() noexcept {}

size_t AllreducePlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType) + sizeof(mStrategy) + sizeof(mConfig) + sizeof(mOp)
        + sizeof(mEps) + sizeof(mAffine) + sizeof(mBias) + sizeof(mScale);
}

void AllreducePlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mStrategy);
    write(d, mConfig);
    write(d, mOp);
    write(d, mEps);
    write(d, mAffine);
    write(d, mBias);
    write(d, mScale);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void AllreducePlugin::destroy() noexcept
{
    delete this;
}


AllreducePluginCreator::AllreducePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("strategy", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("config", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("fusion_op", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("counter", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("affine", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kINT8, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* AllreducePluginCreator::getPluginName() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

char const* AllreducePluginCreator::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

PluginFieldCollection const* AllreducePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AllreducePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    AllReduceStrategyType strategy;
    AllReduceStrategyConfig config;
    AllReduceFusionOp fusion_op;
    int32_t counter;
    float eps;
    int8_t affine;
    int8_t bias;
    int8_t scale;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            auto const* r = static_cast<int const*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "strategy"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            strategy = static_cast<AllReduceStrategyType>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "config"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            config = static_cast<AllReduceStrategyConfig>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "fusion_op"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            fusion_op = static_cast<AllReduceFusionOp>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "counter"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            counter = *static_cast<int32_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "eps"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            eps = *static_cast<float const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "affine"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            affine = *static_cast<int8_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "bias"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            bias = *static_cast<int8_t const*>(fields[i].data);
        }
        else if (!strcmp(attrName, "scale"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            scale = *static_cast<int8_t const*>(fields[i].data);
        }
    }
    try
    {
        auto* obj = new AllreducePlugin(group, type, strategy, config, fusion_op, counter, eps, affine, bias, scale);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AllreducePluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new AllreducePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
