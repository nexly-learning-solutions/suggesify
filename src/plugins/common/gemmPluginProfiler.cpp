
#include "../plugins/common/gemmPluginProfiler.h"
#include "../common/cublasMMWrapper.h"
#include "../src/cutlass_kernels/fp8_rowwise_gemm/fp8_rowwise_gemm.h"
#include "../src/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "../src/cutlass_kernels/fused_gated_gemm/fused_gated_gemm.h"
#include "../src/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "../plugins/lowLatencyGemmPlugin/lowLatencyGemmPlugin.h"
#include "../plugins/lowLatencyGemmSwigluPlugin/lowLatencyGemmSwigluPlugin.h"
#include "../plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"

namespace suggestify::plugins
{

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler()
{
    mMNKProfileMap = std::make_shared<MNKProfileMap>();

    auto const skipEnv = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
    mSkip = (skipEnv != NULL && std::stoi(skipEnv));
    if (mSkip)
    {
        LOG_DEBUG(
            "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin profilings. It could result in runtime error "
            "if default tactic is not defined.");
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::serialize(
    char*& buffer, GemmIdType const& gemmId) const
{
    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

    write(buffer, static_cast<int>(mProfileMap->size()));
    for (auto const& pair : *mProfileMap)
    {
        write(buffer, pair);
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::deserialize(
    char const*& data, GemmDims& dims, GemmIdType const& gemmId)
{
    writer_lock lock(mMNKProfileMap->mutex);

    mDims = dims;

    if (!mMNKProfileMap->existsMProfileMap(gemmId))
    {
        mMNKProfileMap->createMProfileMap(gemmId);
    }
    auto profileMap = mMNKProfileMap->getMProfileMap(gemmId);
    int selectedMapSize;
    read(data, selectedMapSize);
    for (int ii = 0; ii < selectedMapSize; ++ii)
    {
        std::pair<int, std::optional<Config>> config;
        read(data, config);
        profileMap->insert(config);
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
size_t GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getSerializationSize(
    GemmIdType const& gemmId) const
{
    reader_lock lock(mMNKProfileMap->mutex);
    return sizeof(int) +
        mMNKProfileMap->getMProfileMap(gemmId)->size()
        * sizeof(std::pair<int, std::optional<Config>>);
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
int GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getMaxProfileM() const
{
    return 8192;
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::initTmpData(
    int m, int n, int k, char* workspace, size_t size, cudaStream_t stream)
{
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTactics(
    RunnerPtr const& runner, nvinfer1::DataType const& type, GemmDims const& dims, GemmIdType const& gemmId)
{
    writer_lock lock(mMNKProfileMap->mutex);

    if (!dims.isInitialized())
    {
        return;
    }

    mRunner = runner;
    mType = type;

    int const maxM = std::min(nextPowerOfTwo(dims.maxM), getMaxProfileM());
    computeTmpSize(maxM, dims.n, dims.k);

    if (!mMNKProfileMap->existsMProfileMap(gemmId))
    {
        mMNKProfileMap->createMProfileMap(gemmId);
    }

    if (mSkip)
    {
        return;
    }

    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);
    bool isAllocated{false};

    auto profileTactics = [&mProfileMap, &isAllocated, this](int m, int n, int k)
    {
        if (mProfileMap->count(m) == 0)
        {
            if (!isAllocated)
            {
                allocateTmpData();
                isAllocated = true;
            }
            initTmpData(m, n, k, mWorkspaceTmp, mTmpWorkspaceSizeInBytes, mStream);
            auto const tactics = this->getTactics(m, n, k);
            mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics)});
        }
    };

    common::check_cuda_error(cudaStreamCreate(&mStream));

    int const startMinMRounded = nextPowerOfTwo(dims.minM);
    for (int m = std::max(1, startMinMRounded); m < maxM; m *= 2)
    {
        profileTactics(m, dims.n, dims.k);
    }

    profileTactics(maxM, dims.n, dims.k);

    if (isAllocated)
    {
        freeTmpData();
    }
    common::check_cuda_error(cudaStreamDestroy(mStream));
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfig(
    int m, GemmIdType const& gemmId) const
{
    reader_lock lock(mMNKProfileMap->mutex);

    if (mSkip)
    {
        LOG_TRACE("Skip is set, no best config is set for this instance");
        return std::nullopt;
    }

    int const mRounded = std::min(std::max(1, nextPowerOfTwo(m)), getMaxProfileM());
    fflush(stdout);
    return mMNKProfileMap->getMProfileMap(gemmId)->at(mRounded);
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::allocateTmpData()
{
    CHECK_WITH_INFO(mTmpWorkspaceSizeInBytes > 0, "tmpWorkspaceSizeInBytes must be larger than 0");
    auto const status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
    CHECK_WITH_INFO(status == cudaSuccess, "Can't allocate tmp workspace for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::freeTmpData()
{
    auto const status = cudaFree(mWorkspaceTmp);
    CHECK_WITH_INFO(status == cudaSuccess, "Can't free tmp workspace for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, std::vector<Config> const& tactics)
{
    LOG_DEBUG(__PRETTY_FUNCTION__);

    float bestTime = std::numeric_limits<float>::max();
    Config bestConfig;
    bool foundOne = false;

    for (int ii = 0; ii < tactics.size(); ++ii)
    {
        Config const& candidateConfig = tactics[ii];
        float time = std::numeric_limits<float>::max();
        try
        {
            if (!checkTactic(m, n, k, candidateConfig))
            {
                continue;
            }
            time = profileTacticForProblem(m, n, k, candidateConfig);
            foundOne = true;
        }
        catch (std::exception const& e)
        {
            std::ostringstream msg;
            msg << "Cannot profile configuration " << ii;
            if constexpr (std::is_same_v<Config, suggestify::cutlass_extensions::CutlassGemmConfig>)
            {
                msg << ": " << candidateConfig.toString();
            }
            msg << "\n (for"
                << " m=" << m << ", n=" << n << ", k=" << k << ")"
                << ", reason: \"" << e.what() << "\". Skipped";
            LOG_TRACE(msg.str());
            cudaGetLastError();
            continue;
        }

        if (time < bestTime)
        {
            bestConfig = candidateConfig;
            bestTime = time;
        }
    }

    if (!foundOne)
    {
        std::ostringstream msg;
        msg << "Have not found any valid GEMM config for shape ("
            << "m=" << m << ", n=" << n << ", k=" << k << "). Will try to use default or fail at runtime";
        LOG_WARNING(msg.str());
        return std::nullopt;
    }
    return {bestConfig};
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
float GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticForProblem(
    int m, int n, int k, Config const& tactic)
{
    constexpr int warmup = 5;
    constexpr int runs = 10;

    cudaStream_t stream = mStream;

    for (int i = 0; i < warmup; ++i)
    {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    common::check_cuda_error(cudaEventCreate(&start));
    common::check_cuda_error(cudaEventCreate(&stop));
    common::check_cuda_error(cudaStreamSynchronize(stream));
    common::check_cuda_error(cudaEventRecord(start, stream));

    for (int i = 0; i < runs; ++i)
    {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    common::check_cuda_error(cudaEventRecord(stop, stream));

    common::check_cuda_error(cudaEventSynchronize(stop));

    float elapsed;
    common::check_cuda_error(cudaEventElapsedTime(&elapsed, start, stop));

    common::check_cuda_error(cudaEventDestroy(start));
    common::check_cuda_error(cudaEventDestroy(stop));

    return elapsed / runs;
}

template class GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig,
    std::shared_ptr<suggestify::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>, GemmIdCore,
    GemmIdCoreHash>;

template class GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig,
    std::shared_ptr<suggestify::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>, GemmIdCore,
    GemmIdCoreHash>;

template class GemmPluginProfiler<cublasLtMatmulHeuristicResult_t,
    std::shared_ptr<suggestify::common::CublasMMWrapper>, GemmIdCublas, GemmIdCublasHash>;

template class GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig, MixtureOfExpertsPlugin*,
    GemmIDMoe, GemmIDMoeHash>;

template class GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig,
    std::shared_ptr<suggestify::kernels::cutlass_kernels::CutlassFusedGatedGemmRunnerInterface>, GemmIdCore,
    GemmIdCoreHash>;

template class GemmPluginProfiler<suggestify::cutlass_extensions::CutlassGemmConfig,
    std::shared_ptr<suggestify::kernels::cutlass_kernels::CutlassFp8RowwiseGemmRunnerInterface>, GemmIdCore,
    GemmIdCoreHash>;

template class GemmPluginProfiler<LowLatencyGemmPluginProfiler::Config, LowLatencyGemmRunnerPtr, GemmIdCore,
    GemmIdCoreHash>;

template class GemmPluginProfiler<LowLatencyGemmSwigluPluginProfiler::Config, LowLatencyGemmSwigluRunnerPtr, GemmIdCore,
    GemmIdCoreHash>;
}
