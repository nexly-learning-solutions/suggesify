
#include "suggestify/common/cudaUtils.h"
#include "suggestify/common/logger.h"
#include "suggestify/common/mpiUtils.h"
#include "executor.h"
#include "serialization.h"
#include "suggestify/plugins/api/tllmPlugin.h"
#include <csignal>

namespace tle = suggestify::executor;

int main(int argc, char* argv[])
{
#if ENABLE_MULTI_DEVICE

    if (std::getenv("FORCE_NCCL_ALL_REDUCE_STRATEGY") != nullptr)
    {
        LOG_INFO("FORCE_NCCL_ALL_REDUCE_STRATEGY env variable detected in worker");
    }

    initTrtLlmPlugins();

    suggestify::mpi::initialize(suggestify::mpi::MpiThreadSupport::THREAD_MULTIPLE, true);

    MPI_Comm parentComm;
    MPI_Comm_get_parent(&parentComm);
    if (parentComm == MPI_COMM_NULL)
    {
        LOG_ERROR("TRT-LLM worker has no parent!");
        return -1;
    }

    int size;
    MPI_Comm_remote_size(parentComm, &size);
    if (size != 1)
    {
        LOG_ERROR("Parent size is %d, must be 1", size);
        return -1;
    }

    CUDA_CHECK(::cudaSetDeviceFlags(cudaDeviceScheduleYield));


    int64_t bufferSize;
    MPICHECK(MPI_Bcast(&bufferSize, 1, MPI_INT64_T, 0, parentComm));
    std::vector<char> buffer(bufferSize);
    MPICHECK(MPI_Bcast(buffer.data(), bufferSize, MPI_CHAR, 0, parentComm));
    std::istringstream is(std::string(buffer.begin(), buffer.end()));
    auto modelPath = tle::Serialization::deserializeString(is);
    auto modelType = tle::Serialization::deserializeModelType(is);
    auto executorConfig = tle::Serialization::deserializeExecutorConfig(is);

    auto orchLeaderComm = std::make_shared<suggestify::mpi::MpiComm>(parentComm, true);
    auto parallelConfig = executorConfig.getParallelConfig();
    CHECK_WITH_INFO(parallelConfig.has_value(), "Parallel config should have a value.");
    CHECK_WITH_INFO(
        parallelConfig.value().getOrchestratorConfig().has_value(), "Orchestrator config should have a value.");
    auto orchConfig = parallelConfig.value().getOrchestratorConfig().value();
    CHECK_WITH_INFO(parallelConfig.has_value(), "Parallel config should have a value.");
    auto newOrchConfig = tle::OrchestratorConfig(false, orchConfig.getWorkerExecutablePath(), orchLeaderComm);
    parallelConfig.value().setOrchestratorConfig(newOrchConfig);
    executorConfig.setParallelConfig(parallelConfig.value());
    auto executor = tle::Executor(modelPath, modelType, executorConfig);

    MPI_Barrier(parentComm);
    LOG_INFO("Executor instance created by worker");

#endif

    return 0;
}
