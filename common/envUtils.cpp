
#include "envUtils.h"
#include "cudaUtils.h"
#include "logger.h"
#include <cstdlib>

namespace suggestify::common
{

std::optional<int32_t> getIntEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    int32_t const val = std::stoi(env);
    if (val <= 0)
    {
        return std::nullopt;
    }
    return {val};
};

static bool getBoolEnv(char const* name)
{
    char const* env = std::getenv(name);
    return env && env[0] == '1' && env[1] == '\0';
}

bool forceXQAKernels()
{
    static bool const forceXQA = (getIntEnv("TRFORCE_XQA").value_or(0) != 0);
    return forceXQA;
}

std::optional<bool> getEnvEnableXQAJIT()
{
    static bool init = false;
    static bool exists = false;
    static bool enableXQAJIT = false;
    if (!init)
    {
        init = true;
        char const* enable_xqa_jit_var = std::getenv("TRENABLE_XQA_JIT");
        if (enable_xqa_jit_var)
        {
            exists = true;
            if (enable_xqa_jit_var[0] == '1' && enable_xqa_jit_var[1] == '\0')
            {
                enableXQAJIT = true;
            }
        }
    }
    if (exists)
    {
        return enableXQAJIT;
    }
    else
    {
        return std::nullopt;
    }
}

bool getEnvMmhaMultiblockDebug()
{
    static bool init = false;
    static bool forceMmhaMaxSeqLenTile = false;
    if (!init)
    {
        init = true;
        char const* enable_mmha_debug_var = std::getenv("TRENABLE_MMHA_MULTI_BLOCK_DEBUG");
        if (enable_mmha_debug_var)
        {
            if (enable_mmha_debug_var[0] == '1' && enable_mmha_debug_var[1] == '\0')
            {
                forceMmhaMaxSeqLenTile = true;
            }
        }
    }
    return forceMmhaMaxSeqLenTile;
}

int getEnvMmhaBlocksPerSequence()
{
    static bool init = false;
    static int mmhaBlocksPerSequence = 0;
    if (!init)
    {
        init = true;
        char const* mmhaBlocksPerSequenceEnv = std::getenv("TRMMHA_BLOCKS_PER_SEQUENCE");
        if (mmhaBlocksPerSequenceEnv)
        {
            mmhaBlocksPerSequence = std::atoi(mmhaBlocksPerSequenceEnv);
            if (mmhaBlocksPerSequence <= 0)
            {
                LOG_WARNING("Invalid value for TRMMHA_BLOCKS_PER_SEQUENCE. Will use default values instead!");
            }
        }
    }
    return mmhaBlocksPerSequence;
}

int getEnvMmhaKernelBlockSize()
{
    static bool init = false;
    static int mmhaKernelBlockSize = 0;
    if (!init)
    {
        init = true;
        char const* mmhaKernelBlockSizeEnv = std::getenv("TRMMHA_KERNEL_BLOCK_SIZE");
        if (mmhaKernelBlockSizeEnv)
        {
            mmhaKernelBlockSize = std::atoi(mmhaKernelBlockSizeEnv);
            if (mmhaKernelBlockSize <= 0)
            {
                LOG_WARNING("Invalid value for TRMMHA_KERNEL_BLOCK_SIZE. Will use default values instead!");
            }
        }
    }
    return mmhaKernelBlockSize;
}

bool getEnvEnablePDL()
{
    static bool init = false;
    static bool enablePDL = false;
    if (!init)
    {
        init = true;
        if (getSMVersion() >= 90)
        {
            enablePDL = getBoolEnv("TRENABLE_PDL");
        }
    }
    return enablePDL;
}

bool getEnvUseUCXKvCache()
{
    static bool const useUCXKVCache = getBoolEnv("TRUSE_UCX_KVCACHE");
    return useUCXKVCache;
}

std::string getEnvUCXInterface()
{
    static bool init = false;
    static std::string ucxInterface;
    if (!init)
    {
        init = true;
        {
            char const* ucx_interface = std::getenv("TRUCX_INTERFACE");
            if (ucx_interface)
            {
                ucxInterface = ucx_interface;
            }
        }
    }
    return ucxInterface;
}

bool getEnvDisaggLayerwise()
{
    static bool const disaggLayerwise = getBoolEnv("TRDISAGG_LAYERWISE");
    return disaggLayerwise;
}

bool getEnvParallelCacheSend()
{
    static bool const parallelCacheSend = getBoolEnv("TRPARALLEL_CACHE_SEND");
    return parallelCacheSend;
}

bool getEnvRequestKVCacheSerial()
{
    static bool const requestKVCacheSerial = getBoolEnv("TRREQUEST_KV_CACHE_SERIAL");
    return requestKVCacheSerial;
}

bool getEnvDisableKVCacheTransferOverlap()
{
    static bool const disableKVCacheTransferOverlap = getBoolEnv("TRDISABLE_KV_CACHE_TRANSFER_OVERLAP");
    return disableKVCacheTransferOverlap;
}

bool getEnvDisableReceiveKVCacheParallel()
{
    static bool const disableReceiveParallel = getBoolEnv("TRDISABLE_KVCACHE_RECEIVE_PARALLEL");
    return disableReceiveParallel;
}

}
