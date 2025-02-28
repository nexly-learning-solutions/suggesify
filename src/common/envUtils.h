
#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace suggestify::common
{
std::optional<int32_t> getIntEnv(char const* name);

bool forceXQAKernels();

std::optional<bool> getEnvEnableXQAJIT();

bool getEnvMmhaMultiblockDebug();

int getEnvMmhaBlocksPerSequence();

int getEnvMmhaKernelBlockSize();

bool getEnvEnablePDL();

bool getEnvUseUCXKvCache();

std::string getEnvUCXInterface();

bool getEnvDisaggLayerwise();

bool getEnvParallelCacheSend();

bool getEnvRequestKVCacheSerial();

bool getEnvDisableKVCacheTransferOverlap();

bool getEnvDisableReceiveKVCacheParallel();

}
