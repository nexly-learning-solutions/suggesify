
#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_multihead_attention_common.h"
#include "fused_multihead_attention_v2.h"
#include "../common/cudaUtils.h"
#include "tmaDescriptor.h"

namespace suggestify
{
namespace kernels
{


class FusedMHARunnerV2
{
public:
    FusedMHARunnerV2(MHARunnerFixedParams fixedParams);

    ~FusedMHARunnerV2() = default;

    bool isFmhaSupported();

    bool isSeparateQAndKvInput() const
    {
        return mFixedParams.attentionInputLayout != AttentionInputLayout::PACKED_QKV;
    }

    void run(MHARunnerParams runnerParams);

private:
    void setupKernelParams(MHARunnerParams runnerParams);

    void setupLaunchParams(MHARunnerParams runnerParams);

    void setPackedQkvTmaDescriptors(MHARunnerParams runnerParams);

    void setSeparateQKvTmaDescriptors(MHARunnerParams runnerParams);

    bool isValidS(int s) const;

    int getSFromMaxSeqLen(int const max_seq_len) const;

private:
    MHARunnerFixedParams mFixedParams;
    MHARunnerParams mRunnerParams;
    Launch_params mLaunchParams;
    Fused_multihead_attention_params_v2 mKernelParams;
    int mSM = suggestify::common::getSMVersion();
    int mMultiProcessorCount;
    int mDeviceL2CacheSize;
    size_t mTotalDeviceMemory;
    FusedMultiHeadAttentionXMMAKernelV2 const* xmmaKernel;
};

}
}
