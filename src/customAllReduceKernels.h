
#pragma once

#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../common/assert.h"
#include "../common/cudaUtils.h"

namespace suggestify::kernels
{

constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
constexpr size_t MAX_RANKS_PER_NODE = 8;
constexpr size_t DEFAULT_BLOCK_SIZE = 512;

namespace reduce_fusion::details
{
static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;
static constexpr int kClusterMaxSize = 8;
static constexpr int kLamportTokenNumThreshold = 16;
static constexpr int kLamportHiddenSizeThreshold = 256;
};

enum class AllReduceStrategyType : int8_t
{
    NCCL = 0,
    ONESHOT = 1,
    TWOSHOT = 2,
    UB = 3,
    AUTO = 4,
};

enum class AllReduceStrategyConfig : int8_t
{
    USE_MEMCPY = 1 << 0,
    PUSH_MODE = 1 << 1,
};

enum class AllReduceFusionOp : int8_t
{
    NONE = 0,
    RESIDUAL_RMS_NORM = 1,
    LAST_PROCESS_FOR_UB = 2,
    RESIDUAL_RMS_PREPOST_NORM = 3,
};

struct AllReduceFusionParams
{
    AllReduceFusionParams()
        : bias_buffer(nullptr)
        , residual_buffer(nullptr)
        , weight_buffer(nullptr)
        , weight_buffer_pre_residual_norm(nullptr)
        , intermediate_buffer(nullptr)
    {
    }

    void const* bias_buffer;
    void const* residual_buffer;
    int hidden_size;
    void const* weight_buffer;
    void const* weight_buffer_pre_residual_norm;
    float eps;
    void* intermediate_buffer;
    void* lamport_peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE * 3];
};

struct AllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node;
    size_t local_rank;
    uint32_t barrier_flag;
    uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void* local_output_buffer_ptr;
    void const* local_input_buffer_ptr;

    AllReduceFusionParams fusion_params;

    static AllReduceParams deserialize(int64_t* buffer, size_t tpSize, size_t tpRank, nvinfer1::DataType dataType,
        int token_num, int hidden_size, AllReduceFusionOp op);
};

bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks, nvinfer1::DataType type);

void customAllReduce(kernels::AllReduceParams& params, nvinfer1::DataType dataType, AllReduceStrategyType strat,
    AllReduceStrategyConfig config, AllReduceFusionOp fusionOp, cudaStream_t stream);

void residualRmsNorm(
    kernels::AllReduceParams& params, nvinfer1::DataType dataType, cudaStream_t stream, AllReduceFusionOp fusionOp);

void lamportInitialize(void* buffer, size_t size, nvinfer1::DataType dataType, cudaStream_t stream);

}
