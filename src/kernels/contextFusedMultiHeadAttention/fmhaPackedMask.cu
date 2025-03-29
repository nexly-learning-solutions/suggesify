#include "fmhaPackedMask.h"
#include "../common/assert.h"
#include "../common/cudaBf16Wrapper.h"
#include "../common/cudaFp8Utils.h"
#include "../common/cudaUtils.h"
#include "../common/mathUtils.h"
#include "../common/reduceKernelUtils.cuh"
#include <cub/cub.cuh>

namespace suggestify {
namespace kernels {


struct BlockPrefixCallbackOp {
  int mRunningTotal;

  __device__ BlockPrefixCallbackOp(int runningTotal) : mRunningTotal(runningTotal) {}

  __device__ int operator()(int blockAggregate) {
    int oldPrefix = mRunningTotal;
    mRunningTotal += blockAggregate;
    return oldPrefix;
  }
};


template <int THREADS_PER_BLOCK>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void buildCuMaskRows(int batchSize,
                                                                    const int *qSeqLens,
                                                                    int *cuQSeqLens,
                                                                    int *cuMaskRows) {
  using BlockScan = cub::BlockScan<int, THREADS_PER_BLOCK>;

  __shared__ typename BlockScan::TempStorage tempStorage;
  __shared__ typename BlockScan::TempStorage tempStorageForMask;

  BlockPrefixCallbackOp prefixOp(0);
  BlockPrefixCallbackOp prefixOpForMask(0);

  constexpr int batchSizeBound = THREADS_PER_BLOCK * divUp(batchSize, THREADS_PER_BLOCK);
  const bool storeOffsets = blockIdx.x == (batchSize - 1);

  for (int batchOffset = 0; batchOffset <= batchSizeBound; batchOffset += THREADS_PER_BLOCK) {
    int batchIdx = batchOffset + threadIdx.x;

    int maskRows = 0;
    int qSeqLen = 0;
    if (batchIdx < batchSize) {
      qSeqLen = qSeqLens[batchIdx];
      maskRows = divUp(qSeqLens[batchIdx], int(FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT)) *
                 FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT;
    }

    int qSeqLenOffset;
    int maskRowOffset;
    BlockScan(tempStorage).ExclusiveSum(qSeqLen, qSeqLenOffset, prefixOp);
    BlockScan(tempStorageForMask).ExclusiveSum(maskRows, maskRowOffset, prefixOpForMask);

    if (batchIdx < batchSize && storeOffsets) {
      if (cuQSeqLens) {
        cuQSeqLens[batchIdx] = qSeqLenOffset;
      }
      cuMaskRows[batchIdx] = maskRowOffset;
    }

    __syncthreads();
  }
}


template <typename MaskInputDataType, ContextAttentionMaskType MaskType>
__global__ void packFlashAttentionMask(PackedMaskParams<MaskInputDataType> params) {

  int batchIdx = blockIdx.y;
  int mmasNIdx = blockIdx.x;
  int mmasN = gridDim.x;

  int qSeqLenBound = params.actualQSeqLens[batchIdx];
  int kvSeqLenBound = params.actualKvSeqLens[batchIdx];

  size_t maskInputBatchOffset =
      (params.cuQSeqLens ? params.cuQSeqLens[batchIdx] : batchIdx * params.maxQSeqLen) *
      params.maxKvSeqLen;
  int actualMaskRows = params.cuMaskRows[batchIdx + 1] - params.cuMaskRows[batchIdx];
  int mmasM = actualMaskRows / (FLASH_ATTEN_WARPS_M * 16);
  int cuMmasM = params.cuMaskRows[batchIdx] / (FLASH_ATTEN_WARPS_M * 16);

  for (size_t mi = threadIdx.x; mi < mmasM; mi += blockDim.x) {
    for (size_t tidx = 0; tidx < NUM_THREADS_PER_WARP_GROUP; ++tidx) {
      size_t warp = tidx / 32;
      size_t lane = tidx % 32;

      size_t warpM = warp % FLASH_ATTEN_WARPS_M;
      size_t warpN = warp / FLASH_ATTEN_WARPS_M;
      size_t row = warpM * 16 + lane / 4;
      size_t col = warpN * 16 + lane % 4 * 2;

      row += mi * FLASH_ATTEN_WARPS_M * 16;
      col += mmasNIdx * NUM_CORE_MMAS_N * 8;

      size_t offset = maskInputBatchOffset + row * params.maxKvSeqLen + col;

      uint32_t mask = 0u;

      #pragma unroll
      for (size_t ni = 0; ni < NUM_CORE_MMAS_N; ++ni, offset += 8 * FLASH_ATTEN_WARPS_N,
                                         col += 8 * FLASH_ATTEN_WARPS_N) {
        bool validMasks[4] = {
            row < qSeqLenBound && col < kvSeqLenBound,
            row < qSeqLenBound && (col + 1) < kvSeqLenBound,
            (row + 8) < qSeqLenBound && col < kvSeqLenBound,
            (row + 8) < qSeqLenBound && (col + 1) < kvSeqLenBound,
        };

        if constexpr (MaskType == ContextAttentionMaskType::CUSTOM_MASK) {
          validMasks[0] = validMasks[0] && (params.maskInput[offset + 0 * params.maxKvSeqLen + 0] ==
                                            params.validPosVal);
          validMasks[1] = validMasks[1] && (params.maskInput[offset + 0 * params.maxKvSeqLen + 1] ==
                                            params.validPosVal);
          validMasks[2] = validMasks[2] && (params.maskInput[offset + 8 * params.maxKvSeqLen + 0] ==
                                            params.validPosVal);
          validMasks[3] = validMasks[3] && (params.maskInput[offset + 8 * params.maxKvSeqLen + 1] ==
                                            params.validPosVal);
        } else if constexpr (MaskType == ContextAttentionMaskType::CAUSAL) {
          validMasks[0] &= (col <= row);
          validMasks[1] &= ((col + 1) <= row);
          validMasks[2] &= (col <= (row + 8));
          validMasks[3] &= ((col + 1) <= (row + 8));
        } else if constexpr (MaskType == ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL) {
          validMasks[0] &= (col <= row) && (col > (row - params.slidingWindowSize));
          validMasks[1] &= ((col + 1) <= row) && ((col + 1) > (row - params.slidingWindowSize));
          validMasks[2] &= (col <= (row + 8) && col > ((row + 8) - params.slidingWindowSize));
          validMasks[3] &= ((col + 1) <= (row + 8) && ((col + 1) > ((row + 8) - params.slidingWindowSize)));
        }

        mask |= (validMasks[0] ? 1u : 0u) << (4 * ni + 0);
        mask |= (validMasks[1] ? 1u : 0u) << (4 * ni + 1);
        mask |= (validMasks[2] ? 1u : 0u) << (4 * ni + 2);
        mask |= (validMasks[3] ? 1u : 0u) << (4 * ni + 3);
      }

      size_t mOffset = (cuMmasM + mi) * mmasN * NUM_THREADS_PER_WARP_GROUP;
      size_t nOffset = mmasNIdx * NUM_THREADS_PER_WARP_GROUP;
      params.packedMask[mOffset + nOffset + tidx] = mask;
    }
  }
}


template <typename MaskInputDataType>
void invokeBuildPackedMask(const PackedMaskParams<MaskInputDataType> ¶ms, cudaStream_t stream) {
  buildCuMaskRows<256><<<params.batchSize, 256, 0, stream>>>(
      params.batchSize, params.actualQSeqLens, params.cuQSeqLens, params.cuMaskRows);
  sync_check_cuda_error();

  size_t mmasN = (divUp(params.maxKvSeqLen, size_t(FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT)) *
                  FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT) /
                 FLASH_ATTEN_PACKED_MASK_MMA_N;
  dim3 grid(mmasN, params.batchSize);

  if (params.attentionMaskType == ContextAttentionMaskType::PADDING) {
    packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::PADDING>
        <<<grid, 256, 0, stream>>>(params);
  } else if (params.attentionMaskType == ContextAttentionMaskType::CAUSAL) {
    packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::CAUSAL>
        <<<grid, 256, 0, stream>>>(params);
  } else if (params.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK) {
    packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::CUSTOM_MASK>
        <<<grid, 256, 0, stream>>>(params);
  } else if (params.attentionMaskType == ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL) {
    packFlashAttentionMask<MaskInputDataType, ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL>
        <<<grid, 256, 0, stream>>>(params);
  } else {
    CHECK_WITH_INFO(false, "The attention mask type is not supported.");
  }
  sync_check_cuda_error();
}


template void invokeBuildPackedMask(const PackedMaskParams<float> &, cudaStream_t);
template void invokeBuildPackedMask(const PackedMaskParams<half> &, cudaStream_t);
template void invokeBuildPackedMask(const PackedMaskParams<bool> &, cudaStream_t);
template void invokeBuildPackedMask(const PackedMaskParams<int> &, cudaStream_t);
#ifdef ENABLE_BF16
template void invokeBuildPackedMask(const PackedMaskParams<__nv_bfloat16> &, cudaStream_t);
#endif


}
}