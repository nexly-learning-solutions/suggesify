
#pragma once

#include "../common/assert.h"
#include <limits.h>
#include <stdint.h>

namespace suggestify
{
namespace kernels
{
enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};


static inline size_t get_size_in_bytes(size_t n, Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_FP32: return n * 4;
    case DATA_TYPE_FP16: return n * 2;
    case DATA_TYPE_INT32: return n * 4;
    case DATA_TYPE_INT8: return n;
    case DATA_TYPE_BF16: return n * 2;
    case DATA_TYPE_E4M3: return n;
    case DATA_TYPE_E5M2: return n;
    default: TLLM_CHECK_WITH_INFO(false, "FMHA Data Type is not supported."); return 0;
    }
}


static inline size_t get_size_in_bytes(Data_type dtype)
{
    return get_size_in_bytes(1, dtype);
}


constexpr int32_t kSM_70 = 70;
constexpr int32_t kSM_72 = 72;
constexpr int32_t kSM_75 = 75;
constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;

}
}
