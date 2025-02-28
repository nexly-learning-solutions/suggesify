
#pragma once
#include "../common/cudaUtils.h"
#include "../src/weightOnlyBatchedGemv/common.h"
#include "../src/weightOnlyBatchedGemv/details.h"

namespace sugesstify
{
namespace kernels
{
namespace weight_only
{
template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s);

inline void kernel_launcher(int arch, Params& params, cudaStream_t s)
{
#define EXEC(KType, A, B, Layout, ConverterInterleave)                                                                 \
    if (params.type == KType)                                                                                          \
    {                                                                                                                  \
        select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 64>>(       \
            params, s);                                                                                                \
        return;                                                                                                        \
    }
#define EXEC_W4A8(KType, A, B, Layout, ConverterInterleave)                                                            \
    if (params.type == KType && params.apply_alpha_in_advance)                                                         \
    {                                                                                                                  \
        select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 128>>(      \
            params, s);                                                                                                \
        return;                                                                                                        \
    }
    if (arch >= 75 && arch < 80)
    {
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::FP16Int4PerChannel, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    }
    else if (arch >= 80 && arch < 90)
    {
        if (arch >= 89)
        {
            EXEC_W4A8(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
            EXEC_W4A8(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
        }
        EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::FP16Int4PerChannel, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
        EXEC(KernelType::BF16Int4PerChannel, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    }
    else if (arch >= 90)
    {
        EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajor, false);
        EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajor, false);
        EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajor, false);
        EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajor, false);
        EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajor, false);
        EXEC(KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajor, false);
        EXEC(KernelType::FP16Int4PerChannel, FP16DetailsA, Int4DetailsW, ColumnMajor, false);
        EXEC(KernelType::BF16Int4PerChannel, BF16DetailsA, Int4DetailsW, ColumnMajor, false);
    }
#undef EXEC
}

inline bool is_supported(int arch, KernelType kernel_type)
{
#define SUPPORT(Type)                                                                                                  \
    if (kernel_type == Type)                                                                                           \
        return true;
    if (arch >= 75 && arch < 80)
    {
        SUPPORT(KernelType::FP16Int4Groupwise);
        SUPPORT(KernelType::FP16Int8PerChannel);
        SUPPORT(KernelType::FP16Int4PerChannel);
    }
    else if (arch >= 80 && arch < 90)
    {
        SUPPORT(KernelType::FP16Int8Groupwise);
        SUPPORT(KernelType::BF16Int8Groupwise);
        SUPPORT(KernelType::FP16Int4Groupwise);
        SUPPORT(KernelType::BF16Int4Groupwise);
        SUPPORT(KernelType::FP16Int8PerChannel);
        SUPPORT(KernelType::BF16Int8PerChannel);
        SUPPORT(KernelType::FP16Int4PerChannel);
        SUPPORT(KernelType::BF16Int4PerChannel);
    }
    else if (arch >= 90)
    {
        SUPPORT(KernelType::FP16Int8Groupwise);
        SUPPORT(KernelType::BF16Int8Groupwise);
        SUPPORT(KernelType::FP16Int4Groupwise);
        SUPPORT(KernelType::BF16Int4Groupwise);
        SUPPORT(KernelType::FP16Int8PerChannel);
        SUPPORT(KernelType::BF16Int8PerChannel);
        SUPPORT(KernelType::FP16Int4PerChannel);
        SUPPORT(KernelType::BF16Int4PerChannel);
    }
    return false;
#undef SUPPORT
}
}
}
}
