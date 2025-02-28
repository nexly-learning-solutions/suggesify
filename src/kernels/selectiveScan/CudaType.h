
#pragma once

#include <cuda_fp8.h>
#include <mma.h>

typedef __nv_half fp16_t;
typedef __nv_bfloat16 bf16_t;
typedef float fp32_t;
typedef double fp64_t;
typedef __nv_fp8_e4m3 e4m3_t;
typedef __nv_fp8_e5m2 e5m2_t;

enum CudaType
{
    CT_FP16,
    CT_BF16,
    CT_FP32,
    CT_FP64,
    CT_E4M3,
    CT_E5M2,
};

template <CudaType>
struct EnToTp;

template <>
struct EnToTp<CT_FP16>
{
    typedef fp16_t type;
};

template <>
struct EnToTp<CT_BF16>
{
    typedef bf16_t type;
};

template <>
struct EnToTp<CT_FP32>
{
    typedef fp32_t type;
};

template <>
struct EnToTp<CT_FP64>
{
    typedef fp64_t type;
};

template <>
struct EnToTp<CT_E4M3>
{
    typedef e4m3_t type;
};

template <>
struct EnToTp<CT_E5M2>
{
    typedef e5m2_t type;
};

template <CudaType en_>
using EnToTp_t = typename EnToTp<en_>::type;

template <class Tp_>
struct TpToEn;

template <>
struct TpToEn<fp16_t>
{
    static constexpr CudaType value = CT_FP16;
};

template <>
struct TpToEn<bf16_t>
{
    static constexpr CudaType value = CT_BF16;
};

template <>
struct TpToEn<fp32_t>
{
    static constexpr CudaType value = CT_FP32;
};

template <>
struct TpToEn<fp64_t>
{
    static constexpr CudaType value = CT_FP64;
};

template <>
struct TpToEn<e4m3_t>
{
    static constexpr CudaType value = CT_E4M3;
};

template <>
struct TpToEn<e5m2_t>
{
    static constexpr CudaType value = CT_E5M2;
};

template <class Tp_>
constexpr CudaType TpToEn_v = TpToEn<Tp_>::value;

