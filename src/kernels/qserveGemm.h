
#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"

namespace suggestify
{
namespace kernels
{
namespace qserve
{

struct ParamsPerGroup
{
    int8_t const* A;
    int8_t const* B;
    int8_t const* s2_zeros;
    int8_t const* s2_scales;
    half const* s1_scales;
    half const* act_scales;
    half* C;
    int m;
    int n;
    int k;
};

struct ParamsPerChannel
{
    int8_t const* A;
    int8_t const* B;
    half const* s1_scales;
    half const* s1_szeros;
    half const* act_sums;
    half const* act_scales;
    half* C;
    int m;
    int n;
    int k;
};

class QServeGemmRunner
{
public:
    void gemmPerGroup(ParamsPerGroup const& params, cudaStream_t stream);
    void gemmPerChannel(ParamsPerChannel const& params, cudaStream_t stream);


    size_t getWorkspaceSize(int const m, int const n, int const k);

};

}
}
}
