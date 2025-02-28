#pragma once

#include "cutlass/arch/mma_sm90.h"
#include "cutlass_extensions/epilogue_helpers.h"

namespace suggestify::kernels::cutlass_kernels
{

template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault>
constexpr bool isValidHopperMOESpecialisation()
{
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    return cutlass::platform::is_same<T, WeightType>::value
        && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#else
    return false;
#endif
}

template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault>
constexpr bool isValidAmpereMOESpecialisation()
{
    return true;
}

}
