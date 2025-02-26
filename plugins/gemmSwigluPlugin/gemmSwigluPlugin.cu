

#include "gemmSwigluPlugin.h"

#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass_extensions/gemm_configs.h"

using namespace nvinfer1;
using namespace suggestify::common;
using namespace suggestify::kernels::cutlass_kernels;
using suggestify::plugins::GemmSwigluPluginCreator;
using suggestify::plugins::GemmSwigluPlugin;
using suggestify::plugins::GemmSwigluPluginProfiler;
using suggestify::plugins::read;
using suggestify::plugins::write;

void GemmSwigluPluginProfiler::initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream)
{
    size_t bpe = getBytePerElement(mType);

    if (mType == nvinfer1::DataType::kFP8)
    {
        cutlass::reference::device::BlockFillRandomUniform(reinterpret_cast<cutlass::float_e4m3_t*>(workspace),
            m * k + n * k + 1 * n, 42, cutlass::float_e4m3_t{128}, -cutlass::float_e4m3_t{128}, -1, stream);
    }
}
