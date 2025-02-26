
#pragma once
#include "cubinObj.h"
#include "../src/decoderMaskedMultiheadAttention/xqaParams.h"
#include <nvrtc.h>
#include <string>
#include <vector>

namespace suggestify
{
namespace kernels
{
namespace jit
{

// A thin wrapper around NVRTC for compiling CUDA programs.
class CompileEngine
{
public:
    CompileEngine(int SM, XQAParams const& xqaParams);

    CubinObj compile() const;

    ~CompileEngine() = default;

private:
    int mSM;
    XQAParams const& mXqaParams;
};

} // namespace jit
} // namespace kernels
} // namespace suggestify
