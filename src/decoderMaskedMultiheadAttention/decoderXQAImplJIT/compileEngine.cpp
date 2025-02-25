
#include "compileEngine.h"

#include "cubinObj.h"
#include "nvrtcWrapper/include/nvrtcWrapper.h"
#include "suggestify/common/assert.h"
#include "suggestify/common/stringUtils.h"
#include "suggestify/common/tllmException.h"
#include "suggestify/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include <string>
#include <vector>

namespace
{

void CHECK_XQA_JIT_ERROR_(tllmXqaJitStatus result, char const* const func, char const* const file, int const line)
{
    if (result != XQA_JIT_SUCCESS)
    {
        std::vector<char> log(tllmXqaJitGetLastErrorStringSize());
        tllmXqaJitGetLastErrorString(log.data());
        throw suggestify::common::TllmException(file, line,
            suggestify::common::fmtstr("[TensorRT-LLM][ERROR] TllmXqaJit runtime error in %s: %s", func, log.data()));
    }
}

#define CHECK_XQA_JIT_ERROR(val) CHECK_XQA_JIT_ERROR_((val), #val, __FILE__, __LINE__)

} // anonymous namespace

namespace suggestify
{
namespace kernels
{
namespace jit
{

CubinObj CompileEngine::compile() const
{
    tllmXqaJitProgram program;
    bool useQGMMAKernel = supportConfigQGMMA(mXqaParams, mSM, true);
    tllmXqaJitContext context{/*sm=*/mSM,
        /*head_size=*/static_cast<uint32_t>(mXqaParams.head_size),
        /*num_q_heads=*/static_cast<uint32_t>(mXqaParams.num_q_heads),
        /*num_kv_heads=*/static_cast<uint32_t>(mXqaParams.num_kv_heads),
        /*beam_width=*/static_cast<uint32_t>(mXqaParams.beam_width),
        /*tokens_per_block=*/static_cast<uint32_t>(mXqaParams.tokens_per_block),
        /*multi_query_tokens=*/mXqaParams.multi_query_tokens,
        /*paged_kv_cache=*/mXqaParams.paged_kv_cache,
        /*data_type=*/static_cast<int>(mXqaParams.data_type),
        /*kv_cache_data_type=*/static_cast<int>(mXqaParams.kv_cache_data_type),
        /*kernel_type=*/useQGMMAKernel ? XQA_JIT_QGMMA : XQA_JIT_HMMA};

    CHECK_XQA_JIT_ERROR(tllmXqaJitCreateAndCompileProgram(&program, &context));

    size_t cubinSize;
    CHECK_XQA_JIT_ERROR(tllmXqaJitGetCUBINSize(program, &cubinSize));
    std::string cubinContent(cubinSize, ' ');
    CHECK_XQA_JIT_ERROR(tllmXqaJitGetCUBIN(program, const_cast<char*>(cubinContent.c_str())));

    CHECK_XQA_JIT_ERROR(tllmXqaJitDestroyProgram(&program));

    return CubinObj(cubinContent);
}

CompileEngine::CompileEngine(int SM, XQAParams const& xqaParams)
    : mSM(SM)
    , mXqaParams(xqaParams)
{
}

} // namespace jit
} // namespace kernels
} // namespace suggestify
