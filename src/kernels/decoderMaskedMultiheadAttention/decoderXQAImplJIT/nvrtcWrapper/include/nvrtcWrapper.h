
#pragma once
#include <stddef.h>

#ifdef _WIN32

#if COMPILING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#else             // _WIN32
#define DLLEXPORT // Nothing.
#endif

#if __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        // sm >= 80
        XQA_JIT_HMMA = 0,
        // sm == 90
        XQA_JIT_QGMMA = 1
    } tllmXqaJitKernelType;

    typedef struct
    {
        // Compute capability, e.g. 89.
        int sm;

        unsigned int head_size;
        unsigned int num_q_heads;
        unsigned int num_kv_heads;
        unsigned int beam_width;
        unsigned int tokens_per_block;
        bool multi_query_tokens;
        bool paged_kv_cache;

        // Actual type: suggestify::kernels::Data_type
        int data_type;
        int kv_cache_data_type;

        tllmXqaJitKernelType kernel_type;
    } tllmXqaJitContext;

    // tllmXqaJitProgram is an opaque handle for a program.
    typedef struct _tllmXqaJitProgram* tllmXqaJitProgram;

    typedef enum
    {
        XQA_JIT_SUCCESS = 0,
        XQA_JIT_INVALID_INPUT = 1,
        XQA_JIT_INTERNAL_ERROR = 2,
    } tllmXqaJitStatus;

    // context must outlive prog.
    DLLEXPORT tllmXqaJitStatus tllmXqaJitCreateAndCompileProgram(
        tllmXqaJitProgram* prog, tllmXqaJitContext const* context);
    DLLEXPORT tllmXqaJitStatus tllmXqaJitGetCUBINSize(tllmXqaJitProgram prog, size_t* cubinSizeRet);
    DLLEXPORT tllmXqaJitStatus tllmXqaJitGetCUBIN(tllmXqaJitProgram prog, char* cubin);
    DLLEXPORT tllmXqaJitStatus tllmXqaJitDestroyProgram(tllmXqaJitProgram* prog);

    // Returns the size of the error string associated with the last non-success tllmXqaJit function call (including the
    // trailing \0). Returns 0 if there is no such non-success function call.
    DLLEXPORT size_t tllmXqaJitGetLastErrorStringSize();
    // Returns the error string.
    // Output can be nullptr if the returned value of tllmGetLastErrorStringSize() is 0.
    DLLEXPORT void tllmXqaJitGetLastErrorString(char* output);

#if __cplusplus
} // extern "C"
#endif
