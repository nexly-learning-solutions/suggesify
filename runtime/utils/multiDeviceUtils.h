
#pragma once

#include "../common/assert.h"
#include "../common/stringUtils.h"

#if ENABLE_MULTI_DEVICE
#include <mpi.h>
#include <nccl.h>

#define TLLM_MPI_CHECK(cmd)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        auto e = cmd;                                                                                                  \
        TLLM_CHECK_WITH_INFO(e == MPI_SUCCESS, "Failed: MPI error %s:%d '%d'", __FILE__, __LINE__, e);                 \
    } while (0)

#define TLLM_NCCL_CHECK(cmd)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        TLLM_CHECK_WITH_INFO(                                                                                          \
            r == ncclSuccess, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));           \
    } while (0)
#endif
