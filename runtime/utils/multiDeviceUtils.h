
#pragma once

#include "../common/assert.h"
#include "../common/stringUtils.h"

#if ENABLE_MULTI_DEVICE
#include <mpi.h>
#include <nccl.h>

#define MPI_CHECK(cmd)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        auto e = cmd;                                                                                                  \
        CHECK_WITH_INFO(e == MPI_SUCCESS, "Failed: MPI error %s:%d '%d'", __FILE__, __LINE__, e);                 \
    } while (0)

#define NCCL_CHECK(cmd)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        CHECK_WITH_INFO(                                                                                          \
            r == ncclSuccess, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));           \
    } while (0)
#endif
