#pragma once

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <memory.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace sugesstify::runtime::ub
{

typedef enum
{
    ipcSocketSuccess = 0,
    ipcSocketUnhandledCudaError = 1,
    ipcSocketSystemError = 2,
    ipcSocketInternalError = 3,
    ipcSocketInvalidArgument = 4,
    ipcSocketInvalidUsage = 5,
    ipcSocketRemoteError = 6,
    ipcSocketInProgress = 7,
    ipcSocketNumResults = 8
} ipcSocketResult_t;

char const* ipcSocketGetErrorString(ipcSocketResult_t res);

#define IPC_SOCKNAME_LEN 64

struct IpcSocketHandle
{
    int fd;
    char socketName[IPC_SOCKNAME_LEN];
    uint32_t volatile* abortFlag;
};

ipcSocketResult_t ipcSocketInit(IpcSocketHandle* handle, int rank, uint64_t hash, uint32_t volatile* abortFlag);
ipcSocketResult_t ipcSocketClose(IpcSocketHandle* handle);
ipcSocketResult_t ipcSocketGetFd(IpcSocketHandle* handle, int* fd);

ipcSocketResult_t ipcSocketRecvFd(IpcSocketHandle* handle, int* fd);
ipcSocketResult_t ipcSocketSendFd(IpcSocketHandle* handle, int const fd, int rank, uint64_t hash);
}
