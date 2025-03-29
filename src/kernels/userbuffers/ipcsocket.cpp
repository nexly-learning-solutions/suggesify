#include "ipcsocket.h"
#include "../common/assert.h"
#include "../common/logger.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <cstdio>

#if ENABLE_MULTI_DEVICE

namespace sugesstify::runtime::ub
{

constexpr auto ipcSocketResultStrings = std::array{
    "Success",
    "Unhandled CUDA error",
    "System error",
    "Internal error",
    "Invalid argument",
    "Invalid usage",
    "Remote error",
    "In progress",
};

std::string_view ipcSocketGetErrorString(ipcSocketResult_t res)
{
    if (static_cast<int>(res) < 0 || static_cast<int>(res) >= ipcSocketResultStrings.size())
    {
        return "Unknown error";
    }
    return ipcSocketResultStrings[static_cast<int>(res)];
}

constexpr auto IPC_SOCKNAME_STR = "/tmp/ub-ipc-socket-%d-%lx";


ipcSocketResult_t ipcSocketInit(IpcSocketHandle* handle, int rank, uint64_t hash, std::atomic<uint32_t>* abortFlag)
{
    if (!handle)
    {
        return ipcSocketInternalError;
    }

    handle->fd = -1;
    handle->socketName[0] = '\0';

    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd == -1)
    {
        LOG_WARNING("UDS: Socket creation error: %s", strerror(errno));
        return ipcSocketSystemError;
    }

    sockaddr_un cliaddr{};
    cliaddr.sun_family = AF_UNIX;

    char temp[IPC_SOCKNAME_LEN];
    const int len = std::snprintf(temp, IPC_SOCKNAME_LEN, IPC_SOCKNAME_STR, rank, hash);

    if (len < 0 || len >= IPC_SOCKNAME_LEN)
    {
        LOG_WARNING("UDS: Socket name too long, rank %d, hash %lx", rank, hash);
        close(fd);
        return ipcSocketInternalError;
    }

    std::string_view socket_name(temp, len);

#ifdef USE_ABSTRACT_SOCKET
    cliaddr.sun_path[0] = '\0';
    std::memcpy(cliaddr.sun_path + 1, temp, len);
    socklen_t addrlen = 1 + len;
#else
    std::strncpy(cliaddr.sun_path, temp, sizeof(cliaddr.sun_path) - 1);
    cliaddr.sun_path[sizeof(cliaddr.sun_path) - 1] = '\0';
    socklen_t addrlen = SUN_LEN(&cliaddr);
    if (unlink(temp) == -1 && errno != ENOENT) {
        LOG_WARNING("UDS: Unlink failed for %s: %s", temp, strerror(errno));
        close(fd);
        return ipcSocketSystemError;
    }
#endif

    if (bind(fd, reinterpret_cast<sockaddr*>(&cliaddr), addrlen) == -1)
    {
        LOG_WARNING("UDS: Binding to socket %s failed: %s", socket_name.data(), strerror(errno));
        close(fd);
        return ipcSocketSystemError;
    }

    handle->fd = fd;
    std::strncpy(handle->socketName, temp, sizeof(handle->socketName) - 1);
    handle->socketName[sizeof(handle->socketName) - 1] = '\0';

    if (abortFlag)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1)
        {
            LOG_WARNING("UDS: setting non-blocking failed %s", strerror(errno));
        }
    }

    return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketGetFd(IpcSocketHandle* handle, int* fd)
{
    if (!handle)
    {
        LOG_WARNING("ipcSocketGetFd: null socket handle");
        return ipcSocketInvalidArgument;
    }
    if (!fd)
    {
        LOG_WARNING("ipcSocketGetFd: null fd pointer");
        return ipcSocketInvalidArgument;
    }
    *fd = handle->fd;
    return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketClose(IpcSocketHandle* handle)
{
    if (!handle)
    {
        return ipcSocketInternalError;
    }
    if (handle->fd == -1)
    {
        return ipcSocketSuccess;
    }
#ifndef USE_ABSTRACT_SOCKET
    if (handle->socketName[0] != '\0')
    {
        if (unlink(handle->socketName) == -1 && errno != ENOENT)
        {
            LOG_WARNING("UDS: Unlink failed for %s: %s", handle->socketName, strerror(errno));
        }
    }
#endif
    if (close(handle->fd) == -1)
    {
        LOG_WARNING("UDS: Close failed for fd %d: %s", handle->fd, strerror(errno));
    }
    handle->fd = -1;

    return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketRecvMsg(IpcSocketHandle* handle, void* hdr, int hdrLen, int* recvFd)
{
    if (!handle)
    {
        LOG_WARNING("ipcSocketRecvMsg: null handle");
        return ipcSocketInternalError;
    }

    msghdr msg{};
    iovec iov[1];

    union
    {
        cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    char dummy_buffer[1];

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);


    if (!hdr)
    {
        iov[0].iov_base = dummy_buffer;
        iov[0].iov_len = sizeof(dummy_buffer);
    }
    else
    {
        iov[0].iov_base = hdr;
        iov[0].iov_len = hdrLen;
    }

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;


    int ret;
    while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0)
    {
        if (ret == 0)
        {
            LOG_DEBUG("UDS: Connection closed by peer");
            return ipcSocketRemoteError;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            LOG_WARNING("UDS: Receiving data over socket failed: %s", strerror(errno));
            return ipcSocketSystemError;
        }
        if (handle->abortFlag && handle->abortFlag->load())
        {
            LOG_DEBUG("UDS: Aborting recv due to abort flag");
            return ipcSocketInternalError;
        }
    }

    if (recvFd)
    {
        cmsghdr* cmptr = CMSG_FIRSTHDR(&msg);
        if (cmptr && cmptr->cmsg_len == CMSG_LEN(sizeof(int)))
        {
            if (cmptr->cmsg_level != SOL_SOCKET || cmptr->cmsg_type != SCM_RIGHTS)
            {
                LOG_WARNING("UDS: Invalid control message received");
                return ipcSocketSystemError;
            }

            std::memcpy(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
        }
        else
        {
            LOG_WARNING("UDS: No file descriptor received");
            return ipcSocketSystemError;
        }
    }

    return ipcSocketSuccess;
}


ipcSocketResult_t ipcSocketRecvFd(IpcSocketHandle* handle, int* recvFd)
{
    return ipcSocketRecvMsg(handle, nullptr, 0, recvFd);
}

ipcSocketResult_t ipcSocketSendMsg(
    IpcSocketHandle* handle, void* hdr, int hdrLen, int const sendFd, int rank, uint64_t hash)
{
    if (!handle)
    {
        LOG_WARNING("ipcSocketSendMsg: null handle");
        return ipcSocketInternalError;
    }

    msghdr msg{};
    iovec iov[1];

    union
    {
        cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    char dummy_buffer[1];
    sockaddr_un cliaddr{};

    const int len = std::snprintf(cliaddr.sun_path, sizeof(cliaddr.sun_path), IPC_SOCKNAME_STR, rank, hash);
    if (len < 0 || len >= static_cast<int>(sizeof(cliaddr.sun_path)))
    {
        LOG_WARNING("UDS: Socket name too long, rank %d, hash %lx", rank, hash);
        return ipcSocketInternalError;
    }

    cliaddr.sun_family = AF_UNIX;
    std::string_view socket_name(cliaddr.sun_path, len);

#ifdef USE_ABSTRACT_SOCKET
    cliaddr.sun_path[0] = '\0';
    std::memcpy(cliaddr.sun_path + 1, socket_name.data(), len);
    socklen_t addrlen = 1 + len;
#else
    socklen_t addrlen = SUN_LEN(&cliaddr);
#endif

    if (sendFd != -1)
    {
        msg.msg_control = control_un.control;
        msg.msg_controllen = sizeof(control_un.control);

        cmsghdr* cmptr = CMSG_FIRSTHDR(&msg);
        cmptr->cmsg_len = CMSG_LEN(sizeof(int));
        cmptr->cmsg_level = SOL_SOCKET;
        cmptr->cmsg_type = SCM_RIGHTS;
        std::memcpy(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
    }

    msg.msg_name = reinterpret_cast<void*>(&cliaddr);
    msg.msg_namelen = addrlen;

    if (!hdr)
    {
        iov[0].iov_base = dummy_buffer;
        iov[0].iov_len = sizeof(dummy_buffer);
    }
    else
    {
        iov[0].iov_base = hdr;
        iov[0].iov_len = hdrLen;
    }
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_flags = 0;

    ssize_t sendResult;
    while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            LOG_WARNING("UDS: Sending data over socket %s failed: %s", socket_name.data(), strerror(errno));
            return ipcSocketSystemError;
        }
        if (handle->abortFlag && handle->abortFlag->load())
        {
            LOG_DEBUG("UDS: Aborting send due to abort flag");
            return ipcSocketInternalError;
        }
    }

    return ipcSocketSuccess;
}


ipcSocketResult_t ipcSocketSendFd(IpcSocketHandle* handle, int const sendFd, int rank, uint64_t hash)
{
    return ipcSocketSendMsg(handle, nullptr, 0, sendFd, rank, hash);
}

}
#endif