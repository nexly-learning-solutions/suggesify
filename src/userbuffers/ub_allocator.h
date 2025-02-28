#pragma once
#include "../runtime/bufferManager.h"
#include "../runtime/iBuffer.h"
#include "../runtime/tllmBuffers.h"
#if ENABLE_MULTI_DEVICE
#include "userbuffers.h"
#endif
#include <unordered_map>
#include <vector>

namespace sugesstify::runtime::ub
{
static char const* tensor_prefix = "allreduce_ub_";

struct UBBuffer
{
    void* addr;
    int handle;
    size_t size;

    UBBuffer(void* a = nullptr, int h = -1, size_t s = 0)
        : addr(a)
        , handle(h)
        , size(s)
    {
    }

    bool invalid()
    {
        return (addr == nullptr) || (handle == -1) || (size == 0);
    }
};
#if ENABLE_MULTI_DEVICE
class UserBufferAllocator
{
public:
    static UserBufferAllocator& Instance();

    UserBufferAllocator() {}

    void initialize(int tp);
    bool is_initialized();
    UBBuffer register_ub_buffer(size_t bytes);
    void* allocate(int idx, size_t bytes);
    void deallocate(void* addr);
    UBBuffer get(int idx);
    communicator* comm();

private:
    communicator* ub_comm_;
    std::array<UBBuffer, 2> buffers_;
    bool is_initialized_;
    int tp_;
};
#else
using communicator = void;
#endif
};
