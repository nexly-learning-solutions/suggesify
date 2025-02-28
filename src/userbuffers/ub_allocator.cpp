#include "ub_allocator.h"

namespace sugesstify::runtime::ub
{
UserBufferAllocator& UserBufferAllocator::Instance()
{
    static UserBufferAllocator _;
    return _;
}

void UserBufferAllocator::initialize(int tp)
{
    if (!is_initialized())
    {
        ub_comm_ = nullptr;
        create_communicator_grouped2(&ub_comm_, 1, 1, tp, 1);
        TLLM_CHECK(ub_comm_ != nullptr);
        is_initialized_ = true;
        tp_ = tp;
    }
}

bool UserBufferAllocator::is_initialized()
{
    return is_initialized_;
}

UBBuffer UserBufferAllocator::register_ub_buffer(size_t bytes)
{
    TLLM_CHECK(is_initialized());
    void* addr = nullptr;
    int handle = -1;
    handle = register_user_buffer_collective((void**) &addr, bytes, ub_comm_, true);
    return {addr, handle, bytes};
}

void* UserBufferAllocator::allocate(int idx, size_t bytes)
{
    TLLM_CHECK(is_initialized() && idx < buffers_.size() && buffers_[idx].invalid());
    auto ub_buffer = register_ub_buffer(bytes);
    TLLM_CHECK(!ub_buffer.invalid());
    buffers_[idx] = ub_buffer;
    return ub_buffer.addr;
}

void UserBufferAllocator::deallocate(void* addr) {}

UBBuffer UserBufferAllocator::get(int idx)
{
    TLLM_CHECK(is_initialized() && idx < buffers_.size() && !buffers_[idx].invalid());
    return buffers_[idx];
}

communicator* UserBufferAllocator::comm()
{
    TLLM_CHECK(is_initialized());
    return ub_comm_;
}
};
