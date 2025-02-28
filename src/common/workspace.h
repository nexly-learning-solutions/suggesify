#pragma once
#include <cstddef>
#include <cstdint>

namespace suggestify::common
{

std::uintptr_t constexpr kCudaMemAlign = 128;

inline int8_t* alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}

constexpr size_t alignSize(size_t size, size_t to)
{
    if ((size % to) != 0U)
    {
        size += to - size % to;
    }
    return size;
}

inline int8_t* nextWorkspacePtrCommon(int8_t* ptr, uintptr_t previousWorkspaceSize, uintptr_t const alignment)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*) addr, alignment);
}

inline int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, kCudaMemAlign);
}

inline int8_t* nextWorkspacePtr(
    int8_t* const base, uintptr_t& offset, uintptr_t const size, uintptr_t const alignment = kCudaMemAlign)
{
    uintptr_t curr_offset = offset;
    uintptr_t next_offset = curr_offset + ((size + alignment - 1) / alignment) * alignment;
    int8_t* newptr = size == 0 ? nullptr : base + curr_offset;
    offset = next_offset;
    return newptr;
}

inline int8_t* nextWorkspacePtrWithAlignment(
    int8_t* ptr, uintptr_t previousWorkspaceSize, uintptr_t const alignment = kCudaMemAlign)
{
    return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, alignment);
}

inline size_t calculateTotalWorkspaceSize(
    size_t const* workspaces, int count, uintptr_t const alignment = kCudaMemAlign)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % alignment)
        {
            total += alignment - (workspaces[i] % alignment);
        }
    }
    return total;
}

};
