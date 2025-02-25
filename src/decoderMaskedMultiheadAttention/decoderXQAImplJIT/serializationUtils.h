
#pragma once
#include <cstddef>
#include <cstdint>

#include "suggestify/common/assert.h"

namespace suggestify
{
namespace kernels
{
namespace jit
{

template <typename T>
T readFromBuffer(uint8_t const*& buffer, size_t& remaining_buffer_size)
{
    CHECK(sizeof(T) <= remaining_buffer_size);

    T result = *reinterpret_cast<T const*>(buffer);
    buffer += sizeof(T);
    remaining_buffer_size -= sizeof(T);
    return result;
}

template <typename T>
void writeToBuffer(T output, uint8_t*& buffer, size_t& remaining_buffer_size)
{
    CHECK(sizeof(T) <= remaining_buffer_size);

    *reinterpret_cast<T*>(buffer) = output;
    buffer += sizeof(T);
    remaining_buffer_size -= sizeof(T);
}

} // namespace jit
} // namespace kernels
} // namespace suggestify
