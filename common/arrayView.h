
#pragma once

#include "assert.h"
#include <cstdint>

namespace suggestify::common
{

template <typename T>
class ArrayView
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = value_type const&;
    using pointer = T*;
    using const_pointer = T const*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    ArrayView(T* data, size_type size)
        : mData{data}
        , mSize{size}
    {
    }

    [[nodiscard]] iterator begin()
    {
        return mData;
    }

    [[nodiscard]] iterator end()
    {
        return mData + mSize;
    }

    [[nodiscard]] const_iterator begin() const
    {
        return mData;
    }

    [[nodiscard]] const_iterator end() const
    {
        return mData + mSize;
    }

    [[nodiscard]] const_iterator cbegin() const
    {
        return mData;
    }

    [[nodiscard]] const_iterator cend() const
    {
        return mData + mSize;
    }

    [[nodiscard]] size_type size() const
    {
        return mSize;
    }

    [[nodiscard]] reference operator[](size_type index)
    {
#ifdef INDEX_RANGE_CHECK
        CHECK_WITH_INFO(index < mSize, "Index %lu is out of bounds [0, %lu)", index, mSize);
#endif
        return mData[index];
    }

    [[nodiscard]] const_reference operator[](size_type index) const
    {
#ifdef INDEX_RANGE_CHECK
        CHECK_WITH_INFO(index < mSize, "Index %lu is out of bounds [0, %lu)", index, mSize);
#endif
        return mData[index];
    }

private:
    T* mData;
    size_type mSize;
};

}
