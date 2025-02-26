
#include "tllmBuffers.h"

namespace suggestify::runtime
{

template <typename TAllocator>
typename PoolAllocator<TAllocator>::PoolType& PoolAllocator<TAllocator>::getPool()
{
    static PoolType pool;
    return pool;
}

template class PoolAllocator<PinnedAllocator>;
}
