
#include <cuda_runtime_api.h>

#include "topkLastDim.h"
#include <cub/cub.cuh>
#include <cuda/atomic>

namespace suggestify
{
namespace kernels
{
using SizeType32 = suggestify::runtime::SizeType32;



#if 1

namespace air_topk_stable
{
using WideT = float4;
constexpr int VECTORIZED_READ_SIZE = 16;
constexpr int WARP_SIZE = 32;


template <typename IdxT>
struct ComputeOffset
{
    __host__ __device__ explicit ComputeOffset(IdxT const& cols)
        : cols_(cols)
    {
    }

    __host__ __device__ IdxT operator()(IdxT const& x) const
    {
        return cols_ * x;
    }

    IdxT cols_;
};

template <int BitsPerPass>
__host__ __device__ constexpr int calc_num_buckets()
{
    return 1 << BitsPerPass;
}

template <typename IntType>
constexpr __host__ __device__ IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}

template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calc_num_passes()
{
    return ceildiv<int>(sizeof(T) * 8, BitsPerPass);
}

__host__ __device__ int round(int num, int round_value)
{
    return ((num - 1) / round_value + 1) * round_value;
}

template <typename T, int BitsPerPass>
__device__ constexpr int calc_start_bit(int pass)
{
    int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    if (start_bit < 0)
    {
        start_bit = 0;
    }
    return start_bit;
}

template <typename T, int BitsPerPass>
__device__ constexpr unsigned calc_mask(int pass)
{
    static_assert(BitsPerPass <= 31);
    int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
    return (1 << num_bits) - 1;
}

template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddle_in(T key, bool select_min)
{
    auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
    bits = cub::Traits<T>::TwiddleIn(bits);
    if (!select_min)
    {
        bits = ~bits;
    }
    return bits;
}

template <typename T>
__device__ T twiddle_out(typename cub::Traits<T>::UnsignedBits bits, bool select_min)
{
    if (!select_min)
    {
        bits = ~bits;
    }
    bits = cub::Traits<T>::TwiddleOut(bits);
    return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool select_min)
{
    static_assert(BitsPerPass <= sizeof(int) * 8 - 1, "BitsPerPass is too large that the result type could not be int");
    return (twiddle_in(x, select_min) >> start_bit) & mask;
}

template <typename I>
constexpr inline std::enable_if_t<std::is_integral<I>::value, bool> is_a_power_of_two(I val) noexcept
{
    return ((val - 1) & val) == 0;
}

template <typename T, typename IdxT, typename RATIO_T = float>
__host__ __device__ IdxT calc_buf_len(IdxT len)
{
    constexpr RATIO_T ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
    IdxT buf_len = len / (ratio * 8);

    static_assert(is_a_power_of_two(sizeof(T)));
    static_assert(is_a_power_of_two(sizeof(IdxT)));
    constexpr IdxT aligned = 256 / std::min(sizeof(T), sizeof(IdxT));
    buf_len = buf_len & (~(aligned - 1));
    return buf_len;
}

template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, T const* in, idxT len, Func f)
{
    if constexpr (sizeof(T) >= sizeof(WideT))
    {
        for (idxT i = thread_rank; i < len; i += num_threads)
        {
            f(in[i], i);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

        union
        {
            WideT scalar;
            T array[items_per_scalar];
        } wide;

        int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
        if (skip_cnt > len)
        {
            skip_cnt = len;
        }
        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        const idxT len_cast = (len - skip_cnt) / items_per_scalar;

        for (idxT i = thread_rank; i < len_cast; i += num_threads)
        {
            wide.scalar = in_cast[i];
            const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for (int j = 0; j < items_per_scalar; ++j)
            {
                f(wide.array[j], real_i + j);
            }
        }

        static_assert(WARP_SIZE >= items_per_scalar);
        if (thread_rank < skip_cnt)
        {
            f(in[thread_rank], thread_rank);
        }
        const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if (remain_i < len)
        {
            f(in[remain_i], remain_i);
        }
    }
}

template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(T const* in, idxT len, Func f, int sync_width)
{
    const idxT stride = blockDim.x * gridDim.x;
    const idxT tid = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (sizeof(T) >= sizeof(WideT))
    {
        for (idxT i = tid; i < len; i += stride)
        {
            f(in[i], i, true);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

        union
        {
            WideT scalar;
            T array[items_per_scalar];
        } wide;

        int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
            : 0;
        if (skip_cnt > len)
        {
            skip_cnt = len;
        }
        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        const idxT len_cast = (len - skip_cnt) / items_per_scalar;

        const idxT len_cast_for_sync = ((len_cast - 1) / sync_width + 1) * sync_width;
        for (idxT i = tid; i < len_cast_for_sync; i += stride)
        {
            bool valid = i < len_cast;
            if (valid)
            {
                wide.scalar = in_cast[i];
            }
            const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for (int j = 0; j < items_per_scalar; ++j)
            {
                f(wide.array[j], real_i + j, valid);
            }
        }

        static_assert(WARP_SIZE >= items_per_scalar);
        if (tid < sync_width)
        {
            bool valid = tid < skip_cnt;
            T value = valid ? in[tid] : T();
            f(value, tid, valid);

            const idxT remain_i = skip_cnt + len_cast * items_per_scalar + tid;
            valid = remain_i < len;
            value = valid ? in[remain_i] : T();
            f(value, remain_i, valid);
        }
    }
}

template <typename T, typename IdxT>
struct alignas(128) Counter
{
    IdxT k;
    IdxT len;

    IdxT previous_len;

    typename cub::Traits<T>::UnsignedBits kth_value_bits;

    alignas(128) IdxT filter_cnt;

    alignas(128) unsigned int finished_block_cnt;

    alignas(128) IdxT out_cnt;

    alignas(128) IdxT out_back_cnt;
};

template <typename T, typename IdxT, int BitsPerPass>
__device__ void filter_and_histogram(T const* in_buf, IdxT const* in_idx_buf, T* out_buf, IdxT* out_idx_buf, T* out,
    IdxT* out_idx, IdxT previous_len, Counter<T, IdxT>* counter, IdxT* histogram, bool select_min, int pass,
    bool early_stop)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    __shared__ IdxT histogram_smem[num_buckets];
    for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        histogram_smem[i] = 0;
    }
    __syncthreads();

    int const start_bit = calc_start_bit<T, BitsPerPass>(pass);
    unsigned const mask = calc_mask<T, BitsPerPass>(pass);

    if (pass == 0)
    {
        auto f = [select_min, start_bit, mask](T value, IdxT)
        {
            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
            atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
        };
        vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len, f);
    }
    else
    {
        IdxT* p_filter_cnt = &counter->filter_cnt;
        IdxT* p_out_cnt = &counter->out_cnt;
        auto const kth_value_bits = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        auto f = [in_idx_buf, out_buf, out_idx_buf, out, out_idx, select_min, start_bit, mask, previous_start_bit,
                     kth_value_bits, p_filter_cnt, p_out_cnt, early_stop](T value, IdxT i)
        {
            const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit) << previous_start_bit;
            if (previous_bits == kth_value_bits)
            {
                if (early_stop)
                {
                    IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                    out[pos] = value;
                    out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
                }
                else
                {
                    if (out_buf)
                    {
                        IdxT pos = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
                        out_buf[pos] = value;
                        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
                    }

                    int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
                    atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
                }
            }
            else if ((out_buf || early_stop) && previous_bits < kth_value_bits)
            {
                IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                out[pos] = value;
                out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
            }
        };
        vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
            static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len, f);
    }
    if (early_stop)
    {
        return;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        if (histogram_smem[i] != 0)
        {
            atomicAdd(histogram + i, histogram_smem[i]);
        }
    }
}

template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(IdxT volatile* histogram)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    if constexpr (num_buckets >= BlockSize)
    {
        static_assert(num_buckets % BlockSize == 0);
        constexpr int items_per_thread = num_buckets / BlockSize;
        typedef cub::BlockLoad<IdxT, BlockSize, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
        typedef cub::BlockStore<IdxT, BlockSize, items_per_thread, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
        typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

        __shared__ union
        {
            typename BlockLoad::TempStorage load;
            typename BlockScan::TempStorage scan;
            typename BlockStore::TempStorage store;
        } temp_storage;

        IdxT thread_data[items_per_thread];

        BlockLoad(temp_storage.load).Load(histogram, thread_data);
        __syncthreads();

        BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
        __syncthreads();

        BlockStore(temp_storage.store).Store(histogram, thread_data);
    }
    else
    {
        typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;

        IdxT thread_data = 0;
        if (threadIdx.x < num_buckets)
        {
            thread_data = histogram[threadIdx.x];
        }

        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
        __syncthreads();

        if (threadIdx.x < num_buckets)
        {
            histogram[threadIdx.x] = thread_data;
        }
    }
}

template <typename T, typename IdxT, int BitsPerPass>
__device__ void choose_bucket(Counter<T, IdxT>* counter, IdxT const* histogram, const IdxT k, int const pass)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        IdxT prev = (i == 0) ? 0 : histogram[i - 1];
        IdxT cur = histogram[i];

        if (prev < k && cur >= k)
        {
            counter->k = k - prev;
            counter->len = cur - prev;
            typename cub::Traits<T>::UnsignedBits bucket = i;
            int start_bit = calc_start_bit<T, BitsPerPass>(pass);
            counter->kth_value_bits |= bucket << start_bit;
        }
    }
}

template <typename T, typename IdxT, int BitsPerPass, bool prioritize_smaller_indice = false>
__device__ void last_filter(T const* in_buf, IdxT const* in_idx_buf, T* out, IdxT* out_idx, IdxT current_len, IdxT k,
    Counter<T, IdxT>* counter, bool const select_min, int const pass)
{
    auto const kth_value_bits = counter->kth_value_bits;
    int const start_bit = calc_start_bit<T, BitsPerPass>(pass);

    const IdxT num_of_kth_needed = counter->k;
    IdxT* p_out_cnt = &counter->out_cnt;
    IdxT* p_out_back_cnt = &counter->out_back_cnt;
    IdxT* p_equal = out_idx + k - num_of_kth_needed;
    cuda::atomic_ref<IdxT, cuda::thread_scope_block> ref_last(p_equal[num_of_kth_needed - 1]);
    for (IdxT i = threadIdx.x; i < current_len; i += blockDim.x)
    {
        const T value = in_buf[i];
        auto const bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
        if (bits < kth_value_bits)
        {
            IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
            out[pos] = value;
            out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        }
        else if (bits == kth_value_bits)
        {
            IdxT new_idx = in_idx_buf ? in_idx_buf[i] : i;
            IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
            if (back_pos < num_of_kth_needed)
            {
                IdxT pos = k - 1 - back_pos;
                out[pos] = value;
                if constexpr (!prioritize_smaller_indice)
                {
                    out_idx[pos] = new_idx;
                }
            }
            if constexpr (prioritize_smaller_indice)
            {
                if (new_idx < ref_last.load(cuda::memory_order_relaxed))
                {
                    for (int j = 0; j < num_of_kth_needed; j++)
                    {
                        IdxT pre_idx = atomicMin(&p_equal[j], new_idx);
                        if (pre_idx > new_idx)
                        {
                            new_idx = pre_idx;
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename IdxT, int BitsPerPass, bool prioritize_smaller_indice = false>
__global__ void last_filter_kernel(T const* in, IdxT const* in_idx, T const* in_buf, IdxT const* in_idx_buf, T* out,
    IdxT* out_idx, IdxT len, IdxT k, Counter<T, IdxT>* counters, bool const select_min)
{
    const size_t batch_id = blockIdx.y;

    Counter<T, IdxT>* counter = counters + batch_id;
    IdxT previous_len = counter->previous_len;
    if (previous_len == 0)
    {
        return;
    }
    const IdxT buf_len = calc_buf_len<T>(len);
    if (previous_len > buf_len || in_buf == in)
    {
        in_buf = in + batch_id * len;
        in_idx_buf = in_idx ? (in_idx + batch_id * len) : nullptr;
        previous_len = len;
    }
    else
    {
        in_buf += batch_id * buf_len;
        in_idx_buf += batch_id * buf_len;
    }
    out += batch_id * k;
    out_idx += batch_id * k;

    constexpr int pass = calc_num_passes<T, BitsPerPass>() - 1;
    constexpr int start_bit = calc_start_bit<T, BitsPerPass>(pass);

    auto const kth_value_bits = counter->kth_value_bits;
    const IdxT num_of_kth_needed = counter->k;
    IdxT* p_out_cnt = &counter->out_cnt;
    IdxT* p_out_back_cnt = &counter->out_back_cnt;
    IdxT* p_equal = out_idx + k - num_of_kth_needed;
    cuda::atomic_ref<IdxT> ref_last(p_equal[num_of_kth_needed - 1]);
    auto f = [k, select_min, kth_value_bits, num_of_kth_needed, p_out_cnt, p_out_back_cnt, in_idx_buf, out, out_idx,
                 p_equal, ref_last](T value, IdxT i)
    {
        const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
        if (bits < kth_value_bits)
        {
            IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
            out[pos] = value;
            out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        }
        else if (bits == kth_value_bits)
        {
            IdxT new_idx = in_idx_buf ? in_idx_buf[i] : i;
            IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
            if (back_pos < num_of_kth_needed)
            {
                IdxT pos = k - 1 - back_pos;
                out[pos] = value;
                if constexpr (!prioritize_smaller_indice)
                {
                    out_idx[pos] = new_idx;
                }
            }
            if constexpr (prioritize_smaller_indice)
            {
                if (new_idx < ref_last.load(cuda::memory_order_relaxed))
                {
                    for (int j = 0; j < num_of_kth_needed; j++)
                    {
                        IdxT pre_idx = atomicMin(&p_equal[j], new_idx);
                        if (pre_idx > new_idx)
                        {
                            new_idx = pre_idx;
                        }
                    }
                }
            }
        }
    };

    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
        static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len, f);
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool fused_last_filter,
    bool prioritize_smaller_indice = false>
__global__ void radix_kernel(T const* in, IdxT const* in_idx, T const* in_buf, IdxT const* in_idx_buf, T* out_buf,
    IdxT* out_idx_buf, T* out, IdxT* out_idx, Counter<T, IdxT>* counters, IdxT* histograms, const IdxT len,
    const IdxT k, bool const select_min, int const pass)
{
    const size_t batch_id = blockIdx.y;
    auto counter = counters + batch_id;
    IdxT current_k;
    IdxT previous_len;
    IdxT current_len;
    if (pass == 0)
    {
        current_k = k;
        previous_len = len;
        current_len = len;
    }
    else
    {
        current_k = counter->k;
        current_len = counter->len;
        previous_len = counter->previous_len;
    }
    if (current_len == 0)
    {
        return;
    }

    bool const early_stop = (current_len == current_k);
    const IdxT buf_len = calc_buf_len<T>(len);

    if (pass == 0 || pass == 1 || previous_len > buf_len)
    {
        in_buf = in + batch_id * len;
        in_idx_buf = in_idx ? (in_idx + batch_id * len) : nullptr;
        previous_len = len;
    }
    else
    {
        in_buf += batch_id * buf_len;
        in_idx_buf += batch_id * buf_len;
    }
    if (pass == 0 || current_len > buf_len)
    {
        out_buf = nullptr;
        out_idx_buf = nullptr;
    }
    else
    {
        out_buf += batch_id * buf_len;
        out_idx_buf += batch_id * buf_len;
    }
    out += batch_id * k;
    out_idx += batch_id * k;

    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    auto histogram = histograms + batch_id * num_buckets;

    filter_and_histogram<T, IdxT, BitsPerPass>(in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx, previous_len,
        counter, histogram, select_min, pass, early_stop);
    __threadfence();

    bool isLastBlock = false;
    if (threadIdx.x == 0)
    {
        unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
        isLastBlock = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(isLastBlock))
    {
        if (early_stop)
        {
            if (threadIdx.x == 0)
            {
                counter->previous_len = 0;
                counter->len = 0;
            }
            return;
        }

        scan<IdxT, BitsPerPass, BlockSize>(histogram);
        __syncthreads();
        choose_bucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass);
        __syncthreads();

        constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
        if (pass != num_passes - 1)
        {
            for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
            {
                histogram[i] = 0;
            }
        }
        if (threadIdx.x == 0)
        {
            counter->previous_len = current_len;
            counter->filter_cnt = 0;
        }

        if (pass == num_passes - 1)
        {
            const volatile IdxT num_of_kth_needed = counter->k;
            for (IdxT i = threadIdx.x; i < num_of_kth_needed; i += blockDim.x)
            {
                out_idx[k - num_of_kth_needed + i] = cuda::std::numeric_limits<IdxT>::max();
            }
            __syncthreads();
            if constexpr (fused_last_filter)
            {
                last_filter<T, IdxT, BitsPerPass, prioritize_smaller_indice>(out_buf ? out_buf : in_buf,
                    out_idx_buf ? out_idx_buf : in_idx_buf, out, out_idx, out_buf ? current_len : len, k, counter,
                    select_min, pass);
            }
        }
    }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
unsigned calc_grid_dim(int batch_size, IdxT len, int sm_cnt)
{
    static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);

    int active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, true>, BlockSize, 0);
    active_blocks *= sm_cnt;

    IdxT best_num_blocks = 0;
    float best_tail_wave_penalty = 1.0f;
    const IdxT max_num_blocks = ceildiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize);
    for (int num_waves = 1;; ++num_waves)
    {
        IdxT num_blocks
            = std::min(max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
        IdxT items_per_thread = ceildiv<IdxT>(len, num_blocks * BlockSize);
        items_per_thread = alignTo<IdxT>(items_per_thread, VECTORIZED_READ_SIZE / sizeof(T));
        num_blocks = ceildiv<IdxT>(len, items_per_thread * BlockSize);
        float actual_num_waves = static_cast<float>(num_blocks) * batch_size / active_blocks;
        float tail_wave_penalty = (ceilf(actual_num_waves) - actual_num_waves) / ceilf(actual_num_waves);

        if (tail_wave_penalty < 0.15)
        {
            best_num_blocks = num_blocks;
            break;
        }
        else if (tail_wave_penalty < best_tail_wave_penalty)
        {
            best_num_blocks = num_blocks;
            best_tail_wave_penalty = tail_wave_penalty;
        }

        if (num_blocks == max_num_blocks)
        {
            break;
        }
    }
    return best_num_blocks;
}

template <typename T, typename IdxT>
__host__ __device__ void set_buf_pointers(T const* in, IdxT const* in_idx, T* buf1, IdxT* idx_buf1, T* buf2,
    IdxT* idx_buf2, int pass, T const*& in_buf, IdxT const*& in_idx_buf, T*& out_buf, IdxT*& out_idx_buf)
{
    if (pass == 0)
    {
        in_buf = in;
        in_idx_buf = nullptr;
        out_buf = nullptr;
        out_idx_buf = nullptr;
    }
    else if (pass == 1)
    {
        in_buf = in;
        in_idx_buf = in_idx;
        out_buf = buf1;
        out_idx_buf = idx_buf1;
    }
    else if (pass % 2 == 0)
    {
        in_buf = buf1;
        in_idx_buf = idx_buf1;
        out_buf = buf2;
        out_idx_buf = idx_buf2;
    }
    else
    {
        in_buf = buf2;
        in_idx_buf = idx_buf2;
        out_buf = buf1;
        out_idx_buf = idx_buf1;
    }
}

template <typename T, typename IdxT>
__device__ void set_buf_pointers(T const* in, IdxT const* in_idx, char* bufs, IdxT buf_len, int pass, T const*& in_buf,
    IdxT const*& in_idx_buf, T*& out_buf, IdxT*& out_idx_buf)
{
    if (pass == 0)
    {
        in_buf = in;
        in_idx_buf = nullptr;
        out_buf = nullptr;
        out_idx_buf = nullptr;
    }
    else if (pass == 1)
    {
        in_buf = in;
        in_idx_buf = in_idx;
        out_buf = reinterpret_cast<T*>(bufs);
        out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
    }
    else if (pass % 2 == 0)
    {
        in_buf = reinterpret_cast<T*>(bufs);
        in_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
        out_buf = const_cast<T*>(in_buf + buf_len);
        out_idx_buf = const_cast<IdxT*>(in_idx_buf + buf_len);
    }
    else
    {
        out_buf = reinterpret_cast<T*>(bufs);
        out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
        in_buf = out_buf + buf_len;
        in_idx_buf = out_idx_buf + buf_len;
    }
}

template <typename T, typename IdxT, int BitsPerPass>
__device__ void filter_and_histogram_for_one_block(T const* in_buf, IdxT const* in_idx_buf, T* out_buf,
    IdxT* out_idx_buf, T* out, IdxT* out_idx, const IdxT previous_len, Counter<T, IdxT>* counter, IdxT* histogram,
    bool select_min, int pass)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        histogram[i] = 0;
    }
    IdxT* p_filter_cnt = &counter->filter_cnt;
    if (threadIdx.x == 0)
    {
        *p_filter_cnt = 0;
    }
    __syncthreads();

    int const start_bit = calc_start_bit<T, BitsPerPass>(pass);
    unsigned const mask = calc_mask<T, BitsPerPass>(pass);

    if (pass == 0)
    {
        auto f = [histogram, select_min, start_bit, mask](T value, IdxT)
        {
            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
            atomicAdd(histogram + bucket, static_cast<IdxT>(1));
        };
        vectorized_process(threadIdx.x, blockDim.x, in_buf, previous_len, f);
    }
    else if (!out_buf)
    {
        auto const kth_value_bits = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x)
        {
            const T value = in_buf[i];
            auto const previous_bits = (twiddle_in(value, select_min) >> previous_start_bit) << previous_start_bit;
            if (previous_bits == kth_value_bits)
            {
                int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
                atomicAdd(histogram + bucket, static_cast<IdxT>(1));
            }
        }
    }
    else
    {
        IdxT* p_out_cnt = &counter->out_cnt;
        auto const kth_value_bits = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x)
        {
            const T value = in_buf[i];
            auto const previous_bits = (twiddle_in(value, select_min) >> previous_start_bit) << previous_start_bit;
            if (previous_bits == kth_value_bits)
            {
#if CUDART_VERSION < 12000
                volatile
#endif
                    IdxT pos
                    = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
                out_buf[pos] = value;
                out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;

                int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
                atomicAdd(histogram + bucket, static_cast<IdxT>(1));
            }
            else if (previous_bits < kth_value_bits)
            {
                IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                out[pos] = value;
                out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
            }
        }
    }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool prioritize_smaller_indice = false>
__global__ void radix_topk_one_block_kernel(T const* in, IdxT const* in_idx, const IdxT len, const IdxT k, T* out,
    IdxT* out_idx, bool const select_min, char* bufs)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    __shared__ Counter<T, IdxT> counter;
    __shared__ IdxT histogram[num_buckets];

    if (threadIdx.x == 0)
    {
        counter.k = k;
        counter.len = len;
        counter.previous_len = len;
        counter.kth_value_bits = 0;
        counter.out_cnt = 0;
        counter.out_back_cnt = 0;
    }
    __syncthreads();

    const size_t batch_id = blockIdx.x;
    in += batch_id * len;
    if (in_idx)
    {
        in_idx += batch_id * len;
    }

    out += batch_id * k;
    out_idx += batch_id * k;
    const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
    bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
    for (int pass = 0; pass < num_passes; ++pass)
    {
        T const* in_buf = nullptr;
        IdxT const* in_idx_buf = nullptr;
        T* out_buf = nullptr;
        IdxT* out_idx_buf = nullptr;
        set_buf_pointers(in, in_idx, bufs, buf_len, pass, in_buf, in_idx_buf, out_buf, out_idx_buf);

        const IdxT current_len = counter.len;
        const IdxT current_k = counter.k;
        IdxT previous_len = counter.previous_len;
        if (previous_len > buf_len)
        {
            in_buf = in;
            in_idx_buf = in_idx;
            previous_len = len;
        }
        if (current_len > buf_len)
        {
            out_buf = nullptr;
            out_idx_buf = nullptr;
        }

        filter_and_histogram_for_one_block<T, IdxT, BitsPerPass>(in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx,
            previous_len, &counter, histogram, select_min,
            pass);
        __syncthreads();

        scan<IdxT, BitsPerPass, BlockSize>(histogram);
        __syncthreads();

        choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
        if (threadIdx.x == 0)
        {
            counter.previous_len = current_len;
        }
        __syncthreads();

        if ((pass == num_passes - 1))
        {
            if constexpr (prioritize_smaller_indice)
            {
                const IdxT num_of_kth_needed = counter.k;
                for (IdxT i = threadIdx.x; i < num_of_kth_needed; i += blockDim.x)
                {
                    out_idx[k - num_of_kth_needed + i] = cuda::std::numeric_limits<IdxT>::max();
                }
                __syncthreads();
            }
            last_filter<T, IdxT, BitsPerPass, prioritize_smaller_indice>(out_buf ? out_buf : in,
                out_buf ? out_idx_buf : in_idx, out, out_idx, out_buf ? current_len : len, k, &counter, select_min,
                pass);
            break;
        }
        else if (counter.len == counter.k)
        {
            last_filter<T, IdxT, BitsPerPass, false>(out_buf ? out_buf : in, out_buf ? out_idx_buf : in_idx, out,
                out_idx, out_buf ? current_len : len, k, &counter, select_min, pass);
            break;
        }
    }
}
}



inline size_t calc_aligned_size(std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);
    size_t total = 0;
    for (auto sz : sizes)
    {
        total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }
    return total + ALIGN_BYTES - 1;
}

inline std::vector<void*> calc_aligned_pointers(void const* p, std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK = ~(ALIGN_BYTES - 1);

    char* ptr = reinterpret_cast<char*>((reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

    std::vector<void*> aligned_pointers;
    aligned_pointers.reserve(sizes.size());
    for (auto sz : sizes)
    {
        aligned_pointers.push_back(ptr);
        ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }

    return aligned_pointers;
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void standalone_stable_radix_topk_(void* buf, size_t& buf_size, T const* in, IdxT const* in_idx, int batch_size,
    IdxT len, IdxT k, T* out, IdxT* out_idx, bool select_min, bool fused_last_filter, unsigned grid_dim,
    cudaStream_t stream, bool sorted = false)
{
    static_assert(air_topk_stable::calc_num_passes<T, BitsPerPass>() > 1);
    constexpr int num_buckets = air_topk_stable::calc_num_buckets<BitsPerPass>();

    air_topk_stable::Counter<T, IdxT>* counters = nullptr;
    IdxT* histograms = nullptr;
    T* buf1 = nullptr;
    IdxT* idx_buf1 = nullptr;
    T* buf2 = nullptr;
    IdxT* idx_buf2 = nullptr;

    void* sort_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t temp_storage_bytes_sort = 0;
    T* topk_out = nullptr;
    IdxT* topk_out_idx = nullptr;
    T* sort_in = nullptr;
    IdxT* sort_in_idx = nullptr;

    air_topk_stable::ComputeOffset<IdxT> computeoffset(k);
    cub::CountingInputIterator<IdxT> counting_iter(0);
    cub::TransformInputIterator<IdxT, air_topk_stable::ComputeOffset<IdxT>, cub::CountingInputIterator<IdxT>>
        transform_iter(counting_iter, computeoffset);
    cub::DeviceSegmentedSort::SortPairs(NULL, temp_storage_bytes, out_idx, out_idx, out, out, k * batch_size,
        batch_size, transform_iter, transform_iter + 1, stream);
    if (sorted)
    {
        if (select_min)
        {
            cub::DeviceSegmentedSort::StableSortPairs(NULL, temp_storage_bytes_sort, out, out, out_idx, out_idx,
                k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
        else
        {
            cub::DeviceSegmentedSort::StableSortPairsDescending(NULL, temp_storage_bytes_sort, out, out, out_idx,
                out_idx, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
    }
    temp_storage_bytes = max(temp_storage_bytes, temp_storage_bytes_sort);

    {
        IdxT len_candidates = air_topk_stable::calc_buf_len<T>(len);
        size_t sort_buffer_size = 0;
        if (sorted)
        {
            sort_buffer_size = k * batch_size;
        }
        std::vector<size_t> sizes = {sizeof(*counters) * batch_size, sizeof(*histograms) * num_buckets * batch_size,
            sizeof(*buf1) * len_candidates * batch_size, sizeof(*idx_buf1) * len_candidates * batch_size,
            sizeof(*buf2) * len_candidates * batch_size, sizeof(*idx_buf2) * len_candidates * batch_size,
            temp_storage_bytes, sizeof(*topk_out) * k * batch_size, sizeof(*topk_out_idx) * k * batch_size,
            sizeof(*sort_in) * sort_buffer_size, sizeof(*sort_in_idx) * sort_buffer_size};
        size_t total_size = calc_aligned_size(sizes);
        if (!buf)
        {
            buf_size = total_size;
            return;
        }

        std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
        counters = static_cast<decltype(counters)>(aligned_pointers[0]);
        histograms = static_cast<decltype(histograms)>(aligned_pointers[1]);
        buf1 = static_cast<decltype(buf1)>(aligned_pointers[2]);
        idx_buf1 = static_cast<decltype(idx_buf1)>(aligned_pointers[3]);
        buf2 = static_cast<decltype(buf2)>(aligned_pointers[4]);
        idx_buf2 = static_cast<decltype(idx_buf2)>(aligned_pointers[5]);
        sort_temp_storage = aligned_pointers[6];
        topk_out = static_cast<decltype(topk_out)>(aligned_pointers[7]);
        topk_out_idx = static_cast<decltype(topk_out_idx)>(aligned_pointers[8]);
        if (sorted)
        {
            sort_in = static_cast<decltype(sort_in)>(aligned_pointers[9]);
            sort_in_idx = static_cast<decltype(sort_in_idx)>(aligned_pointers[10]);
        }
        cudaMemsetAsync(
            buf, 0, static_cast<char*>(aligned_pointers[2]) - static_cast<char*>(aligned_pointers[0]), stream);
    }

    T const* in_buf = nullptr;
    IdxT const* in_idx_buf = nullptr;
    T* out_buf = nullptr;
    IdxT* out_idx_buf = nullptr;

    dim3 blocks(grid_dim, batch_size);

    constexpr int num_passes = air_topk_stable::calc_num_passes<T, BitsPerPass>();

    auto kernel = air_topk_stable::radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, true>;

    for (int pass = 0; pass < num_passes; ++pass)
    {
        air_topk_stable::set_buf_pointers(
            in, in_idx, buf1, idx_buf1, buf2, idx_buf2, pass, in_buf, in_idx_buf, out_buf, out_idx_buf);

        if (fused_last_filter && pass == num_passes - 1)
        {
            kernel = air_topk_stable::radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, true>;
        }

        kernel<<<blocks, BlockSize, 0, stream>>>(in, in_idx, in_buf, in_idx_buf, out_buf, out_idx_buf, topk_out,
            topk_out_idx, counters, histograms, len, k, select_min, pass);
    }

    if (!fused_last_filter)
    {
        air_topk_stable::last_filter_kernel<T, IdxT, BitsPerPass, true><<<blocks, BlockSize, 0, stream>>>(
            in, in_idx, out_buf, out_idx_buf, topk_out, topk_out_idx, len, k, counters, select_min);
    }

    T* idx_sort_out = sorted ? sort_in : out;
    IdxT* idx_sort_out_idx = sorted ? sort_in_idx : out_idx;

    cub::DeviceSegmentedSort::SortPairs(sort_temp_storage, temp_storage_bytes, topk_out_idx, idx_sort_out_idx, topk_out,
        idx_sort_out, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
    if (sorted)
    {
        if (select_min)
        {
            cub::DeviceSegmentedSort::StableSortPairs(sort_temp_storage, temp_storage_bytes, sort_in, out, sort_in_idx,
                out_idx, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
        else
        {
            cub::DeviceSegmentedSort::StableSortPairsDescending(sort_temp_storage, temp_storage_bytes, sort_in, out,
                sort_in_idx, out_idx, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
    }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void standalone_stable_radix_topk_one_block_(void* buf, size_t& buf_size, T const* in, IdxT const* in_idx,
    int batch_size, IdxT len, IdxT k, T* out, IdxT* out_idx, bool select_min, cudaStream_t stream, bool sorted = false)
{
    static_assert(air_topk_stable::calc_num_passes<T, BitsPerPass>() > 1);

    char* bufs = nullptr;
    void* sort_temp_storage = nullptr;
    T* topk_out = nullptr;
    IdxT* topk_out_idx = nullptr;
    T* sort_in = nullptr;
    IdxT* sort_in_idx = nullptr;

    size_t temp_storage_bytes = 0;
    size_t temp_storage_bytes_sort = 0;
    const IdxT buf_len = air_topk_stable::calc_buf_len<T, IdxT, unsigned>(len);

    air_topk_stable::ComputeOffset<IdxT> computeoffset(k);
    cub::CountingInputIterator<IdxT> counting_iter(0);
    cub::TransformInputIterator<IdxT, air_topk_stable::ComputeOffset<IdxT>, cub::CountingInputIterator<IdxT>>
        transform_iter(counting_iter, computeoffset);

    cub::DeviceSegmentedSort::SortPairs(NULL, temp_storage_bytes, out_idx, out_idx, out, out, k * batch_size,
        batch_size, transform_iter, transform_iter + 1, stream);
    if (sorted)
    {
        if (select_min)
        {
            cub::DeviceSegmentedSort::StableSortPairs(NULL, temp_storage_bytes_sort, out, out, out_idx, out_idx,
                k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
        else
        {
            cub::DeviceSegmentedSort::StableSortPairsDescending(NULL, temp_storage_bytes_sort, out, out, out_idx,
                out_idx, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
    }

    temp_storage_bytes = max(temp_storage_bytes, temp_storage_bytes_sort);
    {
        size_t total_size = 0;
        size_t sort_buffer_size = 0;
        if (sorted)
        {
            sort_buffer_size = k * batch_size;
        }
        std::vector<size_t> sizes = {buf_len * 2 * (sizeof(T) + sizeof(IdxT)) * batch_size, temp_storage_bytes,
            sizeof(*topk_out) * k * batch_size, sizeof(*topk_out_idx) * k * batch_size,
            sizeof(*sort_in) * sort_buffer_size, sizeof(*sort_in_idx) * sort_buffer_size};
        total_size = calc_aligned_size(sizes);

        if (!buf)
        {
            buf_size = total_size;
            return;
        }

        std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
        bufs = static_cast<decltype(bufs)>(aligned_pointers[0]);
        sort_temp_storage = aligned_pointers[1];
        topk_out = static_cast<decltype(topk_out)>(aligned_pointers[2]);
        topk_out_idx = static_cast<decltype(topk_out_idx)>(aligned_pointers[3]);
        if (sorted)
        {
            sort_in = static_cast<decltype(sort_in)>(aligned_pointers[4]);
            sort_in_idx = static_cast<decltype(sort_in_idx)>(aligned_pointers[5]);
        }
    }

    air_topk_stable::radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize, true>
        <<<batch_size, BlockSize, 0, stream>>>(in, in_idx, len, k, topk_out, topk_out_idx, select_min, bufs);

    T* idx_sort_out = sorted ? sort_in : out;
    IdxT* idx_sort_out_idx = sorted ? sort_in_idx : out_idx;
    cub::DeviceSegmentedSort::SortPairs(sort_temp_storage, temp_storage_bytes, topk_out_idx, idx_sort_out_idx, topk_out,
        idx_sort_out, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
    if (sorted)
    {
        if (select_min)
        {
            cub::DeviceSegmentedSort::StableSortPairs(sort_temp_storage, temp_storage_bytes, sort_in, out, sort_in_idx,
                out_idx, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
        else
        {
            cub::DeviceSegmentedSort::StableSortPairsDescending(sort_temp_storage, temp_storage_bytes, sort_in, out,
                sort_in_idx, out_idx, k * batch_size, batch_size, transform_iter, transform_iter + 1, stream);
        }
    }
}

template <typename T, typename idxT, bool sorted = false>
void standalone_stable_radix_11bits(void* buf, size_t& buf_size, T const* in, int batch_size, idxT len, idxT k, T* out,
    idxT* out_idx, bool greater, cudaStream_t stream = 0)
{
    constexpr int items_per_thread = 32;
    constexpr int block_dim = 512;
    constexpr bool fused_last_filter = false;
    if (len <= block_dim * items_per_thread)
    {
        standalone_stable_radix_topk_one_block_<T, idxT, 11, block_dim>(
            buf, buf_size, in, static_cast<idxT*>(nullptr), batch_size, len, k, out, out_idx, !greater, stream, sorted);
    }
    else
    {
        int sm_cnt = suggestify::common::getMultiProcessorCount();
        unsigned grid_dim = air_topk_stable::calc_grid_dim<T, idxT, 11, block_dim>(batch_size, len, sm_cnt);

        if (grid_dim == 1)
        {
            standalone_stable_radix_topk_one_block_<T, idxT, 11, block_dim>(buf, buf_size, in,
                static_cast<idxT*>(nullptr), batch_size, len, k, out, out_idx, !greater, stream, sorted);
        }
        else
        {
            standalone_stable_radix_topk_<T, idxT, 11, block_dim>(buf, buf_size, in, static_cast<idxT*>(nullptr),
                batch_size, len, k, out, out_idx, !greater, fused_last_filter, grid_dim, stream, sorted);
        }
    }
}

#endif


template <typename T>
size_t invokeComputeTopkLastDimWorkspaceSize(
    SizeType32 batchSize, SizeType32 inputLength, SizeType32 k, bool is_largest)
{
    size_t buf_size = 0;
    void* workspace = nullptr;
    T const* in = nullptr;
    T* out_val = nullptr;
    SizeType32* out_idx = nullptr;
    standalone_stable_radix_11bits<T, SizeType32, true>(
        workspace, buf_size, in, batchSize, inputLength, k, out_val, out_idx, is_largest, 0);
    return buf_size;
}

#define INSTANTIATE_COMPUTE_TOPK_LastDim_WORKSPACE_SIZE_DATA_TYPE(T)                                                   \
    template size_t invokeComputeTopkLastDimWorkspaceSize<T>(                                                          \
        SizeType32 batchSize, SizeType32 inputLength, SizeType32 k, bool is_largest)

INSTANTIATE_COMPUTE_TOPK_LastDim_WORKSPACE_SIZE_DATA_TYPE(int);
INSTANTIATE_COMPUTE_TOPK_LastDim_WORKSPACE_SIZE_DATA_TYPE(float);
INSTANTIATE_COMPUTE_TOPK_LastDim_WORKSPACE_SIZE_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_COMPUTE_TOPK_LastDim_WORKSPACE_SIZE_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_COMPUTE_TOPK_LastDim_WORKSPACE_SIZE_DATA_TYPE



template <typename T>
void invokeTopkLastDim(SizeType32 batchSize, SizeType32 inputLength, SizeType32 k, bool is_largest,
    void const* __restrict__ input, void* __restrict__ out_val, void* __restrict__ out_idx, void* workspace,
    cudaStream_t stream)
{
    size_t buf_size = 0;
    T const* in = reinterpret_cast<T const*>(input);
    T* out_val_ = reinterpret_cast<T*>(out_val);
    SizeType32* out_idx_ = reinterpret_cast<SizeType32*>(out_idx);
    standalone_stable_radix_11bits<T, SizeType32, true>(
        workspace, buf_size, in, batchSize, inputLength, k, out_val_, out_idx_, is_largest, stream);
}

#define INSTANTIATE_TOPK_LastDim_DATA_TYPE(T)                                                                          \
    template void invokeTopkLastDim<T>(SizeType32 batchSize, SizeType32 inputLength, SizeType32 k, bool is_largest,    \
        void const* __restrict__ input, void* __restrict__ out_val, void* __restrict__ out_idx, void* workspace,       \
        cudaStream_t stream)

INSTANTIATE_TOPK_LastDim_DATA_TYPE(int);
INSTANTIATE_TOPK_LastDim_DATA_TYPE(float);
INSTANTIATE_TOPK_LastDim_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_TOPK_LastDim_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_TOPK_LastDim_DATA_TYPE

}
}
