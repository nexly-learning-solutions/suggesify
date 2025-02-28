

#include "../common/cudaUtils.h"
#include "../src/banRepeatNgram.h"

using namespace sugesstify::common;
using namespace sugesstify::runtime;

namespace sugesstify
{
namespace kernels
{

template <typename T>
__global__ void ban_repeat_ngram(T* logits, TokenIdType const** output_ids_buf, FinishedState const* finished_buf,
    SizeType32 const** parent_ids_buf, SizeType32 const* batch_slots, SizeType32 batch_size, SizeType32 beam_width,
    SizeType32 max_seq_len, SizeType32 const* no_repeat_ngram_size_buf, SizeType32 vocab_size_padded,
    SizeType32 const* sequence_lengths)
{

    auto const output_idx = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    auto const local_batch_idx = blockIdx.y / beam_width;
    auto const batch_slot = batch_slots != nullptr ? batch_slots[local_batch_idx] : local_batch_idx;
    auto const beam_idx = blockIdx.y % beam_width;
    bool const beam_search = beam_width > 1;
    auto const no_repeat_ngram_size = no_repeat_ngram_size_buf[batch_slot];
    auto const step = sequence_lengths[batch_slot];

    if (no_repeat_ngram_size == 0 || step < no_repeat_ngram_size)
    {
        return;
    }

    if ((finished_buf != nullptr) && (finished_buf[batch_slot * beam_width + beam_idx].isFinished()))
    {
        return;
    }

    extern __shared__ TokenIdType shared_tokens[];
    auto const shared_tokens_length = blockDim.x + no_repeat_ngram_size - 1;
    auto* last_tokens = &shared_tokens[shared_tokens_length];
    auto const last_tokens_length = no_repeat_ngram_size - 1;
    if (threadIdx.x == 0)
    {
        auto parent_id = beam_idx;
        auto const start_record_idx = min(output_idx + shared_tokens_length, static_cast<SizeType32>(step));
        auto shared_token_idx = start_record_idx == step ? step - output_idx - 1 : shared_tokens_length - 1;
        auto last_token_idx = last_tokens_length - 1;

        for (auto curr_idx = step - 1; curr_idx >= output_idx; curr_idx--)
        {
            if (last_token_idx >= 0)
            {
                last_tokens[last_token_idx--] = output_ids_buf[batch_slot][parent_id * max_seq_len + curr_idx];
            }

            if (curr_idx < start_record_idx)
            {
                shared_tokens[shared_token_idx--] = output_ids_buf[batch_slot][parent_id * max_seq_len + curr_idx];
            }

            if (beam_search)
            {
                parent_id = parent_ids_buf[batch_slot][parent_id * max_seq_len + curr_idx];
            }
        }
    }

    __syncthreads();

    if (output_idx > step - no_repeat_ngram_size)
    {
        return;
    }

    bool ban_ngram = true;

    for (SizeType32 ngram_idx = 0; ngram_idx < no_repeat_ngram_size - 1; ngram_idx++)
    {
        if (shared_tokens[threadIdx.x + ngram_idx] != last_tokens[ngram_idx])
        {
            ban_ngram = false;
            break;
        }
    }

    if (ban_ngram)
    {
        auto const banned_token
            = shared_tokens[threadIdx.x + no_repeat_ngram_size - 1];
        logits[local_batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token]
            = static_cast<T>(-INFINITY);
    }
}

template <typename T>
void invokeBanRepeatNgram(T* logits, TokenIdType const** output_ids_buf, FinishedState const* finished_buf,
    SizeType32 const** parent_ids_buf, SizeType32 const* batch_slot, SizeType32 const* sequence_lengths,
    SizeType32 batch_size, SizeType32 beam_width, SizeType32 max_seq_len, SizeType32 const* no_repeat_ngram_size_buf,
    SizeType32 vocab_size_padded, SizeType32 max_step, cudaStream_t stream)
{
    int max_no_repeat_ngram_size = 32;

    dim3 block, grid;
    constexpr SizeType32 max_blocks{256};
    block.x = min(((max_step + 32 - 1) / 32) * 32, max_blocks);
    grid.x = (max_step + block.x - 1) / block.x;
    grid.y = batch_size * beam_width;

    ban_repeat_ngram<<<grid, block, (block.x + 2 * (max_no_repeat_ngram_size - 1)) * sizeof(int), stream>>>(logits,
        output_ids_buf, finished_buf, parent_ids_buf, batch_slot, batch_size, beam_width, max_seq_len,
        no_repeat_ngram_size_buf, vocab_size_padded, sequence_lengths);
    sync_check_cuda_error();
}

#define INVOKE_BAN_REPEAT_NGRAM(T)                                                                                     \
    template void invokeBanRepeatNgram(T* logits, TokenIdType const** output_ids_buf,                                  \
        const FinishedState* finished_buf, SizeType32 const** parent_ids_buf, SizeType32 const* batch_slot,            \
        SizeType32 const* sequence_lengths, SizeType32 batch_size, SizeType32 beam_width, SizeType32 max_seq_len,      \
        SizeType32 const* no_repeat_ngram_size_buf, SizeType32 vocab_size_padded, SizeType32 max_step,                 \
        cudaStream_t stream);

INVOKE_BAN_REPEAT_NGRAM(float)
INVOKE_BAN_REPEAT_NGRAM(half)
#ifdef ENABLE_BF16
INVOKE_BAN_REPEAT_NGRAM(__nv_bfloat16)
#endif
#undef INVOKE_BAN_REPEAT_NGRAM

} // namespace kernels

} // namespace sugesstify
