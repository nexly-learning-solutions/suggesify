
#pragma once

#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace suggestify
{
namespace kernels
{

template <typename T>
void invokeBanBadWords(T* logits, runtime::TokenIdType const** output_ids_ptr,
    runtime::SizeType32 const** parent_ids_ptr, runtime::SizeType32 const* batch_slot, runtime::SizeType32 batch_size,
    runtime::SizeType32 beam_width, runtime::TokenIdType const* const* bad_words,
    runtime::SizeType32 const* bad_words_len, runtime::SizeType32 max_bad_words_len,
    runtime::SizeType32 vocab_size_padded, runtime::SizeType32 const* sequence_lengths, runtime::SizeType32 max_seq_len,
    cudaStream_t stream);

}
}
