
#pragma once

#include "../src/decodingCommon.h"
#include "../runtime/common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sugesstify
{
namespace kernels
{

template <typename T>
void invokeBanRepeatNgram(T* logits, runtime::TokenIdType const** output_ids_buf, FinishedState const* finished_buf,
    runtime::SizeType32 const** parent_ids_buf, runtime::SizeType32 const* batch_slot,
    runtime::SizeType32 const* sequence_lengths, runtime::SizeType32 batch_size, runtime::SizeType32 beam_width,
    runtime::SizeType32 max_seq_len, runtime::SizeType32 const* no_repeat_ngram_size_buf,
    runtime::SizeType32 vocab_size_padded, runtime::SizeType32 max_step, cudaStream_t stream);

}
}
