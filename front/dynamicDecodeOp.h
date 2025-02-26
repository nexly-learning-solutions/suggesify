
#include "suggestify/common/cudaBf16Wrapper.h"
#include "suggestify/layers/dynamicDecodeLayer.h"
#include "suggestify/runtime/iTensor.h"
#include "thUtils.h"

namespace th = torch;

namespace torch_ext
{

class IFtDynamicDecode
{
public:
    virtual void setup(size_t const batch_size, size_t const beam_width, th::optional<th::Tensor> runtime_top_k_opt,
        th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
        th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
        th::optional<th::Tensor> frequency_penalty_opt, th::optional<th::Tensor> min_length_opt,
        th::optional<th::Tensor> length_penalty_opt, th::optional<th::Tensor> early_stopping_opt,
        th::optional<th::Tensor> beam_search_diversity_rate_opt, th::optional<th::Tensor> random_seed_opt,
        th::optional<th::Tensor> top_p_decay_opt, th::optional<th::Tensor> top_p_min_opt,
        th::optional<th::Tensor> top_p_reset_ids_opt, th::optional<th::Tensor> no_repeat_ngram_size_opt,
        bool output_log_probs, bool cum_log_probs)
        = 0;

    virtual void forward(th::Tensor const& logits, int const step, int const max_input_length,
        int const max_attention_window, int const sink_token_length, uint64_t const ite, int const local_batch_size,
        th::Tensor end_id, th::optional<th::Tensor> embedding_bias_opt, th::optional<th::Tensor> input_lengths_opt,
        th::optional<th::Tensor> sequence_limit_length_opt, th::optional<th::Tensor> stop_words_list_ptrs_opt,
        th::optional<th::Tensor> stop_words_lens_opt, int32_t const max_stop_words_len,
        th::optional<th::Tensor> bad_words_list_ptrs_opt, th::optional<th::Tensor> bad_words_lens_opt,
        int32_t const max_bad_words_len, th::optional<th::Tensor> src_cache_indirection_opt,
        th::Tensor& output_token_ids, th::Tensor& newTokens, th::Tensor& should_stop,
        th::optional<th::Tensor> finished_input, th::optional<th::Tensor> finished_output,
        th::optional<th::Tensor> sequence_lengths_opt, th::optional<th::Tensor> cum_log_probs_opt,
        th::optional<th::Tensor> output_log_probs_opt, th::optional<th::Tensor> output_log_probs_tiled_opt,
        th::optional<th::Tensor> parent_ids_opt, th::optional<th::Tensor> tgt_cache_indirection_opt,
        th::optional<th::Tensor> beam_hyps_output_ids_cba_opt, th::optional<th::Tensor> beam_hyps_seq_len_cba_opt,
        th::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt,
        th::optional<th::Tensor> beam_hyps_normed_scores_cba_opt, th::optional<th::Tensor> beam_hyps_log_probs_cba_opt,
        th::optional<th::Tensor> beam_hyps_min_normed_scores_opt, th::optional<th::Tensor> beam_hyps_num_beams_opt,
        th::optional<th::Tensor> beam_hyps_is_done_opt, bool const use_beam_hyps)
        = 0;
};

template <typename T>
class FtDynamicDecode : public IFtDynamicDecode
{
public:
    FtDynamicDecode(size_t const max_batch_size, size_t const max_beam_width, size_t const vocab_size,
        size_t const vocab_size_padded, int const tensor_para_size, int const pipeline_para_size);

    void setup(size_t const batch_size, size_t const beam_width, th::optional<th::Tensor> runtime_top_k_opt,
        th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
        th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
        th::optional<th::Tensor> frequency_penalty_opt, th::optional<th::Tensor> min_length_opt,
        th::optional<th::Tensor> length_penalty_opt, th::optional<th::Tensor> early_stopping_opt,
        th::optional<th::Tensor> beam_search_diversity_rate_opt, th::optional<th::Tensor> random_seed_opt,
        th::optional<th::Tensor> top_p_decay_opt, th::optional<th::Tensor> top_p_min_opt,
        th::optional<th::Tensor> top_p_reset_ids_opt, th::optional<th::Tensor> no_repeat_ngram_size_opt,
        bool output_log_probs, bool cum_log_probs) override;

    void forward(th::Tensor const& logits, int const step, int const max_input_length, int const max_attention_window,
        int const sink_token_length, uint64_t const ite, int const local_batch_size, th::Tensor end_id,
        th::optional<th::Tensor> embedding_bias_opt, th::optional<th::Tensor> input_lengths_opt,
        th::optional<th::Tensor> sequence_limit_length_opt, th::optional<th::Tensor> stop_words_list_ptrs_opt,
        th::optional<th::Tensor> stop_words_lens_opt, int32_t const max_stop_words_len,
        th::optional<th::Tensor> bad_words_list_ptrs_opt, th::optional<th::Tensor> bad_words_lens_opt,
        int32_t const max_bad_words_len, th::optional<th::Tensor> src_cache_indirection_opt,
        th::Tensor& output_token_ids, th::Tensor& newTokens, th::Tensor& should_stop,
        th::optional<th::Tensor> finished_input, th::optional<th::Tensor> finished_output,
        th::optional<th::Tensor> sequence_lengths_opt, th::optional<th::Tensor> cum_log_probs_opt,
        th::optional<th::Tensor> output_log_probs_opt, th::optional<th::Tensor> output_log_probs_tiled_opt,
        th::optional<th::Tensor> parent_ids_opt, th::optional<th::Tensor> tgt_cache_indirection_opt,
        th::optional<th::Tensor> beam_hyps_output_ids_cba_opt, th::optional<th::Tensor> beam_hyps_seq_len_cba_opt,
        th::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt,
        th::optional<th::Tensor> beam_hyps_normed_scores_cba_opt, th::optional<th::Tensor> beam_hyps_log_probs_cba_opt,
        th::optional<th::Tensor> beam_hyps_min_normed_scores_opt, th::optional<th::Tensor> beam_hyps_num_beams_opt,
        th::optional<th::Tensor> beam_hyps_is_done_opt, bool const use_beam_hyps) override;

private:
    suggestify::runtime::ITensor::SharedPtr mFinishedSum;
    std::shared_ptr<suggestify::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
    std::shared_ptr<suggestify::runtime::DecodingLayerWorkspace> mDecodingWorkspace;
    std::optional<size_t> mBeamWidth;
    suggestify::runtime::ITensor::SharedConstPtr mBatchSlots;
};

class DynamicDecodeOp : public th::jit::CustomClassHolder
{
public:
    DynamicDecodeOp(int64_t const max_batch_size, int64_t const max_beam_width, int64_t const vocab_size,
        int64_t const vocab_size_padded, int64_t const tensor_para_size, int64_t const pipeline_para_size,
        at::ScalarType const scalar_type);

    void setup(int64_t const batch_size, int64_t const beam_width, th::optional<th::Tensor> runtime_top_k_opt,
        th::optional<th::Tensor> runtime_top_p_opt, th::optional<th::Tensor> temperature_opt,
        th::optional<th::Tensor> repetition_penalty_opt, th::optional<th::Tensor> presence_penalty_opt,
        th::optional<th::Tensor> frequency_penalty_opt, th::optional<th::Tensor> min_length_opt,
        th::optional<th::Tensor> length_penalty_opt, th::optional<th::Tensor> early_stopping_opt,
        th::optional<th::Tensor> beam_search_diversity_rate_opt, th::optional<th::Tensor> random_seed_opt,
        th::optional<th::Tensor> top_p_decay_opt, th::optional<th::Tensor> top_p_min_opt,
        th::optional<th::Tensor> top_p_reset_ids_opt, th::optional<th::Tensor> no_repeat_ngram_size_opt,
        bool output_log_probs, bool cum_log_probs);

    th::Tensor forward(th::Tensor const& logits, int64_t const step, int64_t const max_input_length,
        int64_t const max_attention_window, int64_t const sink_token_length, int64_t const ite,
        int64_t const local_batch_size, th::Tensor end_id, th::optional<th::Tensor> embedding_bias_opt,
        th::optional<th::Tensor> input_lengths_opt, th::optional<th::Tensor> sequence_limit_length_opt,
        th::optional<th::Tensor> stop_words_list_ptrs_opt, th::optional<th::Tensor> stop_words_lens_opt,
        int64_t const max_stop_words_len, th::optional<th::Tensor> bad_words_list_ptrs_opt,
        th::optional<th::Tensor> bad_words_lens_opt, int64_t const max_bad_words_len,
        th::optional<th::Tensor> src_cache_indirection_opt, th::Tensor output_token_ids, th::Tensor newTokens,
        th::optional<th::Tensor> finished_input, th::optional<th::Tensor> finished_output,
        th::optional<th::Tensor> sequence_lengths_opt, th::optional<th::Tensor> cum_log_probs_opt,
        th::optional<th::Tensor> output_log_probs_opt, th::optional<th::Tensor> output_log_probs_tiled_opt,
        th::optional<th::Tensor> parent_ids_opt, th::optional<th::Tensor> tgt_cache_indirection_opt,
        th::optional<th::Tensor> beam_hyps_output_ids_cba_opt, th::optional<th::Tensor> beam_hyps_seq_len_cba_opt,
        th::optional<th::Tensor> beam_hyps_cum_log_probs_cba_opt,
        th::optional<th::Tensor> beam_hyps_normed_scores_cba_opt, th::optional<th::Tensor> beam_hyps_log_probs_cba_opt,
        th::optional<th::Tensor> beam_hyps_min_normed_scores_opt, th::optional<th::Tensor> beam_hyps_num_beams_opt,
        th::optional<th::Tensor> beam_hyps_is_done_opt, bool const use_beam_hyps);

private:
    size_t const maxBatchSize_;
    size_t const maxBeamWidth_;
    size_t const vocabSize_;
    size_t const vocabSizePadded_;
    int const tensorParaSize_;
    int const pipelineParaSize_;
    at::ScalarType const scalarType_;
    std::unique_ptr<IFtDynamicDecode> dynamicDecode_;

    void createInstance();
};

}
