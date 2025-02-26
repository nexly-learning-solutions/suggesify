
#include "suggestify/common/cudaUtils.h"
#include "suggestify/kernels/beamSearchKernels.h"
#include "suggestify/kernels/decodingCommon.h"
#include "suggestify/kernels/decodingKernels.h"
#include "thUtils.h"

namespace th = torch;
namespace tl = suggestify;
namespace tk = suggestify::kernels;

namespace torch_ext
{

th::Tensor gatherTree(
    th::Tensor& sequence_lengths,
    th::Tensor& output_ids,
    th::Tensor& parent_ids,
    th::Tensor& end_ids,
    th::Tensor& tiled_input_lengths,
    th::optional<th::Tensor> cum_log_probs_opt,
    th::optional<th::Tensor> log_probs_opt,
    th::optional<th::Tensor> log_probs_tiled_opt,
    th::optional<th::Tensor> beam_hyps_output_ids_cba,
    th::optional<th::Tensor> beam_hyps_seq_len_cba,
    th::optional<th::Tensor> beam_hyps_cum_log_probs_cba,
    th::optional<th::Tensor> beam_hyps_normed_scores_cba,
    th::optional<th::Tensor> beam_hyps_log_probs_cba,
    th::optional<th::Tensor> beam_hyps_min_normed_scores,
    th::optional<th::Tensor> beam_hyps_num_beams,
    th::optional<th::Tensor> beam_hyps_is_done,
    th::optional<th::Tensor> finished,
    th::Tensor& length_penalty,
    int64_t const batch_size,
    int64_t const beam_width,
    int64_t const max_seq_len,
    bool const use_beam_hyps
)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    th::Tensor final_output_ids = torch::zeros(
        {batch_size, beam_width, max_seq_len}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    if (use_beam_hyps && beam_width > 1)
    {
        int32_t* final_output_ids_ptr = get_ptr<int32_t>(final_output_ids);
        tk::invokeInitializeOutput(
            final_output_ids_ptr, get_ptr<int32_t>(end_ids), batch_size, beam_width, max_seq_len, stream);

        tk::BeamHypotheses bh;
        bh.nBatchSize = batch_size;
        bh.nBeamWidth = beam_width;
        bh.nMaxSeqLen = max_seq_len;
        bh.lengthPenalties = get_ptr<float>(length_penalty);
        bh.inputLengths = get_ptr<int32_t>(tiled_input_lengths);
        bh.outputIds = final_output_ids_ptr;
        bh.logProbs = log_probs_opt.has_value() ? get_ptr<float>(log_probs_opt.value()) : nullptr;
        bh.logProbsTiled = log_probs_tiled_opt.has_value() ? get_ptr<float>(log_probs_tiled_opt.value()) : nullptr;
        bh.sequenceLengths = get_ptr<int32_t>(sequence_lengths);
        bh.cumLogProbs = cum_log_probs_opt.has_value() ? get_ptr<float>(cum_log_probs_opt.value()) : nullptr;
        bh.outputIdsCBA = get_ptr<int32_t>(beam_hyps_output_ids_cba.value());
        bh.logProbsCBA = get_ptr<float>(beam_hyps_log_probs_cba.value());
        bh.sequenceLengthsCBA = get_ptr<int32_t>(beam_hyps_seq_len_cba.value());
        bh.cumLogProbsCBA = get_ptr<float>(beam_hyps_cum_log_probs_cba.value());
        bh.normedScoresCBA = get_ptr<float>(beam_hyps_normed_scores_cba.value());
        bh.numBeamsCBA = get_ptr<int32_t>(beam_hyps_num_beams.value());
        bh.minNormedScoresCBA = get_ptr<float>(beam_hyps_min_normed_scores.value());
        bh.batchDones = get_ptr<bool>(beam_hyps_is_done.value());
        bh.finished
            = reinterpret_cast<tk::FinishedState*>(get_ptr<tk::FinishedState::UnderlyingType>(finished.value()));
        bh.outputIdsUnfinish = get_ptr<int32_t>(output_ids);
        bh.parentIdsUnfinish = get_ptr<int32_t>(parent_ids);

        tk::invokeInsertUnfinishedPath(bh, stream);
        sync_check_cuda_error();

        tk::invokeFinalize(bh, stream);
        sync_check_cuda_error();
    }
    else if (!use_beam_hyps && beam_width > 1)
    {
        th::Tensor workspace = torch::zeros(batch_size * beam_width * max_seq_len * sizeof(int32_t),
            torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        tk::gatherTreeParam param;
        param.beams = get_ptr<int32_t>(workspace);
        param.sequenceLengths = get_ptr<int32_t>(sequence_lengths);
        param.maxSequenceLengthFinalStep = 1;
        param.responseInputLengths = nullptr;
        param.maxSeqLen = max_seq_len;
        param.batchSize = batch_size;
        param.beamWidth = beam_width;
        param.stepIds = get_ptr<int32_t>(output_ids);
        param.parentIds = beam_width == 1 ? nullptr : get_ptr<int32_t>(parent_ids);
        param.endTokens = get_ptr<int32_t>(end_ids);
        param.inputLengths = get_ptr<int32_t>(tiled_input_lengths);

        param.stream = stream;
        param.outputIds = get_ptr<int32_t>(final_output_ids);
        param.cumLogProbs = cum_log_probs_opt.has_value() ? get_ptr<float>(cum_log_probs_opt.value()) : nullptr;
        param.lengthPenalty = get_val<float>(length_penalty, 0);

        tk::invokeGatherTree(param);
        sync_check_cuda_error();
    }
    else
    {
        cudaMemcpyAsync(get_ptr<int32_t>(final_output_ids), get_ptr<int32_t>(output_ids),
            sizeof(int) * batch_size * beam_width * max_seq_len, cudaMemcpyDeviceToDevice, stream);
        sync_check_cuda_error();
    }
    return final_output_ids;
}

}

static auto gather_tree = torch::RegisterOperators("suggestify::gather_tree", &torch_ext::gatherTree);
