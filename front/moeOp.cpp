
#include "suggestify/common/workspace.h"
#include "../src/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "../src/mixtureOfExperts/moe_kernels.h"
#include "../torchUtils.h"
#include "thUtils.h"

#include <ATen/native/cuda/Resize.h>

#include <functional>

namespace torch_ext
{

namespace common = suggestify::common;
namespace kernels = suggestify::kernels;
using profiler_backend = kernels::GemmProfilerBackend;

struct GemmIDMoe
{
    profiler_backend::GemmToProfile gemm_idx;
    int64_t hidden_size;
    int64_t inter_size;
    int num_experts;
    int top_k;

    bool operator==(GemmIDMoe const& id) const
    {
        return id.gemm_idx == gemm_idx && id.hidden_size == hidden_size && id.inter_size == inter_size
            && id.num_experts == num_experts && id.top_k == top_k;
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIDMoe const& id)
    {
        out << "gemm_idx, hidden_size, inter_size, num_experts, top_k=" << static_cast<int>(id.gemm_idx) << ","
            << id.hidden_size << "," << id.inter_size << "," << id.num_experts << "," << id.top_k;
        return out;
    }
};

struct GemmIDMoeHash
{
    std::size_t operator()(GemmIDMoe const& id) const
    {
        size_t hash = std::hash<int>{}(static_cast<int>(id.gemm_idx));
        hash ^= std::hash<int64_t>{}(id.hidden_size);
        hash ^= std::hash<int64_t>{}(id.inter_size);
        hash ^= std::hash<int>{}(id.num_experts);
        hash ^= std::hash<int>{}(id.top_k);
        return hash;
    }
};

using ProfileId = int;
using MProfileMap = std::unordered_map<int, ProfileId>;
using MProfileMapPtr = std::shared_ptr<MProfileMap>;

struct MNKProfileMap
{
    std::unordered_map<GemmIDMoe, MProfileMapPtr, GemmIDMoeHash> profile_map;

    bool existsMProfileMap(GemmIDMoe const& id)
    {
        auto const iter = profile_map.find(id);
        return iter != profile_map.end();
    }

    void createMProfileMap(GemmIDMoe const& id)
    {
        profile_map[id] = std::make_shared<MProfileMap>();
    }

    MProfileMapPtr getMProfileMap(GemmIDMoe const& id)
    {
        auto const iter = profile_map.find(id);
        if (iter == profile_map.end())
        {
            std::ostringstream msg;
            msg << "Cannot find ID (" << id << ") in the profile map. Abort.";
            C10_THROW_ERROR(Error, msg.str());
        }
        return iter->second;
    }
};

struct RunnerTypeKey
{
    c10::ScalarType activation_dtype;
    c10::ScalarType weight_dtype;

    bool operator==(RunnerTypeKey const& key) const
    {
        return key.activation_dtype == activation_dtype && key.weight_dtype == weight_dtype;
    }
};

struct RunnerTypeKeyHash
{
    std::size_t operator()(RunnerTypeKey const& key) const
    {
        size_t hash = std::hash<int>{}(static_cast<int>(key.activation_dtype));
        hash ^= std::hash<int>{}(static_cast<int>(key.weight_dtype));
        return hash;
    }
};

class FusedMoeRunner : public torch::CustomClassHolder
{
public:
    static c10::intrusive_ptr<FusedMoeRunner> getInstance(
        c10::ScalarType activation_dtype, c10::ScalarType weight_dtype)
    {
        static std::mutex instance_map_mutex;
        std::lock_guard<std::mutex> lock(instance_map_mutex);

        static std::unordered_map<RunnerTypeKey, c10::intrusive_ptr<FusedMoeRunner>, RunnerTypeKeyHash> instance_map;

        auto const key = RunnerTypeKey{activation_dtype, weight_dtype};
        auto const iter = instance_map.find(key);
        if (iter == instance_map.end())
        {
            auto instance = c10::make_intrusive<FusedMoeRunner>(activation_dtype, weight_dtype);
            instance_map[key] = instance;
            return instance;
        }
        return iter->second;
    }

    FusedMoeRunner(c10::ScalarType activation_dtype, c10::ScalarType weight_dtype)
    {
        mActivationDtype = activation_dtype;
        mWeightDtype = weight_dtype;

        if (mActivationDtype == c10::ScalarType::Half && mWeightDtype == c10::ScalarType::Half)
        {
            mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, half>>();
        }
#ifdef ENABLE_BF16
        else if (mActivationDtype == c10::ScalarType::BFloat16 && mWeightDtype == c10::ScalarType::BFloat16)
        {
            mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
        }
#endif
        else
        {
            std::ostringstream msg;
            msg << "Unsupported activation_dtype " << c10::toString(mActivationDtype) << " and weight_dtype "
                << c10::toString(mWeightDtype) << ".";
            C10_THROW_ERROR(NotImplementedError, msg.str());
        }

        mProfiler = std::make_shared<kernels::GemmProfilerBackend>();
        mMNKProfileMap = std::make_shared<MNKProfileMap>();
        mAllProfiles = mKernelRunner->getTactics();
        mMinDimM = -1;
        mMaxDimM = -1;
    }

    ~FusedMoeRunner() = default;
    FusedMoeRunner(FusedMoeRunner const&) = delete;
    void operator=(FusedMoeRunner const&) = delete;

    void runProfile(torch::Tensor const& fc2_expert_weights, int64_t const top_k, int64_t const tp_size,
        int64_t const tp_rank, std::vector<int64_t> num_token_buckets)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        CHECK_INPUT(fc2_expert_weights, mWeightDtype)
        TORCH_CHECK(fc2_expert_weights.dim() == 3, "fc2_expert_weights must be 3D.");

        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t inter_size = fc2_expert_weights.sizes()[2];
        int num_experts = static_cast<int>(fc2_expert_weights.sizes()[0]);

        std::sort(num_token_buckets.begin(), num_token_buckets.end());
        mMinDimM = num_token_buckets.front();
        mMaxDimM = num_token_buckets.back();

        cudaStream_t stream;
        common::check_cuda_error(cudaStreamCreate(&stream));

        profiler_backend::GemmToProfile gemm_idxes[]
            = {profiler_backend::GemmToProfile::GEMM_1, profiler_backend::GemmToProfile::GEMM_2};

        for (auto const& gemm_idx : gemm_idxes)
        {
            runProfileGemmIdx(hidden_size, inter_size, num_experts, static_cast<int>(top_k), static_cast<int>(tp_size),
                static_cast<int>(tp_rank), num_token_buckets, gemm_idx, stream);
        }

        common::check_cuda_error(cudaStreamDestroy(stream));
    }

    c10::optional<std::vector<int64_t>> getProfileIds(
        int64_t const num_tokens, torch::Tensor const& fc2_expert_weights, int64_t const top_k)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        CHECK_INPUT(fc2_expert_weights, mWeightDtype)
        TORCH_CHECK(fc2_expert_weights.dim() == 3, "fc2_expert_weights must be 3D.");

        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t inter_size = fc2_expert_weights.sizes()[2];
        int num_experts = static_cast<int>(fc2_expert_weights.sizes()[0]);
        auto gemm_id_moe1 = GemmIDMoe{
            profiler_backend::GemmToProfile::GEMM_1, hidden_size, inter_size, num_experts, static_cast<int>(top_k)};
        auto gemm_id_moe2 = GemmIDMoe{
            profiler_backend::GemmToProfile::GEMM_2, hidden_size, inter_size, num_experts, static_cast<int>(top_k)};

        if (!mMNKProfileMap->existsMProfileMap(gemm_id_moe1) || !mMNKProfileMap->existsMProfileMap(gemm_id_moe2))
        {
            return c10::nullopt;
        }

        int64_t capped_num_tokens = num_tokens;
        if (num_tokens < mMinDimM)
        {
            capped_num_tokens = mMinDimM;
        }
        else if (num_tokens > mMaxDimM)
        {
            capped_num_tokens = mMaxDimM;
        }

        int gemm1_profile_id = mMNKProfileMap->getMProfileMap(gemm_id_moe1)->at(capped_num_tokens);
        int gemm2_profile_id = mMNKProfileMap->getMProfileMap(gemm_id_moe2)->at(capped_num_tokens);
        std::vector<int64_t> profile_ids = {gemm1_profile_id, gemm2_profile_id};
        return profile_ids;
    }

    torch::Tensor runMoe(torch::Tensor const& input, torch::Tensor const& gating_output,
        torch::Tensor const& fc1_expert_weights, torch::Tensor const& fc2_expert_weights, int64_t const top_k,
        torch::Tensor& workspace, int64_t const tp_size, int64_t const tp_rank,
        torch::optional<c10::ArrayRef<int64_t>> profile_ids)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        CHECK_INPUT(input, mActivationDtype)
        CHECK_INPUT(gating_output, at::ScalarType::Float)
        CHECK_INPUT(fc1_expert_weights, mWeightDtype)
        CHECK_INPUT(fc2_expert_weights, mActivationDtype)
        CHECK_INPUT(workspace, at::ScalarType::Char)

        TORCH_CHECK(input.dim() == 2, "input must be 2D.");
        TORCH_CHECK(gating_output.dim() == 2, "gating_output must be 2D.");
        TORCH_CHECK(fc1_expert_weights.dim() == 3, "fc1_expert_weights must be 3D.");
        TORCH_CHECK(fc2_expert_weights.dim() == 3, "fc2_expert_weights must be 3D.");
        TORCH_CHECK(
            input.sizes()[0] == gating_output.sizes()[0], "input and gating_output must have the same batch size.");
        TORCH_CHECK(input.sizes()[1] == fc1_expert_weights.sizes()[2],
            "input and fc1_expert_weights must have the same hidden size.");
        TORCH_CHECK(input.sizes()[1] == fc2_expert_weights.sizes()[1],
            "input and fc2_expert_weights must have the same hidden size.");
        TORCH_CHECK(gating_output.sizes()[1] == fc1_expert_weights.sizes()[0],
            "gating_output and fc1_expert_weights must have the same number of experts.");
        TORCH_CHECK(fc1_expert_weights.sizes()[0] == fc2_expert_weights.sizes()[0],
            "fc1_expert_weights and fc2_expert_weights must have the same number of experts.");
        TORCH_CHECK(fc1_expert_weights.sizes()[1] == fc2_expert_weights.sizes()[2] * 2,
            "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");

        int64_t num_rows = input.sizes()[0];
        int64_t hidden_size = fc2_expert_weights.sizes()[1];
        int64_t inter_size = fc2_expert_weights.sizes()[2];
        int const num_experts = static_cast<int>(fc2_expert_weights.sizes()[0]);
        int const moe_top_k = static_cast<int>(top_k);
        auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, 1, 0);
        auto activation_type = suggestify::ActivationType::Swiglu;
        auto norm_mode = kernels::MOEExpertScaleNormalizationMode::RENORMALIZE;

        setRunnerProfiles(profile_ids);

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

        std::vector<int64_t> output_shape = {num_rows, hidden_size};
        auto output = torch::empty(output_shape, input.options());

        WorkspaceInfo workspace_info = getWorkspaceInfo(workspace, num_rows, hidden_size, inter_size, num_experts,
            static_cast<int>(top_k), activation_type, norm_mode, parallelism_config);

        kernels::QuantParams quant_params{};
        kernels::LoraParams lora_params{};

        mKernelRunner->runMoe(input.const_data_ptr(), gating_output.const_data_ptr<float>(),
            fc1_expert_weights.const_data_ptr(), nullptr, activation_type, fc2_expert_weights.const_data_ptr(), nullptr,
            quant_params, num_rows, hidden_size, inter_size, num_experts, static_cast<int>(top_k),
            static_cast<char*>(workspace_info.workspace), output.data_ptr(), nullptr, output.sizes()[0],
            workspace_info.scale_probs, static_cast<int*>(workspace_info.src_to_dest_map),
            static_cast<int*>(workspace_info.selected_experts), 0, parallelism_config, norm_mode, false, lora_params,
            stream);

        return output;
    }

private:
    struct WorkspaceInfo
    {
        void* workspace{};
        void* scale_probs{};
        void* src_to_dest_map{};
        void* selected_experts{};
    };

    std::mutex mMutex;
    std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> mKernelRunner;
    std::shared_ptr<kernels::GemmProfilerBackend> mProfiler;
    std::shared_ptr<MNKProfileMap> mMNKProfileMap;
    int64_t mMinDimM;
    int64_t mMaxDimM;
    c10::ScalarType mActivationDtype;
    c10::ScalarType mWeightDtype;

    using Profile = suggestify::cutlass_extensions::CutlassGemmConfig;
    std::vector<Profile> mAllProfiles;

    void runProfileGemmIdx(int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const top_k,
        int const tp_size, int const tp_rank, std::vector<int64_t> const& num_token_buckets,
        profiler_backend::GemmToProfile const gemm_idx, cudaStream_t stream)
    {
        auto gemm_id_moe = GemmIDMoe{gemm_idx, hidden_size, inter_size, num_experts, top_k};

        if (mMNKProfileMap->existsMProfileMap(gemm_id_moe))
        {
            return;
        }

        mMNKProfileMap->createMProfileMap(gemm_id_moe);

        mProfiler->mGemmToProfile = gemm_idx;
        auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, 1, 0);
        mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
            suggestify::runtime::TorchUtils::dataType(mActivationDtype),
            suggestify::runtime::TorchUtils::dataType(mWeightDtype),
            suggestify::runtime::TorchUtils::dataType(mActivationDtype), num_experts, top_k, hidden_size, inter_size,
            suggestify::ActivationType::Swiglu,
 false, false, parallelism_config);

        char* profile_workspace = nullptr;
        size_t tmp_workspace_size = mProfiler->getWorkspaceSize(mMaxDimM);
        auto const cu_malloc_status = cudaMalloc(&profile_workspace, tmp_workspace_size);
        TORCH_CHECK(cu_malloc_status == cudaSuccess, "Can't allocate tmp workspace for MOE GEMM tactics profiling.");

        for (auto const& m : num_token_buckets)
        {
            ProfileId best_profile_id = runProfileM(m, profile_workspace, stream);
            mMNKProfileMap->getMProfileMap(gemm_id_moe)->insert({m, best_profile_id});
        }

        auto const cu_free = cudaFree(profile_workspace);
        TORCH_CHECK(cu_free == cudaSuccess, "Can't free tmp workspace for MOE GEMM profiling.");
    }

    ProfileId runProfileM(int64_t const m, char* profile_workspace, cudaStream_t stream)
    {
        mProfiler->prepare(m, profile_workspace, stream);
        float best_time = std::numeric_limits<float>::max();
        ProfileId best_profile_id;
        for (int i = 0; i < static_cast<int>(mAllProfiles.size()); ++i)
        {
            auto const& profile = mAllProfiles[i];
            float candidate_time = std::numeric_limits<float>::max();
            try
            {
                candidate_time = runSingleProfile(m, profile, profile_workspace, stream);
            }
            catch (std::exception const& e)
            {
                std::ostringstream msg;
                msg << "Cannot profile configuration " << i << ": " << profile.toString() << "\n (for"
                    << " m=" << m << ")"
                    << ", reason: \"" << e.what() << "\". Skipped";
                cudaGetLastError();

                std::cout << "Error: " << msg.str() << std::endl;
                continue;
            }

            if (candidate_time < best_time)
            {
                best_time = candidate_time;
                best_profile_id = i;
            }
        }
        return best_profile_id;
    }

    float runSingleProfile(int64_t const m, Profile const& profile, char* profile_workspace, cudaStream_t stream)
    {
        constexpr int warmup = 3;
        constexpr int runs = 5;

        for (int i = 0; i < warmup; ++i)
        {
            mProfiler->runProfiler(m, profile, profile_workspace, stream);
        }

        cudaEvent_t start;
        cudaEvent_t stop;
        common::check_cuda_error(cudaEventCreate(&start));
        common::check_cuda_error(cudaEventCreate(&stop));
        common::check_cuda_error(cudaStreamSynchronize(stream));
        common::check_cuda_error(cudaEventRecord(start, stream));

        for (int i = 0; i < runs; ++i)
        {
            mProfiler->runProfiler(m, profile, profile_workspace, stream);
        }

        common::check_cuda_error(cudaEventRecord(stop, stream));
        common::check_cuda_error(cudaEventSynchronize(stop));
        float elapsed;
        common::check_cuda_error(cudaEventElapsedTime(&elapsed, start, stop));
        common::check_cuda_error(cudaEventDestroy(start));
        common::check_cuda_error(cudaEventDestroy(stop));
        return elapsed / runs;
    }

    void setRunnerProfiles(torch::optional<c10::ArrayRef<int64_t>> profile_ids)
    {
        auto best_gemm1_profile = mAllProfiles.front();
        auto best_gemm2_profile = mAllProfiles.front();
        if (profile_ids.has_value())
        {
            TORCH_CHECK(profile_ids.value().size() == 2, "Expecting 2 profile ids");
            best_gemm1_profile = mAllProfiles.at(profile_ids.value()[0]);
            best_gemm2_profile = mAllProfiles.at(profile_ids.value()[1]);
        }
        mKernelRunner->setTactic(best_gemm1_profile, best_gemm2_profile);
    }

    WorkspaceInfo getWorkspaceInfo(torch::Tensor& workspace, int64_t const num_rows, int64_t const hidden_size,
        int64_t const inter_size, int num_experts, int top_k, suggestify::ActivationType activation_type,
        kernels::MOEExpertScaleNormalizationMode norm_mode, kernels::MOEParallelismConfig const& parallelismConfig)
    {
        size_t moe_workspace_size = mKernelRunner->getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts,
            top_k, activation_type, norm_mode, parallelismConfig, false);
        size_t scale_prob_size = num_rows * num_experts * sizeof(float);
        size_t src_to_dest_map_size = top_k * num_rows * sizeof(int);
        size_t selected_expert_size = top_k * num_rows * sizeof(int);
        std::vector<size_t> workspaces{moe_workspace_size, scale_prob_size, src_to_dest_map_size, selected_expert_size};

        size_t total_workspace_size = common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
        at::native::resize_impl_cuda_(
            workspace.unsafeGetTensorImpl(), {static_cast<int64_t>(total_workspace_size)}, std::nullopt);

        WorkspaceInfo info{};
        info.workspace = workspace.data_ptr();
        info.scale_probs = common::nextWorkspacePtr(static_cast<int8_t*>(workspace.data_ptr()), moe_workspace_size);
        info.src_to_dest_map = common::nextWorkspacePtr(static_cast<int8_t*>(info.scale_probs), scale_prob_size);
        info.selected_experts
            = common::nextWorkspacePtr(static_cast<int8_t*>(info.src_to_dest_map), src_to_dest_map_size);

        return info;
    }
};

torch::Tensor fused_moe(torch::Tensor const& input, torch::Tensor const& gating_output,
    torch::Tensor const& fc1_expert_weights, torch::Tensor const& fc2_expert_weights, int64_t const top_k,
    torch::Tensor& workspace, int64_t const tp_size, int64_t const tp_rank,
    torch::optional<c10::ArrayRef<int64_t>> profile_ids)
{
    return FusedMoeRunner::getInstance(input.scalar_type(), fc1_expert_weights.scalar_type())
        ->runMoe(input, gating_output, fc1_expert_weights, fc2_expert_weights, top_k, workspace, tp_size, tp_rank,
            profile_ids);
}

}

TORCH_LIBRARY(trtllm, m)
{
    m.class_<torch_ext::FusedMoeRunner>("FusedMoeProfiler")
        .def_static("get_instance", &torch_ext::FusedMoeRunner::getInstance)
        .def("run_profile", &torch_ext::FusedMoeRunner::runProfile)
        .def("get_profile_ids", &torch_ext::FusedMoeRunner::getProfileIds);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_moe(Tensor input, Tensor gating_output, "
        "Tensor fc1_expert_weights, Tensor fc2_expert_weights, "
        "int top_k, Tensor workspace, "
        "int tp_size, int tp_rank, int[]? profile_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_moe", &torch_ext::fused_moe);
}
