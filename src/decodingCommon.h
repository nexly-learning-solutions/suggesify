
#pragma once

#include "../common/assert.h"
#include "../executor/types.h"
#include <cstdint>
#include <curand_kernel.h>

namespace suggestify::kernels
{

class FinishedState
{
public:
    static auto constexpr empty()
    {
        return FinishedState{0};
    }

    static auto constexpr finished()
    {
        return FinishedState{kFinished};
    }

    static auto constexpr skipDecoding()
    {
        return FinishedState{kSkipDecoding};
    }

    static auto constexpr finishedEOS()
    {
        return FinishedState{kFinishedEos};
    }

    static auto constexpr finishedMaxLength()
    {
        return FinishedState{kFinishedMaxLength};
    }

    static auto constexpr finishedStopWords()
    {
        return FinishedState{kFinishedStopWords};
    }

    __host__ __device__ void constexpr setFinishedEOS()
    {
        mState |= kFinishedEos;
    }

    __host__ __device__ bool constexpr isFinishedEOS() const
    {
        return anyBitSet(kFinishedEos);
    }

    __host__ __device__ void constexpr setFinishedStopWords()
    {
        mState |= kFinishedStopWords;
    }

    __host__ __device__ bool constexpr isFinishedStopWords() const
    {
        return anyBitSet(kFinishedStopWords);
    }

    __host__ __device__ void constexpr setFinishedMaxLength()
    {
        mState |= kFinishedMaxLength;
    }

    __host__ __device__ bool constexpr isFinishedMaxLength() const
    {
        return anyBitSet(kFinishedMaxLength);
    }

    __host__ __device__ void constexpr setFinished()
    {
        mState |= kFinished;
    }

    __host__ __device__ bool constexpr isFinished() const
    {
        return anyBitSet(kFinished);
    }

    __host__ __device__ void constexpr setSkipDecoding()
    {
        mState = kSkipDecoding;
    }

    __host__ __device__ bool constexpr isSkipDecoding() const
    {
        return anyBitSet(kSkipDecoding);
    }

    [[nodiscard]] constexpr executor::FinishReason toFinishReason() const
    {
        if (isFinishedEOS())
        {
            return executor::FinishReason::kEND_ID;
        }
        if (isFinishedStopWords())
        {
            return executor::FinishReason::kSTOP_WORDS;
        }
        if (isFinishedMaxLength())
        {
            return executor::FinishReason::kLENGTH;
        }
        return executor::FinishReason::kNOT_FINISHED;
    }

    using UnderlyingType = uint8_t;

    [[nodiscard]] constexpr UnderlyingType toUnderlying() const noexcept
    {
        return mState;
    }

private:
    __host__ __device__ constexpr FinishedState(UnderlyingType state)
        : mState(state)
    {
    }

    static UnderlyingType constexpr kFinishedEos{1u << 0};
    static UnderlyingType constexpr kFinishedStopWords{1u << 1};
    static UnderlyingType constexpr kFinishedMaxLength{1u << 2};
    static UnderlyingType constexpr kFinished{kFinishedEos | kFinishedStopWords | kFinishedMaxLength};
    static UnderlyingType constexpr kSkipDecoding{1u << 3};

    __host__ __device__ bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    UnderlyingType mState{};
};

static_assert(!FinishedState::empty().isFinished());
static_assert(!FinishedState::empty().isSkipDecoding());
static_assert(FinishedState::finished().isFinished());
static_assert(FinishedState::skipDecoding().isSkipDecoding());
static_assert(FinishedState::finishedEOS().isFinishedEOS());
static_assert(FinishedState::finishedStopWords().isFinishedStopWords());
static_assert(FinishedState::finishedMaxLength().isFinishedMaxLength());

template <typename T>
struct ScatterDecodingParamEntry
{
    T const* mVector;
    T mScalar;
    T* mTarget;

    ScatterDecodingParamEntry() = default;

    ScatterDecodingParamEntry(T const* vector, T scalar, T* target)
        : mVector(vector)
        , mScalar(scalar)
        , mTarget(target)
    {
    }

    ScatterDecodingParamEntry(void const* vector, T scalar, T* target)
        : ScatterDecodingParamEntry(static_cast<T const*>(vector), scalar, target)
    {
    }
};

void invokeCurandInitialize(
    curandState_t* state, int const* batchSlots, size_t const batchSize, uint64_t randomSeed, cudaStream_t stream);

void invokeCurandBatchInitialize(curandState_t* states, int const* batchSlots, size_t const batchSize,
    uint64_t const* randomSeeds, cudaStream_t stream);

template <typename T>
struct BiasSoftmaxParams
{
    T* logits{nullptr};
    T** logitsPtrs{nullptr};
    T* probs{nullptr};
    float* outputEntropy{nullptr};
    T const* bias{nullptr};
    float const* temperatures{nullptr};
    int32_t const* endIds{nullptr};
    FinishedState const* finished{nullptr};
    int32_t const* beamWidths{nullptr};
    int32_t const* batchSlots{nullptr};
    int32_t batchSize{0};
    int32_t maxBatchSize{0};
    int32_t maxBeamWidth{0};
    int32_t vocabSize{0};
    int32_t vocabSizePadded{0};
    bool skipSoftMax{false};
    bool batchSlotsLogits{false};
    bool ptrsForBeams{false};

    void checkParams()
    {
        TLLM_CHECK(logits || logitsPtrs);
        TLLM_CHECK(((outputEntropy != nullptr) && (probs != nullptr)) || (outputEntropy == nullptr));
        TLLM_CHECK(((outputEntropy != nullptr) && !skipSoftMax) || (outputEntropy == nullptr));

        if (batchSlotsLogits)
        {
            TLLM_CHECK(batchSlots);
        }

        if (ptrsForBeams)
        {
            TLLM_CHECK(logitsPtrs);
        }

        TLLM_CHECK(batchSize > 0);
        TLLM_CHECK(maxBatchSize > 0);
        TLLM_CHECK(batchSize <= maxBatchSize);
        TLLM_CHECK(maxBeamWidth > 0);
        TLLM_CHECK(vocabSize > 0);
        TLLM_CHECK(vocabSizePadded > 0);
        TLLM_CHECK(vocabSize <= vocabSizePadded);
    }
};

template <typename T>
void invokeAddBiasSoftMax(BiasSoftmaxParams<T> const params, cudaStream_t stream);

template <typename T>
void invokeScatterDecodingParams(
    T const* src, T scalar, T* dst, int const* batchSlots, int batchSize, cudaStream_t stream);

}
