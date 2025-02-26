
#pragma once

#include <cstdint>

namespace suggestify::runtime
{

class SpeculativeDecodingMode
{
public:
    static auto constexpr None()
    {
        return SpeculativeDecodingMode{kNone};
    }

    static auto constexpr DraftTokensExternal()
    {
        return SpeculativeDecodingMode{kDraftTokensExternal};
    }

    static auto constexpr Medusa()
    {
        return SpeculativeDecodingMode{kMedusa};
    }

    static auto constexpr LookaheadDecoding()
    {
        return SpeculativeDecodingMode{kLookaheadDecoding};
    }

    static auto constexpr ExplicitDraftTokens()
    {
        return SpeculativeDecodingMode{kExplicitDraftTokens};
    }

    static auto constexpr Eagle()
    {
        return SpeculativeDecodingMode{kEagle};
    }

    [[nodiscard]] bool constexpr isNone() const
    {
        return anyBitSet(kNone);
    }

    [[nodiscard]] bool constexpr isDraftTokensExternal() const
    {
        return anyBitSet(kDraftTokensExternal);
    }

    [[nodiscard]] bool constexpr isMedusa() const
    {
        return anyBitSet(kMedusa);
    }

    [[nodiscard]] bool constexpr isLookaheadDecoding() const
    {
        return anyBitSet(kLookaheadDecoding);
    }

    [[nodiscard]] bool constexpr isExplicitDraftTokens() const
    {
        return anyBitSet(kExplicitDraftTokens);
    }

    [[nodiscard]] bool constexpr isEagle() const
    {
        return anyBitSet(kEagle);
    }

    [[nodiscard]] bool constexpr updatesPositionIds() const
    {
        return anyBitSet(kLookaheadDecoding | kExplicitDraftTokens);
    }

    [[nodiscard]] bool constexpr requiresAttentionMask() const
    {
        return anyBitSet(kLookaheadDecoding | kMedusa | kExplicitDraftTokens | kEagle);
    }

    [[nodiscard]] bool constexpr predictsDraftTokens() const
    {
        return anyBitSet(kLookaheadDecoding | kMedusa | kExplicitDraftTokens | kEagle);
    }

    [[nodiscard]] bool constexpr needsKVCacheRewind() const
    {
        return anyBitSet(kLookaheadDecoding | kMedusa | kExplicitDraftTokens | kEagle);
    }

    [[nodiscard]] bool constexpr variableDraftLength() const
    {
        return anyBitSet(kDraftTokensExternal | kExplicitDraftTokens | kLookaheadDecoding | kEagle);
    }

    [[nodiscard]] bool constexpr hasDraftLogits() const
    {
        return anyBitSet(kMedusa);
    }

    [[nodiscard]] bool constexpr needsDecoderPrologue() const
    {
        return anyBitSet(kExplicitDraftTokens | kLookaheadDecoding | kEagle);
    }

    using UnderlyingType = std::uint8_t;

    bool operator==(SpeculativeDecodingMode const& other) const
    {
        return mState == other.mState;
    }

    explicit constexpr SpeculativeDecodingMode(UnderlyingType state)
        : mState(state)
    {
    }

private:
    static UnderlyingType constexpr kNone{1U << 0U};
    static UnderlyingType constexpr kDraftTokensExternal{1U << 1U};
    static UnderlyingType constexpr kMedusa{1U << 2U};
    static UnderlyingType constexpr kLookaheadDecoding{1U << 3U};
    static UnderlyingType constexpr kExplicitDraftTokens{1U << 4U};
    static UnderlyingType constexpr kEagle{1U << 5U};

    [[nodiscard]] bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    [[nodiscard]] bool constexpr allBitSet(UnderlyingType bits) const
    {
        return (mState & bits) == bits;
    }

    UnderlyingType mState{kNone};
};

static_assert(SpeculativeDecodingMode::None().isNone());
static_assert(!SpeculativeDecodingMode::None().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::None().isMedusa());
static_assert(!SpeculativeDecodingMode::None().isLookaheadDecoding());
static_assert(!SpeculativeDecodingMode::None().isExplicitDraftTokens());

static_assert(SpeculativeDecodingMode::DraftTokensExternal().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::DraftTokensExternal().isNone());
static_assert(!SpeculativeDecodingMode::DraftTokensExternal().isMedusa());
static_assert(!SpeculativeDecodingMode::DraftTokensExternal().isLookaheadDecoding());
static_assert(!SpeculativeDecodingMode::DraftTokensExternal().isExplicitDraftTokens());

static_assert(SpeculativeDecodingMode::Medusa().isMedusa());
static_assert(!SpeculativeDecodingMode::Medusa().isNone());
static_assert(!SpeculativeDecodingMode::Medusa().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::Medusa().isLookaheadDecoding());
static_assert(!SpeculativeDecodingMode::Medusa().isExplicitDraftTokens());

static_assert(SpeculativeDecodingMode::LookaheadDecoding().isLookaheadDecoding());
static_assert(!SpeculativeDecodingMode::LookaheadDecoding().isNone());
static_assert(!SpeculativeDecodingMode::LookaheadDecoding().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::LookaheadDecoding().isMedusa());
static_assert(!SpeculativeDecodingMode::LookaheadDecoding().isExplicitDraftTokens());

static_assert(SpeculativeDecodingMode::ExplicitDraftTokens().isExplicitDraftTokens());
static_assert(!SpeculativeDecodingMode::ExplicitDraftTokens().isNone());
static_assert(!SpeculativeDecodingMode::ExplicitDraftTokens().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::ExplicitDraftTokens().isMedusa());
static_assert(!SpeculativeDecodingMode::ExplicitDraftTokens().isLookaheadDecoding());

static_assert(SpeculativeDecodingMode::Eagle().isEagle());
static_assert(!SpeculativeDecodingMode::Eagle().isNone());
static_assert(!SpeculativeDecodingMode::Eagle().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::Eagle().isMedusa());
static_assert(!SpeculativeDecodingMode::Eagle().isExplicitDraftTokens());
static_assert(!SpeculativeDecodingMode::Eagle().isLookaheadDecoding());

}
