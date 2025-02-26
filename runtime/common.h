#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace suggestify::runtime
{

#define FMT_DIM "%ld"

using SizeType32 = std::int32_t;
using SizeType64 = std::int64_t;

enum class RequestType : std::int32_t
{
    kCONTEXT = 0,
    kGENERATION = 1
};

using TokenIdType = std::int32_t;

using LoraTaskIdType = std::uint64_t;
using TokenExtraIdType = std::uint64_t;
using VecTokenExtraIds = std::vector<TokenExtraIdType>;

struct UniqueToken
{
    TokenIdType tokenId;
    TokenExtraIdType tokenExtraId;

    bool operator==(UniqueToken const& other) const noexcept
    {
        return (tokenId == other.tokenId && tokenExtraId == other.tokenExtraId);
    }
};

using VecUniqueTokens = std::vector<UniqueToken>;

template <typename T>
using StringPtrMap = std::unordered_map<std::string, std::shared_ptr<T>>;

}
