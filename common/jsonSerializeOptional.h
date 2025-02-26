
#pragma once
#include <nlohmann/json.hpp>
#include <optional>

namespace nlohmann
{

template <typename T>
struct adl_serializer<std::optional<T>>
{
    static void to_json(nlohmann::json& j, std::optional<T> const& opt)
    {
        if (opt == std::nullopt)
        {
            j = nullptr;
        }
        else
        {
            j = opt.value();
        }
    }

    static void from_json(nlohmann::json const& j, std::optional<T>& opt)
    {
        if (j.is_null())
        {
            opt = std::nullopt;
        }
        else
        {
            opt = j.template get<T>();
        }
    }
};
}
