
#pragma once

#include <functional>
#include <memory>
#include <optional>

namespace suggestify::common
{

template <typename T>
class OptionalRef
{
private:
    std::optional<std::reference_wrapper<T>> opt;

public:
    OptionalRef() = default;

    OptionalRef(T& ref)
        : opt(std::ref(ref))
    {
    }

    OptionalRef(std::nullopt_t)
        : opt(std::nullopt)
    {
    }

    OptionalRef(std::shared_ptr<T> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    template <typename U = T, typename = std::enable_if_t<std::is_const_v<U>>>
    OptionalRef(std::shared_ptr<std::remove_const_t<T>> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    OptionalRef(std::unique_ptr<T> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    template <typename U = T, typename = std::enable_if_t<std::is_const_v<U>>>
    OptionalRef(std::unique_ptr<std::remove_const_t<T>> const& ptr)
        : opt(ptr ? std::optional<std::reference_wrapper<T>>(std::ref(*ptr)) : std::nullopt)
    {
    }

    T* operator->() const
    {
        return opt ? &(opt->get()) : nullptr;
    }

    T& operator*() const
    {
        return opt->get();
    }

    explicit operator bool() const
    {
        return opt.has_value();
    }

    bool has_value() const
    {
        return opt.has_value();
    }

    T& value() const
    {
        return opt->get();
    }
};

}
