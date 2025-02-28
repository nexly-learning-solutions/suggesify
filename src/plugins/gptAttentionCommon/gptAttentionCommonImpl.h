
#pragma once

#include "gptAttentionCommon.h"

namespace suggestify::plugins
{
template <typename T>
T* GPTAttentionPluginCommon::cloneImpl() const noexcept
{
    static_assert(std::is_base_of_v<GPTAttentionPluginCommon, T>);
    auto* plugin = new T(static_cast<T const&>(*this));
    plugin->setPluginNamespace(mNamespace.c_str());

    plugin->initialize();
    return plugin;
}

template <typename T>
T* GPTAttentionPluginCreatorCommon::deserializePluginImpl(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new T(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
}
