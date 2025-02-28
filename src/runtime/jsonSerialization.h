
#pragma once

#include "../common/jsonSerializeOptional.h"
#include "common.h"
#include "runtimeDefaults.h"
#include <nlohmann/json.hpp>

namespace suggestify::runtime
{

void to_json(nlohmann::json& json, RuntimeDefaults const& runtimeDefaults)
{
    json = nlohmann::json{{"max_attention_window", runtimeDefaults.maxAttentionWindowVec},
        {"sink_token_length", runtimeDefaults.sinkTokenLength}};
}

void from_json(nlohmann::json const& json, RuntimeDefaults& runtimeDefaults)
{
    runtimeDefaults.maxAttentionWindowVec = json.value("max_attention_window", runtimeDefaults.maxAttentionWindowVec);
    runtimeDefaults.sinkTokenLength = json.value("sink_token_length", runtimeDefaults.sinkTokenLength);
}


}
