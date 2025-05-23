
#include "loraModule.h"

namespace suggestify::runtime
{

std::vector<LoraModule> LoraModule::createLoraModules(std::vector<std::string> const& loraModuleNames,
    SizeType32 hiddenSize, SizeType32 mlpHiddenSize, SizeType32 numAttentionHeads, SizeType32 numKvAttentionHeads,
    SizeType32 attentionHeadSize, SizeType32 tpSize, SizeType32 numExperts)
{
    auto const hidden = hiddenSize * tpSize;
    auto const mlpHidden = mlpHiddenSize * tpSize;
    auto const numHeads = numAttentionHeads * tpSize;
    auto const numKvHeads = numKvAttentionHeads * tpSize;
    auto const attnHeadSize = attentionHeadSize;

    std::vector<LoraModule> modules;

    for (auto const& moduleName : loraModuleNames)
    {
        auto t = toModuleType(moduleName);
        switch (t)
        {

        case ModuleType::kATTN_QKV:
        case ModuleType::kCROSS_ATTN_QKV:
            modules.emplace_back(
                t, hidden, (numHeads * attnHeadSize + 2 * numKvHeads * attnHeadSize), false, true, -1, 0);
            break;
        case ModuleType::kATTN_Q:
        case ModuleType::kATTN_K:
        case ModuleType::kATTN_V:
        case ModuleType::kCROSS_ATTN_Q:
        case ModuleType::kCROSS_ATTN_K:
        case ModuleType::kCROSS_ATTN_V: modules.emplace_back(t, hidden, hidden, false, true, -1, 0); break;
        case ModuleType::kATTN_DENSE:
        case ModuleType::kCROSS_ATTN_DENSE: modules.emplace_back(t, hidden, hidden, false, true, 1, -1); break;
        case ModuleType::kMLP_H_TO_4H: modules.emplace_back(t, hidden, mlpHidden, false, true, -1, 0); break;
        case ModuleType::kMLP_GATE: modules.emplace_back(t, hidden, mlpHidden, false, true, -1, 0); break;
        case ModuleType::kMLP_4H_TO_H: modules.emplace_back(t, mlpHiddenSize, hidden, false, true, 1, -1); break;
        case ModuleType::kMOE_H_TO_4H:
        case ModuleType::kMOE_GATE: modules.emplace_back(t, hidden, mlpHidden, false, true, -1, 0); break;
        case ModuleType::kMOE_4H_TO_H: modules.emplace_back(t, mlpHiddenSize, hidden, false, true, 1, -1); break;
        case ModuleType::kMOE_ROUTER: modules.emplace_back(t, hidden, numExperts, false, true, -1, -1); break;
        case ModuleType::kMLP_ROUTER: modules.emplace_back(t, hidden, 1, false, true, -1, -1); break;
        case ModuleType::kINVALID: throw std::runtime_error("Invalid LoRA module " + moduleName);
        }
    }
    return modules;
}
}
