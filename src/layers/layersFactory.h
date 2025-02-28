
#pragma once

#include "banWordsLayer.h"
#include "baseLayer.h"
#include "decodingLayer.h"
#include "penaltyLayer.h"
#include "stopCriteriaLayer.h"
#include <memory>
#include <vector>

namespace suggestify::layers
{
enum DecodingLayers_t
{
    PENALTY_LAYER,
    BAN_WORDS_LAYER,
    DECODING_LAYER,
    STOP_CRITERIA_LAYER
};

static std::vector<DecodingLayers_t> createDecodingLayerTypes(executor::DecodingMode const& mode)
{
    std::vector<DecodingLayers_t> types = {};
    if (mode.isUsePenalty())
    {
        types.push_back(DecodingLayers_t::PENALTY_LAYER);
    }
    if (mode.isUseBanWords())
    {
        types.push_back(DecodingLayers_t::BAN_WORDS_LAYER);
    }
    types.push_back(DecodingLayers_t::DECODING_LAYER);
    if (mode.isUseStopCriteria())
    {
        types.push_back(DecodingLayers_t::STOP_CRITERIA_LAYER);
    }
    return types;
}

template <typename T>
static std::vector<std::unique_ptr<BaseLayer>> createLayers(executor::DecodingMode const& mode,
    DecoderDomain const& decodingDomain, std::shared_ptr<runtime::BufferManager> const& bufferManager)
{
    std::vector<std::unique_ptr<BaseLayer>> layers;
    auto layerTypes = createDecodingLayerTypes(mode);
    if (!mode.isExplicitDraftTokens() && !mode.isEagle())
    {
        CHECK_WITH_INFO(layerTypes.size() && layerTypes[0] == DecodingLayers_t::PENALTY_LAYER,
            "Penalty layer is required to be the first layer for any decoder configuration");
    }
    for (auto&& type : layerTypes)
    {
        std::unique_ptr<BaseLayer> layer;
        switch (type)
        {
        case DecodingLayers_t::PENALTY_LAYER:
            layer = std::make_unique<PenaltyLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        case DecodingLayers_t::BAN_WORDS_LAYER:
            layer = std::make_unique<BanWordsLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        case DecodingLayers_t::DECODING_LAYER:
            layer = std::make_unique<DecodingLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        case DecodingLayers_t::STOP_CRITERIA_LAYER:
            layer = std::make_unique<StopCriteriaLayer<T>>(mode, decodingDomain, bufferManager);
            break;

        default: CHECK_WITH_INFO(false, "Unknown DecodingLayers_t"); break;
        }
        layers.push_back(std::move(layer));
    }
    return layers;
}
}
