
#pragma once

#include "../common/logger.h"
#include "../executor/executor.h"
#include "../layers/defaultDecodingParams.h"
#include "common.h"

#include <functional>
#include <optional>
#include <vector>

namespace suggestify::runtime
{

class SamplingConfig
{
private:
    using FloatType = float;

    template <typename T>
    using OptVec = std::optional<std::vector<T>>;

    template <typename T>
    static OptVec<T> fuseValues(
        std::vector<SamplingConfig> const& configs, std::function<OptVec<T>(size_t ci)> accessor, T defaultValue)
    {
        std::vector<T> values;
        bool atLeastOneHasValue{false};
        for (size_t ci = 0; ci < configs.size(); ++ci)
        {
            auto const& configValue = accessor(ci);
            if (configValue.has_value())
            {
                atLeastOneHasValue = true;
                break;
            }
        }
        if (atLeastOneHasValue)
        {
            for (size_t ci = 0; ci < configs.size(); ++ci)
            {
                auto value = defaultValue;
                auto const& configValue = accessor(ci);
                if (configValue.has_value())
                {
                    CHECK(configValue.value().size() == 1);
                    value = configValue.value().front();
                }
                values.push_back(value);
            }

            return std::make_optional<std::vector<T>>(values);
        }
        else
        {
            return std::nullopt;
        }
    }

    template <typename T>
    using Vec = std::vector<T>;

    template <typename T>
    bool validateVec(std::string name, OptVec<T> const& vec, T min, std::optional<T> max = std::nullopt)
    {
        bool valid{true};
        if (vec)
        {
            valid = std::all_of(vec->begin(), vec->end(),
                [min, max](T elem)
                { return min < elem && ((max.has_value() && elem <= max.value()) || (!max.has_value())); });
            if (!valid)
            {
                std::stringstream ss;
                ss << "Incorrect sampling param. " << name << " is out of range (";
                ss << min << ", ";
                if (max.has_value())
                {
                    ss << max.value();
                }
                else
                {
                    ss << "inf";
                }
                ss << "]";
                LOG_WARNING(valid, ss.str());
            }
        }
        return valid;
    }

public:
    explicit SamplingConfig(SizeType32 beamWidth = 1)
        : beamWidth{beamWidth}
    {
    }

    explicit SamplingConfig(std::vector<SamplingConfig> const& configs)
    {
        CHECK(configs.size() > 0);
        beamWidth = configs.front().beamWidth;
        numReturnSequences = configs.front().numReturnSequences;
        normalizeLogProbs = configs.front().normalizeLogProbs;
        temperature = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].temperature; },
            layers::DefaultDecodingParams::getTemperature());
        originalTemperature = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].originalTemperature; },
            layers::DefaultDecodingParams::getTemperature());
        minLength = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].minLength; },
            layers::DefaultDecodingParams::getMinLength());
        repetitionPenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].repetitionPenalty; },
            layers::DefaultDecodingParams::getRepetitionPenalty());
        presencePenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].presencePenalty; },
            layers::DefaultDecodingParams::getPresencePenalty());
        frequencyPenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].frequencyPenalty; },
            layers::DefaultDecodingParams::getFrequencyPenalty());
        noRepeatNgramSize = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].noRepeatNgramSize; },
            layers::DefaultDecodingParams::getNoRepeatNgramSize());
        topK = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].topK; }, layers::DefaultDecodingParams::getTopK());
        topP = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].topP; }, layers::DefaultDecodingParams::getTopP());
        randomSeed = fuseValues<uint64_t>(
            configs, [&configs](size_t ci) { return configs[ci].randomSeed; },
            layers::DefaultDecodingParams::getSeed());
        topPDecay = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].topPDecay; },
            layers::DefaultDecodingParams::getTopPDecay());
        topPMin = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].topPMin; },
            layers::DefaultDecodingParams::getTopPMin());
        topPResetIds = fuseValues<TokenIdType>(
            configs, [&configs](size_t ci) { return configs[ci].topPResetIds; },
            layers::DefaultDecodingParams::getTopPResetId());
        beamSearchDiversityRate = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].beamSearchDiversityRate; },
            layers::DefaultDecodingParams::getBeamSearchDiversity());
        lengthPenalty = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].lengthPenalty; },
            layers::DefaultDecodingParams::getLengthPenalty());
        earlyStopping = fuseValues<SizeType32>(
            configs, [&configs](size_t ci) { return configs[ci].earlyStopping; },
            layers::DefaultDecodingParams::getEarlyStopping());
        topKMedusaHeads = fuseValues<std::vector<SizeType32>>(
            configs, [&configs](size_t ci) { return configs[ci].topKMedusaHeads; },
            layers::DefaultDecodingParams::getTopKMedusaHeads());
        outputLogProbs = fuseValues<bool>(
            configs, [&configs](size_t ci) { return configs[ci].outputLogProbs; }, false);
        cumLogProbs = fuseValues<bool>(
            configs, [&configs](size_t ci) { return configs[ci].cumLogProbs; }, false);
        draftAcceptanceThreshold = fuseValues<FloatType>(
            configs, [&configs](size_t ci) { return configs[ci].draftAcceptanceThreshold; }, 0);
    }

    explicit SamplingConfig(executor::SamplingConfig const& samplingConfig,
        std::optional<executor::ExternalDraftTokensConfig> const& externalDraftTokensConfig)
        : beamWidth{samplingConfig.getBeamWidth()}
        , numReturnSequences(samplingConfig.getNumReturnSequences())
    {

        if (externalDraftTokensConfig && externalDraftTokensConfig.value().getAcceptanceThreshold())
        {
            draftAcceptanceThreshold
                = Vec<FloatType>{externalDraftTokensConfig.value().getAcceptanceThreshold().value()};
        }

#define SET_FROM_OPTIONAL(varName, VarName, VarType)                                                                   \
                                                                                                                       \
    if (samplingConfig.get##VarName())                                                                                 \
    {                                                                                                                  \
        varName = Vec<VarType>{samplingConfig.get##VarName().value()};                                                 \
    }

        SET_FROM_OPTIONAL(topK, TopK, SizeType32)
        SET_FROM_OPTIONAL(topP, TopP, FloatType)
        SET_FROM_OPTIONAL(topPMin, TopPMin, FloatType)
        SET_FROM_OPTIONAL(topPResetIds, TopPResetIds, TokenIdType)
        SET_FROM_OPTIONAL(topPDecay, TopPDecay, FloatType)
        SET_FROM_OPTIONAL(randomSeed, Seed, uint64_t)
        SET_FROM_OPTIONAL(temperature, Temperature, FloatType)
        SET_FROM_OPTIONAL(originalTemperature, Temperature, FloatType)
        SET_FROM_OPTIONAL(minLength, MinTokens, SizeType32)
        SET_FROM_OPTIONAL(beamSearchDiversityRate, BeamSearchDiversityRate, FloatType)
        SET_FROM_OPTIONAL(repetitionPenalty, RepetitionPenalty, FloatType)
        SET_FROM_OPTIONAL(presencePenalty, PresencePenalty, FloatType)
        SET_FROM_OPTIONAL(frequencyPenalty, FrequencyPenalty, FloatType)
        SET_FROM_OPTIONAL(lengthPenalty, LengthPenalty, FloatType)
        SET_FROM_OPTIONAL(earlyStopping, EarlyStopping, SizeType32)
        SET_FROM_OPTIONAL(noRepeatNgramSize, NoRepeatNgramSize, SizeType32)
#undef SET_FROM_OPTIONAL
    }

    bool validate()
    {
        auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();

        bool valid{true};

        valid &= (beamWidth > 0);
        if (!valid)
        {
            LOG_WARNING(
                "Requested beam width %d is incorrect. Must be > 0. To de-activate beam searching set beamWidth to 1.",
                beamWidth);
        }

        if (numReturnSequences)
        {
            valid &= (numReturnSequences.value() > 0);
            if (!valid)
            {
                LOG_WARNING(
                    "Requested numReturnSequences %d is incorrect. Must be > 0.", numReturnSequences.value());
            }
            valid &= (beamWidth == 1 || numReturnSequences.value() <= beamWidth);
            if (!valid)
            {
                LOG_WARNING(
                    "Requested numReturnSequences %d is incorrect. In beam search, numReturnSequences should not "
                    "exceed the beam width %d.",
                    numReturnSequences.value(), beamWidth);
            }
        }

        valid &= validateVec("topK", topK, -1);
        valid &= validateVec("topP", topP, -fltEpsilon, {1.f});
        valid &= validateVec("topPMin", topPMin, 0.f, {1.f});
        valid &= validateVec("topPDecay", topPDecay, 0.f, {1.f});
        valid &= validateVec("topPResetIds", topPResetIds, -1);

        valid &= validateVec("temperature", temperature, -fltEpsilon);
        valid &= validateVec("repetitionPenalty", repetitionPenalty, 0.f);
        valid &= validateVec("minLength", minLength, -1);
        valid &= validateVec("noRepeatNgramSize", noRepeatNgramSize, 0);

        valid &= validateVec("beamSearchDiversityRate", beamSearchDiversityRate, -fltEpsilon);

        if (temperature)
        {
            bool saveOriginalTemperature{false};
            if (!originalTemperature)
            {
                saveOriginalTemperature = true;
                originalTemperature = std::vector<FloatType>(temperature->size());
            }

            for (size_t ti = 0; ti < temperature->size(); ++ti)
            {
                if (temperature->at(ti) == 0.f)
                {
                    if (saveOriginalTemperature)
                    {
                        originalTemperature->at(ti) = 0.f;
                    }
                    temperature->at(ti) = 1.0f;

                    if (topK)
                    {
                        topK->at(ti) = 1;
                    }
                    if (topP)
                    {
                        topP->at(ti) = 1.f;
                    }
                }
                else if (saveOriginalTemperature)
                {
                    originalTemperature->at(ti) = temperature->at(ti);
                }
            }
        }

        return valid;
    }

    template <typename T>
    bool useDefaultValues(OptVec<T> const& vec, T defaultValue)
    {
        bool useDefault{true};
        if (vec)
        {
            useDefault = std::all_of(vec->begin(), vec->end(), [defaultValue](T elem) { return elem == defaultValue; });
        }
        return useDefault;
    }

public:
    SizeType32 beamWidth;
    std::optional<SizeType32> numReturnSequences;

    OptVec<FloatType> temperature;
    OptVec<FloatType> originalTemperature;
    OptVec<SizeType32> minLength;
    OptVec<FloatType> repetitionPenalty;
    OptVec<FloatType> presencePenalty;
    OptVec<FloatType> frequencyPenalty;
    OptVec<SizeType32> noRepeatNgramSize;

    OptVec<bool> outputLogProbs;
    OptVec<bool> cumLogProbs;

    OptVec<SizeType32> topK;
    OptVec<FloatType> topP;
    OptVec<uint64_t> randomSeed;
    OptVec<FloatType> topPDecay;
    OptVec<FloatType> topPMin;
    OptVec<TokenIdType> topPResetIds;

    OptVec<FloatType> beamSearchDiversityRate;
    OptVec<FloatType> lengthPenalty;
    OptVec<SizeType32> earlyStopping;

    OptVec<FloatType> draftAcceptanceThreshold;

    OptVec<std::vector<runtime::SizeType32>> topKMedusaHeads;

    std::optional<bool> normalizeLogProbs;

    bool operator==(SamplingConfig const& other) const
    {
        return beamWidth == other.beamWidth && numReturnSequences == other.numReturnSequences
            && temperature == other.temperature && originalTemperature == other.originalTemperature
            && minLength == other.minLength && repetitionPenalty == other.repetitionPenalty
            && presencePenalty == other.presencePenalty && frequencyPenalty == other.frequencyPenalty
            && noRepeatNgramSize == other.noRepeatNgramSize && topK == other.topK && topP == other.topP
            && randomSeed == other.randomSeed && topPDecay == other.topPDecay && topPMin == other.topPMin
            && topPResetIds == other.topPResetIds && beamSearchDiversityRate == other.beamSearchDiversityRate
            && lengthPenalty == other.lengthPenalty && earlyStopping == other.earlyStopping
            && draftAcceptanceThreshold == other.draftAcceptanceThreshold && topKMedusaHeads == other.topKMedusaHeads
            && normalizeLogProbs == other.normalizeLogProbs && outputLogProbs == other.outputLogProbs
            && cumLogProbs == other.cumLogProbs;
    }

    SizeType32 getNumReturnBeams() const
    {
        if (numReturnSequences && beamWidth > 1)
        {
            return std::min(numReturnSequences.value(), beamWidth);
        }
        return beamWidth;
    }
};

}
