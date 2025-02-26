#include "runtime.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace
{
constexpr int ARGC = 1;
char ARG0[] = "process";
char* ARGV[] = {ARG0};
constexpr unsigned long SEED = 12134ULL;
}

runtime::runtime(std::string const& networkFilename, uint32_t batchSize, int maxK)
    : networkFilename(networkFilename)
    , batchSize(batchSize)
    , maxK(maxK)
{
    getGpu().Startup(ARGC, ARGV);
    getGpu().SetRandomSeed(SEED);

    auto network = LoadNeuralNetworkNetCDF(networkFilename, batchSize);
    getGpu().SetNeuralNetwork(network);

    std::vector<Layer const*> outputLayers;
    network->GetLayers(Layer::Kind::Output, outputLayers);

#pragma omp parallel for
    for (size_t i = 0; i < outputLayers.size(); ++i)
    {
        auto const* layer = outputLayers[i];
        std::string const& layerName = layer->GetName();

        if (maxK != ALL)
        {
            if (layer->GetNumDimensions() > 1)
            {
                throw std::runtime_error("topK only supported on 1-D output layers");
            }
            size_t outputBufferLength = static_cast<size_t>(maxK) * batchSize;
            std::cout << "runtime::runtime: Allocating output score and index buffers, each of size "
                      << outputBufferLength << " for output layer " << layerName << '\n';

#pragma omp critical
            {
                dOutputScores.emplace(layerName, std::make_unique<GpuBuffer<float>>(outputBufferLength, false, false));
                dOutputIndexes.emplace(
                    layerName, std::make_unique<GpuBuffer<uint32_t>>(outputBufferLength, false, false));
            }
        }
    }
}

runtime::~runtime()
{
    std::string const networkName = getNetwork()->GetName();
    dOutputScores.clear();
    dOutputIndexes.clear();

    delete getNetwork();
    getGpu().Shutdown();
    std::cout << "runtime::~runtime: Destroyed context for network " << networkName << '\n';
}

GpuBuffer<float>* runtime::getOutputScoresBuffer(std::string const& layerName)
{
    return dOutputScores.at(layerName).get();
}

GpuBuffer<uint32_t>* runtime::getOutputIndexesBuffer(std::string const& layerName)
{
    return dOutputIndexes.at(layerName).get();
}

Network* runtime::getNetwork() const
{
    return getGpu().network;
}

void runtime::initInputLayerDataSets(std::vector<DataSetDescriptor> const& datasetDescriptors)
{
    std::vector<DataSetBase*> datasets;
    for (auto const& descriptor : datasetDescriptors)
    {
        datasets.push_back(createDataSet(descriptor));
    }

    getNetwork()->PredictBatch();

    getNetwork()->SetPosition(0);
}