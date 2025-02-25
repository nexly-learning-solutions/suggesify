#pragma once

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <limits>
#include <gtest/gtest.h>

#include "../src/gpuTypes.h"
#include "../src/types.h"
#include "../src/utils/netCDFhelper.h"

#define TEST_DATA_PATH "../../../../tests/test_data/"

enum TestDataType {
    Regression = 1,
    Classification = 2,
    ClassificationAnalog = 3,
};

struct DataParameters {
    DataParameters() {
        numberOfSamples = 1024;
        inpFeatureDimensionality = 1;
        outFeatureDimensionality = 1;
        W0 = -2.f;
        B0 = 3.f;
    }
    int numberOfSamples;
    int inpFeatureDimensionality;
    int outFeatureDimensionality;
    float W0;
    float B0;
};

inline void generateTestData(const std::string& path, const TestDataType testDataType, const DataParameters& dataParameters, std::ostream& out) {
    std::vector<std::vector<unsigned int>> vSampleTestInput, vSampleTestInputTime;
    std::vector<std::vector<float>> vSampleTestInputData;
    std::vector<std::vector<unsigned int>> vSampleTestOutput, vSampleTestOutputTime;
    std::vector<std::vector<float>> vSampleTestOutputData;
    std::vector<std::string> vSamplesName(dataParameters.numberOfSamples);
    std::map<std::string, unsigned int> mFeatureNameToIndex;

    for (int d = 0; d < dataParameters.inpFeatureDimensionality; d++) {
        std::string feature_name = "feature" + std::to_string(static_cast<long long>(d));
        mFeatureNameToIndex[feature_name] = d;
    }

    for (int s = 0; s < dataParameters.numberOfSamples; s++) {
        vSamplesName[s] = "sample" + std::to_string(static_cast<long long>(s));
    }

    for (int s = 0; s < dataParameters.numberOfSamples; s++) {
        std::vector<unsigned int> inpFeatureIndex, inpTime;
        std::vector<float> inpFeatureValue;
        std::vector<unsigned int> outFeatureIndex, outTime;
        std::vector<float> outFeatureValue;

        switch (testDataType) {
        case Regression:
            for (int d = 0; d < dataParameters.inpFeatureDimensionality; d++) {
                inpFeatureIndex.push_back(d);
                inpFeatureValue.push_back(static_cast<float>(s));
                inpTime.push_back(s);
            }

            for (int d = 0; d < dataParameters.outFeatureDimensionality; d++) {
                outFeatureIndex.push_back(d);
                outFeatureValue.push_back(dataParameters.W0 * inpFeatureValue[d] + dataParameters.B0);
                outTime.push_back(s);
            }

            vSampleTestInput.push_back(inpFeatureIndex);
            vSampleTestInputData.push_back(inpFeatureValue);
            vSampleTestInputTime.push_back(inpTime);
            vSampleTestOutput.push_back(outFeatureIndex);
            vSampleTestOutputData.push_back(outFeatureValue);
            vSampleTestOutputTime.push_back(outTime);
            break;
        case Classification:
            inpFeatureIndex.push_back(s % dataParameters.inpFeatureDimensionality);
            inpTime.push_back(s);

            outFeatureIndex.push_back(s % dataParameters.outFeatureDimensionality);
            outTime.push_back(s);

            vSampleTestInput.push_back(inpFeatureIndex);
            vSampleTestInputTime.push_back(inpTime);
            vSampleTestOutput.push_back(outFeatureIndex);
            vSampleTestOutputTime.push_back(outTime);
            break;
        case ClassificationAnalog:
            inpFeatureIndex.push_back(s % dataParameters.inpFeatureDimensionality);
            inpFeatureValue.push_back(static_cast<float>(s));
            inpTime.push_back(s);

            for (int d = 0; d < dataParameters.outFeatureDimensionality; d++) {
                outFeatureIndex.push_back(d);
                outFeatureValue.push_back(((s + d) % 2) + 1);
                outTime.push_back(s);
            }
            vSampleTestInput.push_back(inpFeatureIndex);
            vSampleTestInputData.push_back(inpFeatureValue);
            vSampleTestInputTime.push_back(inpTime);

            vSampleTestOutput.push_back(outFeatureIndex);
            vSampleTestOutputData.push_back(outFeatureValue);
            vSampleTestOutputTime.push_back(outTime);

            break;
        default:
            out << "unsupported mode";
            exit(2);
        }
    }

    int minInpDate = std::numeric_limits<int>::max(), maxInpDate = std::numeric_limits<int>::min(),
                    minOutDate = std::numeric_limits<int>::max(), maxOutDate = std::numeric_limits<int>::min();
    const bool alignFeatureNumber = false;
    writeNETCDF(path + "test.nc", vSamplesName, mFeatureNameToIndex, vSampleTestInput, vSampleTestInputTime,
                    vSampleTestInputData, mFeatureNameToIndex, vSampleTestOutput, vSampleTestOutputTime,
                    vSampleTestOutputData, minInpDate, maxInpDate, minOutDate, maxOutDate, alignFeatureNumber, 2);
}

inline bool validateNeuralNetwork(const size_t batch, const std::string& modelPath, const TestDataType testDataType, const DataParameters& dataParameters, std::ostream& out) {
    out << "start validation of " << modelPath << std::endl;

    Network* pNetwork = nullptr;
    std::vector<DataSetBase*> vDataSet;
    const std::string dataName = "test.nc";
    const std::string dataPath(TEST_DATA_PATH);
    generateTestData(dataPath, testDataType, dataParameters, out);
    vDataSet = LoadNetCDF(dataPath + dataName);
    pNetwork = LoadNeuralNetworkJSON(modelPath, batch, vDataSet);
    pNetwork->LoadDataSets(vDataSet);
    pNetwork->SetCheckpoint("check", 1);
    pNetwork->SetTrainingMode(SGD);
    bool valid = pNetwork->Validate();
    if (valid) {
        out << "SUCCESSFUL validation" << std::endl;
    } else {
        out << "FAILED validation" << std::endl;
    }

    int totalGPUMemory, totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    out << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    out << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    delete pNetwork;

    for (auto p : vDataSet) {
        delete p;
    }
    return valid;
}