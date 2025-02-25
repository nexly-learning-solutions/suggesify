#include <gtest/gtest.h>
#include <string>
#include "Utils.h"
#include "../src/gpuTypes.h"
#include "../src/types.h"
#include "TestUtils.h"

class TestCostFunctions : public ::testing::Test {
protected:
    void SetUp() override {
        getGpu().SetRandomSeed(12345);
        getGpu().CopyConstants();
    }
    
    void TearDown() override {
    }
};

TEST_F(TestCostFunctions, TestCostFunctions) {
    {
        const size_t batch = 2;
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_DataScaledMarginalCrossEntropy_02.json";
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 1024;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        bool result = validateNeuralNetwork(batch, modelPath, ClassificationAnalog, dataParameters, std::cout);
        EXPECT_TRUE(result) << "Failed on DataScaledMarginalCrossEntropy";
    }

    {
        const size_t batch = 4;
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_ScaledMarginalCrossEntropy_02.json";
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 1024;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        bool result = validateNeuralNetwork(batch, modelPath, Classification, dataParameters, std::cout);
        EXPECT_TRUE(result) << "Failed on DataScaledMarginalCrossEntropy";
    }

    {
        const size_t batch = 4;
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_02.json";
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 1024;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        dataParameters.W0 = -2.f;
        dataParameters.B0 = 3.f;
        bool result = validateNeuralNetwork(batch, modelPath, Regression, dataParameters, std::cout);
        EXPECT_TRUE(result) << "Failed on DataScaledMarginalCrossEntropy";
    }
}