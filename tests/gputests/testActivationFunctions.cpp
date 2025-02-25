#include <gtest/gtest.h>
#include <string>
#include "Utils.h"
#include "../src/gpuTypes.h"
#include "../src/types.h"
#include "TestUtils.h"

class TestActivationFunctions : public ::testing::Test {
protected:
    void SetUp() override {
        getGpu().SetRandomSeed(12345);
        getGpu().CopyConstants();
    }

    void TearDown() override {
    }
};

TEST_F(TestActivationFunctions, TestActivationFunctions) {
    const size_t numberTests = 2;
    const std::string modelPaths[numberTests] = {
        std::string(TEST_DATA_PATH) + "validate_L2_LRelu_01.json",
        std::string(TEST_DATA_PATH) + "validate_L2_LRelu_02.json"
    };
    const size_t batches[numberTests] = {2, 4};

    for (size_t i = 0; i < numberTests; ++i) {
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 1024;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;

        bool result = validateNeuralNetwork(batches[i], modelPaths[i], Classification, dataParameters, std::cout);
        std::cout << "batches " << batches[i] << ", model " << modelPaths[i] << std::endl;

        EXPECT_TRUE(result) << "Failed on testActivationFunctions with model: " << modelPaths[i];
    }
}