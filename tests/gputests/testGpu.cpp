#include <gtest/gtest.h>
#include <string>

#include "filterKernels.h"

class TestGpu : public ::testing::Test {
protected:
    void SetUp() override {
        getGpu().Startup(0, nullptr);
    }

    void TearDown() override {
        getGpu().Shutdown();
    }
};

TEST_F(TestGpu, ApplyNodeFilter) {
    int outputKeySize = 6;
    int filterSize = 3;
    float localOutputKey[outputKeySize] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float expectedOutputKey[outputKeySize] = {7.0, 16.0, 27.0, 28.0, 40.0, 54.0};
    float localFilter[filterSize] = {7.0, 8.0, 9.0};
    float expectedFilter[filterSize] = {7.0, 8.0, 9.0};

    auto deviceOutputKey = std::make_unique<GpuBuffer<float>>(outputKeySize);
    auto deviceFilter = std::make_unique<GpuBuffer<float>>(filterSize);

    deviceOutputKey->Upload(localOutputKey);
    deviceFilter->Upload(localFilter);

    kApplyNodeFilter(deviceOutputKey->_pDevData, deviceFilter->_pDevData, filterSize, 2);

    deviceOutputKey->Download(localOutputKey);
    deviceFilter->Download(localFilter);

    for (int i = 0; i < outputKeySize; ++i) {
        EXPECT_NEAR(localOutputKey[i], expectedOutputKey[i], 1e-5) << "OutputKey is different at index " << i;
    }

    for (int i = 0; i < filterSize; ++i) {
        EXPECT_NEAR(localFilter[i], expectedFilter[i], 1e-5) << "Filter is different at index " << i;
    }
}
