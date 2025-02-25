#include <gtest/gtest.h>
#include "../src/gpuTypes.h"

/// <summary>
/// Test fixture for GpuBuffer class.
/// </summary>
class TestGpuBuffer : public ::testing::Test {
protected:
    /// <summary>
    /// Sets up the test fixture.
    /// </summary>
    void SetUp() override {
    }

    /// <summary>
    /// Tears down the test fixture.
    /// </summary>
    void TearDown() override {
    }
};

/// <summary>
/// Tests the Resize() method of the GpuBuffer class.
/// </summary>
TEST_F(TestGpuBuffer, testResize) {
    size_t length = 1024;
    GpuBuffer<uint32_t> buff(length, false, true);

    for (size_t i = 0; i < length; ++i) {
        buff._pDevData[i] = i;
    }

    for (uint32_t i = 0; i < length; ++i) {
        EXPECT_EQ(i, buff._pDevData[i]);
    }

    buff.Resize(length);
    for (uint32_t i = 0; i < length; ++i) {
        EXPECT_EQ(i, buff._pDevData[i]);
    }

    buff.Resize(length - 1);
    for (uint32_t i = 0; i < length - 1; ++i) {
        EXPECT_EQ(i, buff._pDevData[i]);
    }

    buff.Resize(length + 1);
    bool isSame = true;
    for (uint32_t i = 0; i < length; ++i) {
        isSame &= (buff._pDevData[i] == i);
    }
    EXPECT_FALSE(isSame);
}

/// <summary>
/// Main function for the test program.
/// </summary>
/// <param name="argc">Number of command line arguments.</param>
/// <param name="argv">Array of command line arguments.</param>
/// <returns>Exit code.</returns>
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}