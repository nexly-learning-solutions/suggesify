#include <gtest/gtest.h>
#include <cstdlib>

#include "TestSort.cpp"
#include "TestActivationFunctions.cpp"
#include "TestCostFunctions.cpp"

int main(int argc, char **argv) {
    getGpu().Startup(0, nullptr);
    getGpu().SetRandomSeed(12345);
    getGpu().CopyConstants();

    ::testing::InitGoogleTest(&argc, argv);

    const int result = RUN_ALL_TESTS();

    getGpu().Shutdown();

    return result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
