#include <cstdlib>
#include <gtest/gtest.h>

#include "TestNetCDFhelper.cpp"
#include "TestUtils.cpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS() ? EXIT_SUCCESS : EXIT_FAILURE;
}
