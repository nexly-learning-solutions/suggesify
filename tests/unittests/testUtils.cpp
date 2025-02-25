#include <gtest/gtest.h>
#include "Utils.h"

class TestUtils : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(TestUtils, TestIsNetCDFfile) {
    bool result = isNetCDFfile("network.nc");
    EXPECT_TRUE(result);

    result = isNetCDFfile("network.nic");
    EXPECT_FALSE(result);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
