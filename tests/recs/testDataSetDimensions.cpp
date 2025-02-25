#include <gtest/gtest.h>

#include "../src/gpuTypes.h"
#include "../src/types.h"
#include "../src/layer.h"

class TestdataSetDimensions : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(TestdataSetDimensions, TestNumDimensions) {
    dataSetDimensions zero_d(1);
    dataSetDimensions one_d(2);
    dataSetDimensions two_d(2, 2);
    dataSetDimensions three_d(2, 2, 2);

    EXPECT_EQ(0U, zero_d._dimensions);
    EXPECT_EQ(1U, one_d._dimensions);
    EXPECT_EQ(2U, two_d._dimensions);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}