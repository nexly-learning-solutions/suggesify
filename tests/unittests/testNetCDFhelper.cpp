#include <map>
#include <string>
#include <sstream>
#include <unordered_map>

#include <gtest/gtest.h>
#include "../src/utils/netCDFhelper.h"

class TestNetCDFhelper : public ::testing::Test {
protected:
    static const std::map<std::string, unsigned int> validFeatureIndex;

    void SetUp() override {
    }

    void TearDown() override {
    }
};

const std::map<std::string, unsigned int> TestNetCDFhelper::validFeatureIndex = {
    { "110510", 26743 },
    { "121019", 26740 },
    { "121017", 26739 },
    { "106401", 26736 },
    { "104307", 26734 }
};

TEST_F(TestNetCDFhelper, LoadIndexWithValidInput) {
    std::stringstream inputStream;
    for (const auto &entry : validFeatureIndex) {
        inputStream << entry.first << "\t" << entry.second << "\n";
    }

    std::unordered_map<std::string, unsigned int> labelsToIndices;
    std::stringstream outputStream;
    EXPECT_TRUE(loadIndex(labelsToIndices, inputStream, outputStream));
    EXPECT_EQ(outputStream.str().find("Error"), std::string::npos);
    EXPECT_EQ(validFeatureIndex.size(), labelsToIndices.size());

    for (const auto &entry : validFeatureIndex) {
        const auto itr = labelsToIndices.find(entry.first);
        EXPECT_NE(itr, labelsToIndices.end());
        EXPECT_EQ(entry.second, itr->second);
    }
}

TEST_F(TestNetCDFhelper, LoadIndexWithDuplicateEntry) {
    std::stringstream inputStream;
    for (const auto &entry : validFeatureIndex) {
        inputStream << entry.first << "\t" << entry.second << "\n";
    }

    const auto itr = validFeatureIndex.begin();
    inputStream << itr->first << "\t" << itr->second << "\n";

    std::unordered_map<std::string, unsigned int> labelsToIndices;
    std::stringstream outputStream;
    EXPECT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
    EXPECT_NE(outputStream.str().find("Error"), std::string::npos);
}

TEST_F(TestNetCDFhelper, LoadIndexWithDuplicateLabelOnly) {
    std::stringstream inputStream;
    for (const auto &entry : validFeatureIndex) {
        inputStream << entry.first << "\t" << entry.second << "\n";
    }

    inputStream << validFeatureIndex.begin()->first << "\t123\n";

    std::unordered_map<std::string, unsigned int> labelsToIndices;
    std::stringstream outputStream;
    EXPECT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
    EXPECT_NE(outputStream.str().find("Error"), std::string::npos);
}

TEST_F(TestNetCDFhelper, LoadIndexWithMissingLabel) {
    std::stringstream inputStream;
    inputStream << "\t123\n";
    std::unordered_map<std::string, unsigned int> labelsToIndices;
    std::stringstream outputStream;
    EXPECT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
    EXPECT_NE(outputStream.str().find("Error"), std::string::npos);
}

TEST_F(TestNetCDFhelper, LoadIndexWithMissingLabelAndTab) {
    std::stringstream inputStream;
    inputStream << "123\n";
    std::unordered_map<std::string, unsigned int> labelsToIndices;
    std::stringstream outputStream;
    EXPECT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
    EXPECT_NE(outputStream.str().find("Error"), std::string::npos);
}

TEST_F(TestNetCDFhelper, LoadIndexWithExtraTab) {
    std::stringstream inputStream;
    inputStream << "110510\t123\t121017\n";
    std::unordered_map<std::string, unsigned int> labelsToIndices;
    std::stringstream outputStream;
    EXPECT_FALSE(loadIndex(labelsToIndices, inputStream, outputStream));
    EXPECT_NE(outputStream.str().find("Error"), std::string::npos);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}