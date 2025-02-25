#include <gtest/gtest.h>
#include <iostream>

class PrintProgressListener : public ::testing::EmptyTestEventListener {
public:
    void OnTestStart(const ::testing::TestInfo& test_info) override {
        std::cout << "Running [" << test_info.test_case_name() << "." << test_info.name() << "]" << std::endl;
    }

    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        std::cout << "Finished [" << test_info.test_case_name() << "." << test_info.name() << "]" << std::endl;
    }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    PrintProgressListener progress_listener;
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(&progress_listener);

    return RUN_ALL_TESTS();
}
