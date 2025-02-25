#pragma once

#include <gtest/gtest.h>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

#include "../src/gpuTypes.h"
#include "../src/types.h"
#include "../src/kernels.cuh"
#include "../src/utils/utils.h"

void randData(float* pTarget, float* pOut, const size_t batch, const size_t nFeatures, const size_t stride) {
    std::memset(pTarget, 0, stride * batch * sizeof(float));
    std::memset(pOut, 0, stride * batch * sizeof(float));
    for (size_t i = 0; i < batch; i++) {
        for (size_t k = 0; k < nFeatures; k++) {
            pTarget[k] = rand(0, nFeatures - 1);
        }
        for (size_t o = 0; o < nFeatures; o++) {
            pOut[o] = rand(0.f, 1.f);
        }
        pTarget += stride;
        pOut += stride;
    }
}

inline double elapsed_seconds(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

bool testTopK(const size_t batch = 128, const size_t topK = 128, const size_t nFeatures = 1024) {
    std::cout << "TEST kCalculateTopK with parameters: batch=" << batch << " topK=" << topK << " nFeatures=" << nFeatures << std::endl;
    bool ret = true;

    const float eps = 1.e-6;
    const size_t stride = ((nFeatures + 127) >> 7) << 7;

    std::unique_ptr<GpuBuffer<float>> pbKey(new GpuBuffer<float>(batch * topK, true));
    std::unique_ptr<GpuBuffer<float>> pbFValue(new GpuBuffer<float>(batch * topK, true));
    std::unique_ptr<GpuBuffer<unsigned int>> pbUIValue(new GpuBuffer<unsigned int>(batch * topK, true));
    std::unique_ptr<GpuBuffer<float>> pbTarget(new GpuBuffer<float>(batch * stride, true));
    std::unique_ptr<GpuBuffer<float>> pbOutput(new GpuBuffer<float>(batch * stride, true));

    std::cout << "1 TEST kCalculateTopK with 3 args" << std::endl;

    {
        float* pTarget = pbTarget->_pSysData;
        float* pOut = pbOutput->_pSysData;

        randData(pTarget, pOut, batch, nFeatures, stride);

        pbTarget->Upload();
        pbOutput->Upload();

        std::memset(pbUIValue->_pSysData, 0, batch * topK * sizeof(unsigned int));
        pbUIValue->Upload();
    }
    {
        auto const start = std::chrono::steady_clock::now();
        kCalculateTopK(pbOutput->_pDevData, pbKey->_pDevData, pbUIValue->_pDevData, batch, stride, topK);
        auto const end = std::chrono::steady_clock::now();
        std::cout << "GPU sort: " << elapsed_seconds(start, end) << std::endl;
    }

    {
        pbOutput->Download();
        pbTarget->Download();
        pbKey->Download();
        pbFValue->Download();
        pbUIValue->Download();

        std::vector<float> keys(nFeatures);
        std::vector<unsigned int> topKvals(topK);
        std::vector<float> topKkeys(topK);

        float* pOutput = pbOutput->_pSysData;
        float* pKey = pbKey->_pSysData;
        unsigned int* pUIValue = pbUIValue->_pSysData;

        int countValueError = 0;
        float sumKeyError = 0.f;
        float cpuSort = 0.f;

        for (size_t i = 0; i < batch; i++) {
            auto const start = std::chrono::steady_clock::now();
            topKsort<float, unsigned int>(pOutput, NULL, nFeatures, &topKkeys[0], &topKvals[0], topK);
            auto const end = std::chrono::steady_clock::now();
            cpuSort += elapsed_seconds(start, end);

            for (size_t k = 0; k < topK; k++) {
                unsigned int GPUvalue = pUIValue[k];
                float GPUkey = pKey[k];

                float CPUvalue = topKvals[k];
                float CPUkey = topKkeys[k];

                if (fabs(GPUvalue - CPUvalue) > eps) {
                    countValueError++;
                }
                sumKeyError += fabs(GPUkey - CPUkey);
            }
            pKey += topK;
            pUIValue += topK;
            pOutput += stride;
        }
        std::cout << "CPU sort: " << cpuSort << std::endl;

        if (countValueError && sumKeyError) {
            std::cout << "1 ERROR kCalculateTopK with 3 args; ";
            ret = false;
        } else {
            std::cout << "1 PASS kCalculateTopK with 3 args; ";
        }
        std::cout << "countValueError " << countValueError << " sumKeyError " << sumKeyError << std::endl;
    }

    std::cout << "2 TEST kCalculateTopK with 4 args" << std::endl;

    {
        float* pTarget = pbTarget->_pSysData;
        float* pOut = pbOutput->_pSysData;

        randData(pTarget, pOut, batch, nFeatures, stride);

        pbTarget->Upload();
        pbOutput->Upload();
    }

    kCalculateTopK(pbOutput->_pDevData, pbTarget->_pDevData, pbKey->_pDevData, pbFValue->_pDevData, batch, stride, topK);

    {
        pbOutput->Download();
        pbTarget->Download();
        pbKey->Download();
        pbFValue->Download();

        std::vector<float> vals(nFeatures);
        std::vector<float> keys(nFeatures);
        std::vector<float> topKvals(topK);
        std::vector<float> topKkeys(topK);

        float* pOutput = pbOutput->_pSysData;
        float* pTarget = pbTarget->_pSysData;
        float* pKey = pbKey->_pSysData;
        float* pValue = pbFValue->_pSysData;

        int countValueError = 0;
        float sumKeyError = 0;

        for (size_t i = 0; i < batch; i++) {
            topKsort<float, float>(pOutput, pTarget, nFeatures, &topKkeys[0], &topKvals[0], topK);

            for (size_t k = 0; k < topK; k++) {
                unsigned int GPUvalue = pValue[k];
                float GPUkey = pKey[k];

                float CPUvalue = topKvals[k];
                float CPUkey = topKkeys[k];

                if (fabs(GPUvalue - CPUvalue) > eps) {
                    countValueError++;
                }
                sumKeyError += fabs(GPUkey - CPUkey);
            }
            pKey += topK;
            pValue += topK;
            pOutput += stride;
            pTarget += stride;
        }

        if (countValueError && sumKeyError) {
            std::cout << "2 ERROR kCalculateTopK with 4 args; ";
            ret = false;
        } else {
            std::cout << "2 PASS kCalculateTopK with 4 args; ";
        }
        std::cout << "countValueError " << countValueError << " sumKeyError " << sumKeyError << std::endl;
    }

    int totalGPUMemory;
    int totalCPUMemory;
    getGpu().GetMemoryUsage(&totalGPUMemory, &totalCPUMemory);
    std::cout << "GPU Memory Usage: " << totalGPUMemory << " KB" << std::endl;
    std::cout << "CPU Memory Usage: " << totalCPUMemory << " KB" << std::endl;

    return ret;
}

class TestSort : public ::testing::Test {
protected:
    void SetUp() override {
        getGpu().SetRandomSeed(12345);
        getGpu().CopyConstants();
    }
};

TEST_F(TestSort, CPU_GPUSort) {
    const size_t numberTests = 5;
    const size_t batches[numberTests] = {128, 128, 128, 128, 128};
    const size_t topK[numberTests] = {128, 128, 64, 32, 1};
    const size_t numberFeatures[numberTests] = {1024, 100000, 1024, 64, 64};

    for (size_t i = 0; i < numberTests; i++) {
        bool result = testTopK(batches[i], topK[i], numberFeatures[i]);
        std::cout << "batches " << batches[i] << ", topK " << topK[i] << ", numberFeatures " << numberFeatures[i] << std::endl;
        EXPECT_TRUE(result) << "Failed gpuSort for batches " << batches[i] << ", topK " << topK[i] << ", numberFeatures " << numberFeatures[i];
    }
}