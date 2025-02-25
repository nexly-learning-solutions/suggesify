#include "gpuTypes.h"
#include "types.h"
#include <limits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>

typedef half2 half2_t;
typedef float2 float2_t;

typedef half (*ActivationFunctionHalf)(half);
typedef float (*ActivationFunctionFloat)(float);

struct ActivationFunction {
    ActivationFunctionHalf halfFunction;
    ActivationFunctionFloat floatFunction;
};

enum ActivationType {
    SIGMOID, TANH, RELU, LRELU, ELU, SELU, SOFTMAX, SWISH, MISH, GELU, 
    BRELU,
    HARD_SIGMOID,
    LINEAR,
    SOFTSIGN,
    EXP,
    LOG,
    SIN,
    COS,
    TAN,
    SINH,
    COSH,
    TANH_SCALED,
    PRELU, 
    GELU_APPROX_1, 
    GELU_APPROX_2, 
    SELU_VARIANT_1,
    SELU_VARIANT_2, 
    SWISH_VARIANT_1,
    SWISH_VARIANT_2, 
    MISH_VARIANT_1,
    MISH_VARIANT_2, 
};

struct ActivationParams {
    ActivationType type;
    float slope;
    float alpha;
    float lambda;
    float beta;
    float scale;
    float a;
    float b;
    float c;
    float d;
};

struct ActivationConfig {
    ActivationParams params;
    bool useSharedMemory;
    int sharedMemorySize;
    int threadsPerBlock;
    int blocksPerGrid;
    cudaStream_t stream;
    bool useDoublePrecision;
};

union DeviceData {
    struct {
        ActivationParams activationParams;
        ActivationConfig activationConfig;
        GpuData gpuData;
        ActivationFunction activationFunction;
    } data;
    char rawData[sizeof(ActivationParams) + sizeof(ActivationConfig) + sizeof(GpuData) + sizeof(ActivationFunction)];
};

static __constant__ DeviceData cDeviceData;

template <typename T>
__device__ inline T atomicMax(T* address, T val) {
    if constexpr (std::is_same_v<T, float>) {
        unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(address);
        unsigned int old = *address_as_ui, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_ui, assumed,
                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    } else if constexpr (std::is_same_v<T, half>) {
        unsigned short* address_as_us = reinterpret_cast<unsigned short*>(address);
        unsigned short old = *address_as_us, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_us, assumed,
                __float2half_rn(::fmaxf(__half2float(assumed), __half2float(val))));
        } while (assumed != old);
        return __half2float(old);
    } else if constexpr (std::is_same_v<T, half2_t>) {
        half2_t old = *address, assumed;
        do {
            assumed = old;
            old.x = atomicMax(&address->x, val.x);
            old.y = atomicMax(&address->y, val.y);
        } while (assumed != old);
        return old;
    } else if constexpr (std::is_same_v<T, float2_t>) {
        float2_t old = *address, assumed;
        do {
            assumed = old;
            old.x = atomicMax(&address->x, val.x);
            old.y = atomicMax(&address->y, val.y);
        } while (assumed != old);
        return old;
    } else {
        static_assert(false, "Unsupported data type for atomicMax.");
    }
}

inline ActivationFunction getActivationFunction(ActivationType type) {
    ActivationFunction activationFunction;
    switch (type) {
        case SIGMOID:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(1.0f / (1.0f + expf(__half2float(-x)))); };
            activationFunction.floatFunction = [](float x) { return 1.0f / (1.0f + expf(-x)); };
            break;
        case TANH:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(tanhf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return tanhf(x); };
            break;
        case RELU:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(fmaxf(0.0f, __half2float(x))); };
            activationFunction.floatFunction = [](float x) { return fmaxf(0.0f, x); };
            break;
        case LRELU:
            activationFunction.halfFunction = [](half x) {
                return __float2half_rn(fmaxf(__half2float(x), __half2float(x) * cDeviceData.data.activationParams.slope));
            };
            activationFunction.floatFunction = [](float x) { return fmaxf(x, x * cDeviceData.data.activationParams.slope); };
            break;
        case ELU:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn((val > 0.0f) ? val : cDeviceData.data.activationParams.alpha * (expf(val) - 1.0f));
            };
            activationFunction.floatFunction = [](float x) { return (x > 0.0f) ? x : cDeviceData.data.activationParams.alpha * (expf(x) - 1.0f); };
            break;
        case SELU:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn((val > 0.0f) ? cDeviceData.data.activationParams.lambda * val : cDeviceData.data.activationParams.lambda * cDeviceData.data.activationParams.alpha * (expf(val) - 1.0f));
            };
            activationFunction.floatFunction = [](float x) { return (x > 0.0f) ? cDeviceData.data.activationParams.lambda * x : cDeviceData.data.activationParams.lambda * cDeviceData.data.activationParams.alpha * (expf(x) - 1.0f); };
            break;
        case SWISH:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(val * (1.0f / (1.0f + expf(-cDeviceData.data.activationParams.beta * val))));
            };
            activationFunction.floatFunction = [](float x) { return x * (1.0f / (1.0f + expf(-cDeviceData.data.activationParams.beta * x))); };
            break;
        case MISH:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(val * tanhf(logf(1.0f + expf(val)) * cDeviceData.data.activationParams.beta));
            };
            activationFunction.floatFunction = [](float x) { return x * tanhf(logf(1.0f + expf(x)) * cDeviceData.data.activationParams.beta); };
            break;
        case GELU:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(0.5f * val * (1.0f + erf(val / sqrtf(2.0f))));
            };
            activationFunction.floatFunction = [](float x) { return 0.5f * x * (1.0f + erf(x / sqrtf(2.0f))); };
            break;
        case BRELU:
            activationFunction.halfFunction = [](half x) { return __float2half_rn((__half2float(x) > 0.0f) ? 1.0f : 0.0f); };
            activationFunction.floatFunction = [](float x) { return (x > 0.0f) ? 1.0f : 0.0f; };
            break;
        case HARD_SIGMOID:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(fmaxf(0.0f, fminf(1.0f, __half2float(x) * 0.2f + 0.5f))); };
            activationFunction.floatFunction = [](float x) { return fmaxf(0.0f, fminf(1.0f, x * 0.2f + 0.5f)); };
            break;
        case LINEAR:
            activationFunction.halfFunction = [](half x) { return x; };
            activationFunction.floatFunction = [](float x) { return x; };
            break;
        case SOFTSIGN:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(x / (1.0f + fabsf(__half2float(x)))); };
            activationFunction.floatFunction = [](float x) { return x / (1.0f + fabsf(x)); };
            break;
        case EXP:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(expf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return expf(x); };
            break;
        case LOG:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(logf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return logf(x); };
            break;
        case SIN:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(sinf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return sinf(x); };
            break;
        case COS:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(cosf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return cosf(x); };
            break;
        case TAN:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(tanf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return tanf(x); };
            break;
        case SINH:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(sinhf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return sinhf(x); };
            break;
        case COSH:
            activationFunction.halfFunction = [](half x) { return __float2half_rn(coshf(__half2float(x))); };
            activationFunction.floatFunction = [](float x) { return coshf(x); };
            break;
        case TANH_SCALED:
            activationFunction.halfFunction = [](half x) {
                return __float2half_rn(cDeviceData.data.activationParams.scale * tanhf(__half2float(x)));
            };
            activationFunction.floatFunction = [](float x) {
                return cDeviceData.data.activationParams.scale * tanhf(x);
            };
            break;
        case PRELU:
            activationFunction.halfFunction = [](half x) {
                return __float2half_rn((__half2float(x) > 0.0f) ? __half2float(x) : cDeviceData.data.activationParams.a * __half2float(x));
            };
            activationFunction.floatFunction = [](float x) { return (x > 0.0f) ? x : cDeviceData.data.activationParams.a * x; };
            break;
        case GELU_APPROX_1:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(0.5f * val * (1.0f + tanhf(0.7978845608028654 * val * (1.0f + 0.044715 * val * val))));
            };
            activationFunction.floatFunction = [](float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654 * x * (1.0f + 0.044715 * x * x))); };
            break;
        case GELU_APPROX_2:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(val * 0.5f * (1.0f + erf(val / sqrtf(2.0f))));
            };
            activationFunction.floatFunction = [](float x) { return x * 0.5f * (1.0f + erf(x / sqrtf(2.0f))); };
            break;
        case SELU_VARIANT_1:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn((val > 0.0f) ? cDeviceData.data.activationParams.b * val : cDeviceData.data.activationParams.b * cDeviceData.data.activationParams.alpha * (expf(val) - 1.0f));
            };
            activationFunction.floatFunction = [](float x) { return (x > 0.0f) ? cDeviceData.data.activationParams.b * x : cDeviceData.data.activationParams.b * cDeviceData.data.activationParams.alpha * (expf(x) - 1.0f); };
            break;
        case SELU_VARIANT_2:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn((val > 0.0f) ? cDeviceData.data.activationParams.b * val : cDeviceData.data.activationParams.b * cDeviceData.data.activationParams.alpha * (expf(val) - 1.0f));
            };
            activationFunction.floatFunction = [](float x) { return (x > 0.0f) ? cDeviceData.data.activationParams.b * x : cDeviceData.data.activationParams.b * cDeviceData.data.activationParams.alpha * (expf(x) - 1.0f); };
            break;
        case SWISH_VARIANT_1:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(val * (1.0f / (1.0f + expf(-cDeviceData.data.activationParams.c * val))));
            };
            activationFunction.floatFunction = [](float x) { return x * (1.0f / (1.0f + expf(-cDeviceData.data.activationParams.c * x))); };
            break;
        case SWISH_VARIANT_2:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(val * (1.0f / (1.0f + expf(-cDeviceData.data.activationParams.c * val))));
            };
            activationFunction.floatFunction = [](float x) { return x * (1.0f / (1.0f + expf(-cDeviceData.data.activationParams.c * x))); };
            break;
        case MISH_VARIANT_1:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(val * tanhf(logf(1.0f + expf(val)) * cDeviceData.data.activationParams.d));
            };
            activationFunction.floatFunction = [](float x) { return x * tanhf(logf(1.0f + expf(x)) * cDeviceData.data.activationParams.d); };
            break;
        case MISH_VARIANT_2:
            activationFunction.halfFunction = [](half x) {
                float val = __half2float(x);
                return __float2half_rn(val * tanhf(logf(1.0f + expf(val)) * cDeviceData.data.activationParams.d));
            };
            activationFunction.floatFunction = [](float x) { return x * tanhf(logf(1.0f + expf(x)) * cDeviceData.data.activationParams.d); };
            break;
        default:
            break;
    }
    return activationFunction;
}

void SetDeviceData(const ActivationParams& params, const ActivationConfig& config) {
    DeviceData deviceData;
    memcpy(&deviceData.data.activationParams, ¶ms, sizeof(ActivationParams));
    memcpy(&deviceData.data.activationConfig, &config, sizeof(ActivationConfig));
    memcpy(&deviceData.data.gpuData, &getGpu()._data, sizeof(GpuData));
    deviceData.data.activationFunction = getActivationFunction(params.type);

    cudaError_t status = cudaMemcpyToSymbolAsync(cDeviceData, &deviceData, sizeof(DeviceData), cudaStreamDefault);
    if (status != cudaSuccess) {
        fprintf(stderr, "Error copying data to device: %s\n", cudaGetErrorString(status));
    }
}

void GetDeviceData() {
    DeviceData deviceData;
    cudaError_t status = cudaMemcpyFromSymbolAsync(&deviceData, cDeviceData, sizeof(DeviceData), cudaStreamDefault);
    if (status != cudaSuccess) {
        fprintf(stderr, "Error copying data from device: %s\n", cudaGetErrorString(status));
    }
    getGpu()._data = deviceData.data.gpuData;
}

template <typename T>
__global__ void
__launch_bounds__(256, 4)
invokeActivation_kernel(T* pData, uint64_t size) {
    uint64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < size) {
        if constexpr (std::is_same_v<T, half>) {
            pData[pos] = cDeviceData.data.activationFunction.halfFunction(pData[pos]);
        } else if constexpr (std::is_same_v<T, float>) {
            pData[pos] = cDeviceData.data.activationFunction.floatFunction(pData[pos]);
        }
    }
}

template <typename T>
__global__ void
__launch_bounds__(256, 4)
invokeSoftMaxActivation_kernel(T* pData, uint32_t stride) {
    __shared__ unsigned long long int sAccumulator;
    __shared__ T sMaxValue;

    if (threadIdx.x == 0) {
        sAccumulator = 0;
        sMaxValue = -std::numeric_limits<T>::infinity();
    }
    __syncthreads();

    pData += blockIdx.x * stride;
    uint32_t pos = threadIdx.x;
    T maxValue = -std::numeric_limits<T>::infinity();

    while (pos < stride) {
        maxValue = fmaxf(maxValue, pData[pos]);
        pos += blockDim.x;
    }

    uint32_t tgx = threadIdx.x & cDeviceData.data.gpuData._warpMask;
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 1));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 2));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 4));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 8));
    maxValue = fmaxf(maxValue, __shfl_sync(0xFFFFFFFF, maxValue, tgx ^ 16));

    if (tgx == 0)
        atomicMax(&sMaxValue, maxValue);
    __syncthreads();
    maxValue = sMaxValue;

    pos = threadIdx.x;
    float sum = 0.0f;
    while (pos < stride) {
        sum += expf(pData[pos] - maxValue);
        pos += blockDim.x;
    }

    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 1);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 2);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 4);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 8);
    sum += __shfl_sync(0xFFFFFFFF, sum, tgx ^ 16);

    unsigned long long int lsum = llitoulli(llrintf(ERRORSCALEF * sum));
    if (tgx == 0)
        atomicAdd(&sAccumulator, lsum);
    __syncthreads();

    float norm = 1.0f / (float)((double)sAccumulator * ONEOVERERRORSCALE);

    pos = threadIdx.x;
    while (pos < stride) {
        pData[pos] = __float2half_rn(fminf(1.0f, expf(pData[pos] - maxValue) * norm));
        pos += blockDim.x;
    }
}

template <typename T>
void invokeActivation(T* pData, uint64_t size, const ActivationConfig& config) {
    SetDeviceData(config.params, config);

    int threadsPerBlock = min(config.threadsPerBlock, static_cast<int>(size));
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    if (config.params.type == SOFTMAX) {
        cudaLaunchKernel(reinterpret_cast<void*>(invokeSoftMaxActivation_kernel<T>),
            dim3(blocksPerGrid), dim3(threadsPerBlock), nullptr, 0, config.stream);
    } else {
        cudaLaunchKernel(reinterpret_cast<void*>(invokeActivation_kernel<T>),
            dim3(blocksPerGrid), dim3(threadsPerBlock), nullptr, 0, config.stream);
    }
}

template <typename T>
void benchmarkActivation(const std::vector<T>& data, const ActivationConfig& config, const std::string& activationName) {
    T* d_data;
    cudaMalloc(&d_data, data.size() * sizeof(T));

    cudaMemcpy(d_data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    invokeActivation<T>(d_data, data.size(), config);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << activationName << " execution time: " << duration.count() << " seconds" << std::endl;

    cudaFree(d_data);
}

template <typename T>
void applyActivation(std::vector<T>& data, const ActivationConfig& config, const std::string& activationName) {
    T* d_data;
    cudaMalloc(&d_data, data.size() * sizeof(T));

    cudaMemcpy(d_data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);

    invokeActivation<T>(d_data, data.size(), config);

    cudaDeviceSynchronize();

    cudaMemcpy(data.data(), d_data, data.size() * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}

std::string activationTypeToString(ActivationType type) {
    switch (type) {
        case SIGMOID:
            return "Sigmoid";
        case TANH:
            return "Tanh";
        case RELU:
            return "ReLU";
        case LRELU:
            return "LeakyReLU";
        case ELU:
            return "ELU";
        case SELU:
            return "SELU";
        case SOFTMAX:
            return "Softmax";
        case SWISH:
            return "Swish";
        case MISH:
            return "Mish";
        case GELU:
            return "GELU";
        case BRELU:
            return "BinaryReLU";
        case HARD_SIGMOID:
            return "HardSigmoid";
        case LINEAR:
            return "Linear";
        case SOFTSIGN:
            return "SoftSign";
        case EXP:
            return "Exp";
        case LOG:
            return "Log";
        case SIN:
            return "Sin";
        case COS:
            return "Cos";
        case TAN:
            return "Tan";
        case SINH:
            return "Sinh";
        case COSH:
            return "Cosh";
        case TANH_SCALED:
            return "ScaledTanh";
        case PRELU:
            return "PReLU";
        case GELU_APPROX_1:
            return "GELU_Approx_1";
        case GELU_APPROX_2:
            return "GELU_Approx_2";
        case SELU_VARIANT_1:
            return "SELU_Variant_1";
        case SELU_VARIANT_2:
            return "SELU_Variant_2";
        case SWISH_VARIANT_1:
            return "Swish_Variant_1";
        case SWISH_VARIANT_2:
            return "Swish_Variant_2";
        case MISH_VARIANT_1:
            return "Mish_Variant_1";
        case MISH_VARIANT_2:
            return "Mish_Variant_2";
        default:
            return "Unknown";
    }
}

template <typename T>
std::vector<T> generateRandomData(size_t size, T minVal = -1.0f, T maxVal = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> distrib(minVal, maxVal);
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&distrib, &gen]() { return distrib(gen); });
    return data;
}

std::vector<float> readDataFromCSV(const std::string& filename) {
    std::vector<float> data;
    std::ifstream inputFile(filename);

    if (inputFile.is_open()) {
        std::string line;
        std::getline(inputFile, line);

        while (std::getline(inputFile, line)) {
            std::istringstream iss(line);
            std::string value;
            std::getline(iss, value, ',');
            std::getline(iss, value, ',');
            data.push_back(std::stof(value));
        }

        inputFile.close();
        std::cout << "Data read from: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open input file: " << filename << std::endl;
    }

    return data;
}

std::vector<float> transformData(const std::vector<float>& data) {
    std::vector<float> transformedData(data.size());
    std::transform(data.begin(), data.end(), transformedData.begin(), [](float value) {
        return value * value;
    });
    return transformedData;
}

void saveDataToCSV(const std::vector<float>& data, const std::string& filename) {
    std::ofstream outputFile(filename);

    if (outputFile.is_open()) {
        outputFile << "Index,Value" << std::endl;

        for (size_t i = 0; i < data.size(); ++i) {
            outputFile << i << "," << data[i] << std::endl;
        }

        outputFile.close();
        std::cout << "Data saved to: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open output file: " << filename << std::endl;
    }
}

std::vector<float> complexTransformData(const std::vector<float>& data, int degree) {
    std::vector<float> transformedData(data.size());
    std::transform(data.begin(), data.end(), transformedData.begin(), [degree](float value) {
        float result = 0.0f;
        for (int i = 0; i <= degree; ++i) {
            result += pow(value, i);
        }
        return result;
    });
    return transformedData;
}