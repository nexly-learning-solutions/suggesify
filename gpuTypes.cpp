#include "GpuTypes.h"
#include "Types.h"
#include "Kernels.cuh"
#include <omp.h>
#include <mpi.h>
#include "Constants.h"
#include "ThreadPool.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <stdexcept>
#include <latch>
#include <ranges>
#include <random>

#ifdef USE_HIGHWAY
#include <highway/highway.h>
using namespace highway;
#endif

/// <summary>
/// Prime number for hash function A.
/// </summary>
inline constexpr unsigned long PRIME_A = 2654435761ul;

/// <summary>
/// Prime number for hash function B.
/// </summary>
inline constexpr unsigned long PRIME_B = 63689ul;

/// <summary>
/// Prime number for hash function C.
/// </summary>
inline constexpr unsigned long PRIME_C = 378551ul;

/// <summary>
/// Prime number for hash function D.
/// </summary>
inline constexpr unsigned long PRIME_D = 6367ul;

/// <summary>
/// XOR mask for hash function A.
/// </summary>
inline constexpr unsigned long XOR_MASK_A = 0x5A17A17Aul;

/// <summary>
/// XOR mask for hash function B.
/// </summary>
inline constexpr unsigned long XOR_MASK_B = 0xC3A5C3A5ul;

/// <summary>
/// XOR mask for hash function C.
/// </summary>
inline constexpr unsigned long XOR_MASK_C = 0x81958195ul;

/// <summary>
/// Number of bits to shift for hash function.
/// </summary>
inline constexpr unsigned long SHIFT_BITS = 7;

/// <summary>
/// Mixes an integer value using a hash function.
/// </summary>
/// <typeparam name="T">The type of the integer value.</typeparam>
/// <param name="x">The integer value to mix.</param>
/// <returns>The mixed integer value.</returns>
template<std::integral T>
T Mix(T x) {
    x ^= (x >> 17);
    x += 0xABCD1234u;
    x ^= (x << 9);
    x ^= (x >> 27);
    return x;
}

/// <summary>
/// Acceptable error for floating point comparisons.
/// </summary>
static const float cAcceptableError = 0.00001f;

/// <summary>
/// Global GPU context.
/// </summary>
static GpuContext gpu;

/// <summary>
/// Gets the global GPU context.
/// </summary>
/// <returns>The global GPU context.</returns>
GpuContext& getGpu() { return gpu; }

/// <summary>
/// Calculates the number of leading zeros in an integer value.
/// </summary>
/// <param name="x">The integer value to calculate the leading zeros for.</param>
/// <returns>The number of leading zeros.</returns>
#if defined(_MSC_VER)

static __forceinline int fls(int x)
{
    if (x == 0) return 0;
    unsigned long index;
    _BitScanReverse(&index, static_cast<unsigned long>(x));
    return static_cast<int>(index) + 1;
}

#elif defined(__GNUC__)

static __inline int fls(int x)
{
    return x ? sizeof(x) * 8 - __builtin_clz(x) : 0;
}

#else
#error Unsupported compiler
#endif

/// <summary>
/// Represents the GPU context.
/// </summary>
GpuContext::GpuContext() :
    _bECCSupport(false),
    _bCanMapHostMemory(false),
    _bCPUValidate(false),
    _bUnifiedMemory(false),
    _acceptableError(cAcceptableError),
    _numprocs(1),
    _id(0),
    _sm_version(SM_3X),
    _sm_major(0),
    _warpSize(32),
    _maxSparse(SM_3X_MAXSPARSE),
    _maxSparseAnalog(SM_3X_MAXSPARSEANALOG),
    _cuBLASHandle(0),
    _cuDNNHandle(0),
    _pbAccumulator()
{
    std::cout << "Initializing GpuContext...\n";
    std::cout << "_bECCSupport: " << _bECCSupport << "\n";
    std::cout << "_bCanMapHostMemory: " << _bCanMapHostMemory << "\n";
    std::cout << "_bCPUValidate: " << _bCPUValidate << "\n";
    std::cout << "_bUnifiedMemory: " << _bUnifiedMemory << "\n";
    std::cout << "_acceptableError: " << _acceptableError << "\n";
    std::cout << "_numprocs: " << _numprocs << "\n";
    std::cout << "_id: " << _id << "\n";
    std::cout << "_sm_version: " << _sm_version << "\n";
    std::cout << "_sm_major: " << _sm_major << "\n";
    std::cout << "_warpSize: " << _warpSize << "\n";
    std::cout << "_maxSparse: " << _maxSparse << "\n";
    std::cout << "_maxSparseAnalog: " << _maxSparseAnalog << "\n";
    std::cout << "_cuBLASHandle: " << _cuBLASHandle << "\n";
    std::cout << "_cuDNNHandle: " << _cuDNNHandle << "\n";
    std::cout << "_pbAccumulator: {}\n";
    std::cout << "GpuContext initialized.";
}

/// <summary>
/// Destructor for the GPU context.
/// </summary>
GpuContext::~GpuContext()
{

}

/// <summary>
/// Sets whether CPU validation is enabled.
/// </summary>
/// <param name="bValidate">True to enable CPU validation, false to disable it.</param>
void GpuContext::SetCPUValidate(bool bValidate)
{
    _bCPUValidate = bValidate;
}

/// <summary>
/// Initializes the GPU context.
/// </summary>
/// <param name="argc">Number of command-line arguments.</param>
/// <param name="argv">Array of command-line arguments.</param>
void GpuContext::Startup(int argc, char** argv)
{

    // Initialize MPI if it's not already initialized.
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init(&argc, &argv);
    }

    // Get the number of processes and the rank of this process.
    MPI_Comm_size(MPI_COMM_WORLD, &_numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &_id);

    std::cout << "GpuContext::Startup: Process " << _id << " out of " << _numprocs << " initialized.";

    // Set up CUDA profiling if environment variables are set.
    char* cudaProfile = nullptr;
#ifdef _WIN32
    if (_dupenv_s(&cudaProfile, nullptr, "CUDA_PROFILE") == 0 && cudaProfile != nullptr) {
#else
    cudaProfile = getenv("CUDA_PROFILE");
    if (cudaProfile != nullptr) {
#endif
        char profile_log[512];
        char* cudaProfileLog = nullptr;
#ifdef _WIN32
        if (_dupenv_s(&cudaProfileLog, nullptr, "CUDA_PROFILE_LOG") == 0 && cudaProfileLog != nullptr) {
#else
        cudaProfileLog = getenv("CUDA_PROFILE_LOG");
        if (cudaProfileLog != nullptr) {
#endif
            snprintf(profile_log, sizeof(profile_log), "%s%d", cudaProfileLog, _id);
#ifdef _WIN32
            free((void*)cudaProfileLog);
#else
            free((void*)const_cast<char*>(cudaProfileLog));
#endif
        }
        else {
            snprintf(profile_log, sizeof(profile_log), "cu%d.csv", _id);
        }

#ifdef _WIN32
        _putenv_s("CUDA_PROFILE_LOG", profile_log);
#else
        setenv("CUDA_PROFILE_LOG", profile_log, 1);
        setenv("CUDA_PROFILE_CSV", "1", 1);
#endif

#ifdef _WIN32
        free(cudaProfile);
#else
        free(cudaProfile);
#endif
    }

    // Initialize device variables.
    int device = -1;
    int gpuCount = 0;
    cudaError_t status;
    cudaDeviceProp deviceProp;

    // Get the number of CUDA-capable devices.
    status = cudaGetDeviceCount(&gpuCount);

    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed with status: " << cudaGetErrorString(status);
    }

    // Exit if no CUDA-capable devices are found.
    if (gpuCount == 0) {
        std::cerr << "GpuContext::Startup: No CUDA-capable devices found, exiting.";
        cudaDeviceReset();
        Shutdown();
        exit(EXIT_FAILURE);
    }

    // Get MPI information about processes.
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int length;

    char myName[MPI_MAX_PROCESSOR_NAME + 1];

    // Allocate memory for storing process names.
    std::vector<char> pName(world_size * (MPI_MAX_PROCESSOR_NAME + 1));
    std::vector<int> pNameCount(world_size);
    std::vector<int> pNameDisp(world_size);

    // Get the processor name for this process.
    MPI_Get_processor_name(myName, &length);

    // Store the processor name in the allocated memory.
    strcpy_s(&pName[static_cast<std::vector<char, std::allocator<char>>::size_type>(world_rank) * (MPI_MAX_PROCESSOR_NAME + 1)], MPI_MAX_PROCESSOR_NAME + 1, myName);

    // Set up information for MPI_Allgatherv.
    for (int i = 0; i < world_size; i++) {
        pNameCount[i] = MPI_MAX_PROCESSOR_NAME + 1;
        pNameDisp[i] = i * (MPI_MAX_PROCESSOR_NAME + 1);
    }

    // Gather processor names from all processes.
    MPI_Allgatherv(myName, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, pName.data(), pNameCount.data(), pNameDisp.data(),
        MPI_CHAR, MPI_COMM_WORLD);

    // Determine if all processes are on the same node.
    bool bSingleNode = true;
    bool bP2P = false;

    for (int i = 0; i < world_size; i++) {
        if (std::string(&pName[i * (MPI_MAX_PROCESSOR_NAME + 1)]) != myName)
            bSingleNode = false;
    }

    // Enable host memory mapping for CUDA.
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // Count the number of processes on the same node as this process.
    int localCount = 0;
    int offset = 1;

    for (int i = 0; i < world_size; i++) {
        if (!strcmp(&pName[static_cast<std::vector<char, std::allocator<char>>::size_type>(i) * (MPI_MAX_PROCESSOR_NAME + 1)], myName)) {
            localCount++;
            if (i < world_rank)
                offset++;
        }
    }

    // If there are multiple processes on the same node, select a device for this process.
    if (localCount > 1) {
        int pos = 0;
        int device = -1;

        // Find the next available device that meets the requirements.
        while (offset > 0) {
#ifdef _WIN32
#else
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, pos);

            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3)) {
                device = pos;
                offset--;
            }
#endif
            pos++;
            if (pos == gpuCount)
                pos = 0;
        }

        // Get the hostname of the current node.
        char hostname[128]{};

#ifdef _WIN32
#else
        gethostname(hostname, sizeof(hostname) - 1);
#endif

        // Output information about the selected device.
        std::cout << "GpuContext::Startup: Process " << _id << " running on device " << device << " out of " << gpuCount << " GPUs on " << hostname;
    }
    else {
        // Allocate memory for lists of GPUs and their scores.
        std::vector<int> pGPUList(gpuCount);
        std::vector<unsigned int> pGPUScore(gpuCount);
        int gpus = 0;

        // Populate the lists with compatible GPUs.
        for (int i = 0; i < gpuCount; i++) {
            cudaGetDeviceProperties(&deviceProp, i);

            if (deviceProp.canMapHostMemory && (deviceProp.major >= 3)) {
                pGPUList[gpus] = i;
                pGPUScore[gpus] = (static_cast<unsigned long long>(deviceProp.major) << 24) + (deviceProp.totalGlobalMem >> 20);
                gpus++;
            }
        }

        // Sort the GPUs by score in descending order.
        if (gpus > 0) {
            bool done = true;
            do {
                done = true;
                for (int i = 0; i < gpus - 1; i++) {
                    if (pGPUScore[i] < pGPUScore[static_cast<std::vector<uint32_t, std::allocator<uint32_t>>::size_type>(i) + 1]) {
                        done = false;
                        int gpu = pGPUList[i];
                        unsigned int score = pGPUScore[i];
                        pGPUList[i] = pGPUList[static_cast<std::vector<int, std::allocator<int>>::size_type>(i) + 1];
                        pGPUScore[i] = pGPUScore[static_cast<std::vector<uint32_t, std::allocator<uint32_t>>::size_type>(i) + 1];
                        pGPUList[static_cast<std::vector<int, std::allocator<int>>::size_type>(i) + 1] = gpu;
                        pGPUScore[static_cast<std::vector<uint32_t, std::allocator<uint32_t>>::size_type>(i) + 1] = score;
                    }
                }
            } while (!done);
        }

        // Set the valid devices for this process.
        status = cudaSetValidDevices(pGPUList.data(), gpus);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error searching for compatible GPU";
        }

        // Free any allocated memory.
        status = cudaFree(0);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error selecting compatible GPU";
        }

        // Get the currently selected device.
        status = cudaGetDevice(&device);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error fetching current GPU";
        }

        // Exit if no compatible GPU is found.
        if (device == -1) {
            std::cerr << "GpuContext::Startup: No Kepler or later GPU located, exiting.";
            cudaDeviceReset();
            Shutdown();
            exit(EXIT_FAILURE);
        }

        // Set the selected device.
        status = cudaSetDevice(device);

        if (status != cudaSuccess) {
            std::cerr << "GpuContext::Startup: Error setting CUDA device";
        }

        cudaDeviceSynchronize();

        // Initialize the accumulator buffer.
        _pbAccumulator.reset(new GpuBuffer<unsigned long long int>((unsigned int)1, true));
        _data._pAccumulator = _pbAccumulator->_pDevData;

        // Get properties of the selected device.
        cudaGetDeviceProperties(&deviceProp, _device);

        // Set SM version and other properties based on the device properties.
        if (deviceProp.major == 3) {
            _sm_version = SM_3X;
            _threadsPerBlock = SM_3X_THREADS_PER_BLOCK;
            _maxSparse = SM_3X_MAXSPARSE;
            _maxSparseAnalog = SM_3X_MAXSPARSEANALOG;
        }
        else if (deviceProp.major == 5) {
            _sm_version = SM_5X;
            _threadsPerBlock = SM_5X_THREADS_PER_BLOCK;
            _maxSparse = SM_5X_MAXSPARSE;
            _maxSparseAnalog = SM_5X_MAXSPARSEANALOG;
        }
        else {
            _sm_version = SM_6X;
            _threadsPerBlock = SM_6X_THREADS_PER_BLOCK;
            _maxSparse = SM_6X_MAXSPARSE;
            _maxSparseAnalog = SM_6X_MAXSPARSEANALOG;
        }
        _sm_major = deviceProp.major;
        _warpSize = deviceProp.warpSize;
        _warpBits = fls(_warpSize) - 1;
        _warpMask = _warpSize - 1;
        _data._warpSize = _warpSize;
        _data._warpBits = _warpBits;
        _data._warpMask = _warpMask;
        _bUnifiedMemory = (deviceProp.managedMemory != 0);

        _data._maxUint32_t = 0xFFFFFFFF;
        _data._maxInt32_t = 0x7FFFFFFF;
        _data._maxUint64_t = 0xFFFFFFFFFFFFFFFF;
        _data._maxInt64_t = 0x7FFFFFFFFFFFFFFF;

        // Output information about the selected GPU.
        if (getGpu()._id == 0)
            std::cout << "GpuContext::Startup: Enumerating GPUs in use.";

        for (size_t i = 0; i < getGpu()._numprocs; i++) {
            if (static_cast<size_t>(getGpu()._id) == i)
                std::cout << "Process: " << i << ", GPU: " << deviceProp.name << ", running SM " << deviceProp.major << "." << deviceProp.minor;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Output information about the single node flag.
        std::cout << "GpuContext::Startup: Single node flag on GPU for process " << _device << " is " << bSingleNode << "\n";

        // Check for P2P support and enable peer access if possible.
        if (bSingleNode) {
            bP2P = true;
            std::vector<int> pDevice(_numprocs);
            pDevice[_id] = device;

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pDevice.data(), 1, MPI_INT, MPI_COMM_WORLD);

            std::vector<int> pUnifiedAddressing(_numprocs);
            cudaGetDeviceProperties(&deviceProp, device);
            pUnifiedAddressing[_id] = deviceProp.unifiedAddressing;

            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pUnifiedAddressing.data(), 1, MPI_INT, MPI_COMM_WORLD);

            for (int i = 0; i < _numprocs; i++) {
                if (pDevice[i] != device) {
                    int canAccessPeer;
                    std::cout << "GpuContext::Startup: Testing P2P for processes " << device << " and " << pDevice[i];
                    cudaError_t status = cudaDeviceCanAccessPeer(&canAccessPeer, device, pDevice[i]);

                    if (status != cudaSuccess) {
                        std::cerr << "cudaDeviceCanAccessPeer";
                    }

                    if (canAccessPeer == 0) {
                        bP2P = false;
                    }
                    else {
                        status = cudaDeviceEnablePeerAccess(pDevice[i], 0);

                        if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
                            std::cerr << "cudaDeviceEnablePeerAccess";
                        }
                        else if (status == cudaErrorPeerAccessAlreadyEnabled) {
                            cudaGetLastError();
                        }
                    }
                }
                if (!pUnifiedAddressing[i])
                    bSingleNode = false;
            }
        }

        // Set P2P and single node flags.
        _bSingleNode = bSingleNode;
        _bP2P = bP2P;

        // Output information about P2P support flags.
        std::cout << "GpuContext::Startup: P2P support flags on GPU for process " << _device << " are " << _bP2P << " " << _bSingleNode;

        // Check if all GPUs support P2P communication.
        MPI_Allreduce(MPI_IN_PLACE, &_bP2P, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (!_bP2P) {
            if (_id == 0)
                std::cout << "GpuContext::Startup: Not all GPUs can P2P between each other, falling back to MPI.";
        }

        // Check if all GPUs are on the same node.
        MPI_Allreduce(MPI_IN_PLACE, &_bSingleNode, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (!_bSingleNode) {
            if (_id == 0)
                std::cout << "GpuContext::Startup: P2P support only works within a single node, falling back to MPI.";
        }

        // Get device properties and check for ECC support.
        cudaGetDeviceProperties(&deviceProp, device);
        _bECCSupport = deviceProp.ECCEnabled || deviceProp.tccDriver;

        // Check if the device name contains "tesla" (indicates ECC support).
        std::string deviceNameLower = deviceProp.name;
        std::transform(deviceNameLower.begin(), deviceNameLower.end(), deviceNameLower.begin(), ::tolower);

        if (deviceNameLower.find("tesla") != std::string::npos) {
            _bECCSupport = true;
        }

        // Check if the device supports host memory mapping.
        _bCanMapHostMemory = deviceProp.canMapHostMemory;

#ifdef GVERBOSE
        // Output detailed information about the selected GPU.
        double memsize = (double)deviceProp.totalGlobalMem / (1024.0 * 1024.0);
        std::cout << "GpuContext::Startup: Using GPU " << device << ", " << deviceProp.name << ", SM " << deviceProp.major << "." << deviceProp.minor << ", " << memsize << " MBytes of memory";
#endif

        // Initialize cuBLAS.
        cublasStatus_t cstatus = cublasCreate(&_cuBLASHandle);

        if (cstatus != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "GpuContext::Startup: Failed to initialize cuBLAS on GPU for process " << _device << ", exiting.";
            Shutdown();
            std::exit(EXIT_FAILURE);
        }

        // Initialize cuDNN.
        cudnnStatus_t cdstatus = cudnnCreate(&_cuDNNHandle);

        if (cdstatus != CUDNN_STATUS_SUCCESS) {
            std::cerr << "GpuContext::Startup: Failed to initialize cuDNN on GPU for process " << _device << ", exiting.";
            Shutdown();
            std::exit(EXIT_FAILURE);
        }

        // Initialize cuRand.
        curandStatus_t crstatus = curandCreateGenerator(&_RNG, CURAND_RNG_PSEUDO_DEFAULT);

        if (crstatus != CURAND_STATUS_SUCCESS) {
            std::cerr << "GpuContext::Startup: Failed to initialize cuRand on GPU for process " << _device << ", exiting.";
            Shutdown();
            std::exit(EXIT_FAILURE);
        }

        // Output message indicating the GPU initialization is complete.
        std::cout << "GpuContext::Startup: GPU for process " << device << " initialized.";
    }
}

/// <summary>
/// Copies constants from the Network object to the GpuData structure.
/// </summary>
void GpuContext::CopyConstants() {
    /// <summary>
    /// Represents a lambda function for setting GPU data.
    /// </summary>
    struct GpuDataLambda {
        /// <summary>
        /// Function pointer to the method that sets GPU data.
        /// </summary>
        std::function<void(GpuContext&, const GpuData&)> setGpuData;
        /// <summary>
        /// Name of the GPU data being set.
        /// </summary>
        const char* name;
    };

    /// <summary>
    /// List of lambda functions for setting various GPU data.
    /// </summary>
    std::vector<GpuDataLambda> gpuDataLambdas = {
        {&GpuContext::SetsortingGpuData, "sortingGpuData"},
        {&GpuContext::SetsparseGpuData, "sparseGpuData"},
        {&GpuContext::SetactivationGpuData, "activationGpuData"},
        {&GpuContext::SetdeltaGpuData, "deltaGpuData"},
        {&GpuContext::SetbanBadWordsGpuData, "banBadWordsGpuData"},
        {&GpuContext::SetbanRepeatNgramGpuData, "banRepeatNgramGpuData"},
        {&GpuContext::SetdecodingGpuData, "decodingGpuData"},
        {&GpuContext::SetbeamSearchPenaltyGpuData, "beamSearchPenaltyGpuData"},
        {&GpuContext::SetonlineSoftmaxBeamsearchGpuData, "onlineSoftmaxBeamsearchGpuData"},
        {&GpuContext::SetstopCriteriaGpuData, "stopCriteriaGpuData"},
        {&GpuContext::SetrmsnormGpuData, "rmsnormGpuData"},
        {&GpuContext::SetquantizationGpuData, "quantizationGpuData"},
        {&GpuContext::SetpreQuantScaleGpuData, "preQuantScaleGpuData"},
        {&GpuContext::SetbeamSearchTopkGpuData, "beamSearchTopkGpuData"},
        {&GpuContext::SetcomputeSeqOffsetsGpuData, "computeSeqOffsetsGpuData"},
        {&GpuContext::SetlayernormalisationGpuData, "layernormalisationGpuData"},
        {&GpuContext::SetlookupGpuData, "lookupGpuData"}
    };

    /// <summary>
    /// Temporary GpuContext and GpuData objects used for copying constants.
    /// </summary>
    GpuContext gpuContext;
    GpuData gpuData{};

    /// <summary>
    /// Iterates through the list of lambda functions and calls them to set GPU data.
    /// </summary>
    for (const auto& lambda : gpuDataLambdas) {
        try {
            lambda.setGpuData(gpuContext, gpuData);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception caught for copying GPU constants " << lambda.name << ": " << e.what() << "\n";
        }
    }
}

/// <summary>
/// Sets the fast math mode for cuBLAS.
/// </summary>
/// <param name="flag">If true, enables fast math mode; otherwise, disables it.</param>
void GpuContext::SetFastMath(bool flag)
{
    try
    {
        // Get the number of CUDA-capable devices.
        int deviceCount;
        cudaError_t cudaError = cudaGetDeviceCount(&deviceCount);

        // Throw an error if no CUDA-capable devices are found.
        if (cudaError != cudaSuccess || deviceCount == 0)
        {
            throw std::runtime_error("No CUDA-compatible GPU found: " + std::string(cudaGetErrorString(cudaError)));
        }

        // Throw an error if the requested number of GPUs exceeds the available number.
        if (NUM_GPUS > deviceCount)
        {
            throw std::runtime_error("Requested number of GPUs (" + std::to_string(NUM_GPUS) +
                ") exceeds available GPUs (" + std::to_string(deviceCount) + ")");
        }

        // Create a vector of threads for setting up each GPU.
        std::vector<std::jthread> gpuThreads;
        gpuThreads.reserve(NUM_GPUS);

        // Create a vector to store cuBLAS handles for each GPU.
        std::vector<cublasHandle_t> cuBLASHandles(NUM_GPUS);

        // Create a latch to synchronize the threads.
        std::latch latch(NUM_GPUS);

        // Start threads for setting up each GPU.
        for (int deviceId = 0; deviceId < NUM_GPUS; ++deviceId)
        {
            gpuThreads.emplace_back([&cuBLASHandles, &latch, deviceId, flag] {
                try
                {
                    // Set the current device.
                    cudaSetDevice(deviceId);

                    // Get the SM major and minor revisions of the current device.
                    int sm_major, sm_minor;
                    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, deviceId);
                    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, deviceId);

                    // Throw an error if the SM revision is less than 8.0.
                    if (sm_major < 8 || (sm_major == 8 && sm_minor < 0))
                    {
                        throw std::runtime_error("GPU SM revision is < 8.0");
                    }

                    // Create a cuBLAS handle.
                    cublasHandle_t cuBLASHandle;
                    cublasCreate(&cuBLASHandle);

                    // Set the math mode for cuBLAS.
                    cublasMath_t mathMode = flag ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
                    cublasSetMathMode(cuBLASHandle, mathMode);

                    // Store the cuBLAS handle.
                    cuBLASHandles[deviceId] = cuBLASHandle;
                }
                catch (const std::exception& e)
                {
                    std::cerr << "GPU " << deviceId << " setup exception: " << e.what() << '\n';
                }
                // Decrement the latch count.
                latch.count_down();
                });
        }

        // Wait for all threads to finish.
        latch.wait();
        // Move the cuBLAS handles to the member variable.
        _cuBLASHandles = std::move(cuBLASHandles);
    }
    catch (const std::exception& e)
    {
        std::cerr << "GpuContext::SetFastMath: " << e.what() << '\n';
    }
}

/// <summary>
/// Shuts down the GPU context.
/// </summary>
void GpuContext::Shutdown() {
    try {
        // Create a lambda function to shut down a library on the GPU.
        auto shutdownLibrary = [this](const char* libraryName, auto destroyFunc, auto& handle, auto successStatus) -> std::future<void> {
            return std::async(std::launch::async, [this, libraryName, destroyFunc, &handle, successStatus]() {
                // Output message indicating the shutdown of the library.
                std::cout << "Shutting down " << libraryName << " on GPU for process " << _device << "\n";

                // Call the destroy function to shut down the library.
                auto status = std::invoke(destroyFunc, handle);

                // Throw an error if the shutdown failed.
                if (status != successStatus) {
                    throw std::runtime_error("Failed to shut down " + std::string(libraryName) + " on GPU for process " + std::to_string(_device) + "\n");
                }

                // Output message indicating the successful shutdown of the library.
                std::cout << libraryName << " shut down on GPU for process " << _device << "\n";
                });
            };

        // Start asynchronous shutdown tasks for cuBLAS, cuDNN, and cuRand.
        auto cuBLASShutdown = shutdownLibrary("cuBLAS", cublasDestroy, _cuBLASHandle, CUBLAS_STATUS_SUCCESS);
        auto cuDNNShutdown = shutdownLibrary("cuDNN", cudnnDestroy, _cuDNNHandle, CUDNN_STATUS_SUCCESS);
        auto cuRandShutdown = shutdownLibrary("cuRand", curandDestroyGenerator, _RNG, CURAND_STATUS_SUCCESS);

        // Wait for the shutdown tasks to complete.
        cuBLASShutdown.wait();
        cuDNNShutdown.wait();
        cuRandShutdown.wait();

        // Reset the CUDA device.
        cudaDeviceReset();

        // Finalize MPI.
        MPI_Finalize();

        // Output message indicating the process finalization.
        std::cout << "Process " << _id << " out of " << _numprocs << " finalized.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error during GPU context shutdown: " << e.what();
    }
}

/// <summary>
/// Sets the Neural Network parameters in the GpuContext.
/// </summary>
/// <param name="pNetwork">Pointer to the Network object containing parameters.</param>
void GpuContext::SetNeuralNetwork(Network* pNetwork)
{
    std::cout << "Setting Neural Network parameters in GpuContext";

    // Check if the Network pointer is valid.
    if (!pNetwork) {
        std::cerr << "Invalid Network pointer provided.";
        return;
    }

    // Get a reference to the GpuData object.
    auto& data = _data;
    // Get a const reference to the Network object.
    const auto& network = *pNetwork;

    try {
        // Copy parameters from the Network object to the GpuData object.
        data._LRN_k = network._LRN_k;
        data._LRN_n = network._LRN_n;
        data._LRN_alpha = network._LRN_alpha;
        data._LRN_beta = network._LRN_beta;
        data._maxout_k = network._maxout_k;
        data._bSparsenessPenalty = network._bSparsenessPenalty;
        data._sparsenessPenalty_p = network._sparsenessPenalty_p;
        data._sparsenessPenalty_beta = network._sparsenessPenalty_beta;
        data._bDenoising = network._bDenoising;
        data._denoising_p = network._denoising_p;

        // Calculate the denoising_q value based on denoising_p.
        if (network._denoising_p != 1.0f) {
            data._denoising_q = 1.0f / (1.0f - network._denoising_p);
        }
        else {
            data._denoising_q = std::numeric_limits<float>::infinity();
        }

        data._deltaBoost_one = network._deltaBoost_one;
        data._deltaBoost_zero = network._deltaBoost_zero;
        data._SMCE_oneTarget = network._SMCE_oneTarget;
        data._SMCE_zeroTarget = network._SMCE_zeroTarget;
        data._SMCE_oneScale = network._SMCE_oneScale;
        data._SMCE_zeroScale = network._SMCE_zeroScale;

        // Check if shuffling indices is allowed in training mode.
        if (data._bShuffleIndices && network._mode == Mode::Training) {
            throw std::runtime_error("Copying constants failed during training mode.");
        }

        // Set the shuffle indices flag.
        data._bShuffleIndices = network._bShuffleIndices && (network._mode == Mode::Training);

        // Copy the shuffle index pointer.
        data._pShuffleIndex = network._pShuffleIndex;

        // Copy the constants for the GPU data.
        CopyConstants();

        std::cout << "Finished setting Neural Network parameters in GpuContext";
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred during parameter setting: " << e.what();
    }
}

/// <summary>
/// Sets the random seed for the GPU.
/// </summary>
/// <param name="seed">The new random seed.</param>
void GpuContext::SetRandomSeed(unsigned long seed)
{
    constexpr unsigned long factor = 76801ull;

    // Set the seed for the cuRand generator.
    curandStatus_t crstatus = curandSetPseudoRandomGeneratorSeed(_RNG, seed + static_cast<unsigned long long>(static_cast<unsigned long>(_device)) * factor);

    // Throw an error if the seed could not be set.
    if (crstatus != CURAND_STATUS_SUCCESS)
    {
        std::ostringstream errorMessage;
        errorMessage << "GpuContext::SetRandomSeed: Failed to set cuRand seed on GPU for process " << _device << ".\n";
        std::cerr << errorMessage.str();
        Shutdown();
        throw std::runtime_error("Failed to set cuRand seed on GPU.");
    }

    // Set the seed for the C++ standard random number generator.
    srand(seed);

    // Output message indicating the seed is set.
    std::ostringstream logMessage;
    logMessage << "GpuContext::SetRandomSeed: Random seed set to " << seed << ".\n";
    std::cout << logMessage.str();
}

/// <summary>
/// Modifies the provided seed using a series of operations.
/// </summary>
/// <param name="seed">The seed to modify.</param>
/// <returns>The modified seed.</returns>
unsigned long GpuContext::ModifySeed(unsigned long seed) const {
    std::random_device rd;
    seed ^= rd();

    std::default_random_engine engine(seed);
    std::uniform_int_distribution<unsigned long> dist(0, 2);

    // Apply a series of modifications to the seed.
    auto modify_seed = [&](unsigned long& s) {
        unsigned long rnd = dist(engine);
        if (rnd == 0) {
            s = (((s * PRIME_A + PRIME_B) | s) ^ PRIME_C) + PRIME_D;
        }
        else if (rnd == 1) {
            s ^= (s << 13);
            s += (s >> 11);
        }
        else {
            s ^= (s >> SHIFT_BITS);
            s += (s << 19);
        }
        s = Mix(s);
        s ^= rd();
        s = ((s << 7) | (s >> (sizeof(s) * 8 - 7))) + (s ^ 0x3FF00FF);
        s ^= ((s >> 21) & 0x12345FF);
        };

    std::ranges::for_each(std::views::iota(0, 30), [&](auto) { modify_seed(seed); });

    seed ^= XOR_MASK_A;

    std::ranges::for_each(std::views::iota(0, 25), [&](auto) {
        seed = Mix(seed);
        seed ^= rd();
        seed ^= ((seed >> 15) & 0x98765432) ^ 0x8D7C1235;
        });

    seed ^= XOR_MASK_B;

    seed = ((seed << 17) | (seed >> (sizeof(seed) * 8 - 17))) ^ XOR_MASK_C;

    std::cout << "Modified seed value: " << seed;

    return seed;
}

#ifdef USE_HIGHWAY

const HWY_FULL(float) df;
const uint32_t end_idx = std::min(bk + block_size, k);
for (uint32_t idx = bk; idx < end_idx; idx += Lanes(df)) {
    const auto aVector = Load(df, &localA[(i - local_start) * k + idx]);
    const auto bVector = Load(df, &transposedB[j * k + idx]);
    const auto mulResult = aVector * bVector;
    sum += mulResult;
}
localC[(i - local_start) * n + j] += GetLane(SumOfLanes(sum));

#else

/// <summary>
/// Implements the verification of SGEMM operation using MPI and OpenMP.
/// </summary>
/// <typeparam name="T">The data type of the matrices.</typeparam>
/// <param name="vA">A vector containing the data for matrix A.</param>
/// <param name="vB">A vector containing the data for matrix B.</param>
/// <param name="vC">A vector containing the data for matrix C (result).</param>
/// <param name="m">The number of rows in matrix A.</param>
/// <param name="k">The number of columns in matrix A and the number of rows in matrix B.</param>
/// <param name="n">The number of columns in matrix B.</param>
/// <param name="tolerance">The tolerance for error verification.</param>
/// <param name="numThreads">The number of OpenMP threads to use (default: -1 for automatic).</param>
/// <param name="printErrors">Whether to print error messages if the verification fails (default: true).</param>
template <typename T>
void verifySGEMMImpl(std::vector<T>& vA, std::vector<T>& vB, std::vector<T>& vC, uint32_t m, uint32_t k, uint32_t n, float tolerance, int numThreads = -1, bool printErrors = true) {
    // Check if the input matrix dimensions match the vector sizes.
    if (vA.size() != m * k || vB.size() != k * n || vC.size() != m * n) {
        throw std::invalid_argument("Input matrix dimensions do not match vector sizes.");
    }

    // Calculate the tolerance squared for error verification.
    const auto toleranceSquared = tolerance * tolerance;

    // Set the number of OpenMP threads if specified.
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }

    // Get the rank and size of the current process in the MPI communicator.
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if MPI is configured with at least two processes.
    if (size < 2) {
        throw std::runtime_error("MPI should be configured with at least two processes.");
    }

    // Calculate the number of rows to be processed by this process.
    const uint32_t local_m = m / size;
    // Calculate the starting row index for this process.
    const uint32_t local_start = rank * local_m;
    // Calculate the ending row index for this process.
    const uint32_t local_end = local_start + local_m;
    // Define the block size for matrix multiplication.
    const uint32_t block_size = 16;

    // Check if the calculated submatrix exceeds the original matrix dimensions.
    if (local_end > m) {
        throw std::runtime_error("Calculated submatrix exceeds original matrix dimensions.");
    }

    // Allocate memory for the submatrices to be processed by this process.
    std::vector<T> localA(local_m * k);
    std::vector<T> localB(k * n);
    std::vector<T> localC(local_m * n);

    // Define a custom MPI datatype for scattering data blocks.
    MPI_Datatype block_type;
    MPI_Type_vector(local_m, k, m, MPI_FLOAT, &block_type);
    MPI_Type_commit(&block_type);
    // Use a shared pointer to automatically free the custom datatype when it's no longer needed.
    std::shared_ptr<void> block_type_guard(nullptr, [&block_type](void*) { MPI_Type_free(&block_type); });

    // Scatter the data for matrix A across all processes.
    MPI_Scatterv(vA.data(), nullptr, nullptr, block_type, localA.data(), local_m * k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Broadcast the data for matrix B to all processes.
    MPI_Bcast(vB.data(), k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Transpose matrix B for efficient multiplication.
    std::vector<T> transposedB(k * n);
    for (size_t idx = 0; idx < vB.size(); ++idx) {
        transposedB[(idx % n) * k + (idx / n)] = vB[idx];
    }

    // Perform matrix multiplication using OpenMP for parallelism.
#pragma omp parallel for
    for (int bi = 0; bi < local_m; bi += block_size) {
        for (uint32_t bj = 0; bj < n; bj += block_size) {
            for (uint32_t bk = 0; bk < k; bk += block_size) {
                // Use prefetching for better performance.
                _mm_prefetch(reinterpret_cast<const char*>(&localA[(bi + block_size) * k + bk]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&transposedB[(bj + block_size) * k + bk]), _MM_HINT_T0);
                for (uint32_t i = bi; i < std::min(bi + block_size, local_m); ++i) {
                    for (uint32_t j = bj; j < std::min(bj + block_size, n); ++j) {
                        // Use prefetching for better performance.
                        _mm_prefetch(reinterpret_cast<const char*>(&localA[(i - bi) * k + bk]), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(&transposedB[j * k + bk]), _MM_HINT_T0);
#ifdef __AVX512F__
                        // Use AVX512 instructions for optimized matrix multiplication.
                        __m512 sum = _mm512_setzero_ps();
                        const uint32_t end_idx = std::min(bk + block_size, k);
                        for (uint32_t idx = bk; idx < end_idx; idx += 16) {
                            __m512 aVector = _mm512_loadu_ps(&localA[(i - bi) * k + idx]);
                            __m512 bVector = _mm512_loadu_ps(&transposedB[j * k + idx]);
                            __m512 mulResult = _mm512_mul_ps(aVector, bVector);
                            sum = _mm512_add_ps(sum, mulResult);
                        }

                        // Calculate the sum of elements using AVX2 instructions.
                        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
                        sum256 = _mm256_hadd_ps(sum256, sum256);
                        sum256 = _mm256_hadd_ps(sum256, sum256);
                        float sum_elements[8];
                        _mm256_storeu_ps(sum_elements, sum256);

                        float temp_sum = 0.0f;
                        for (int k = 0; k < 8; ++k) {
                            temp_sum += sum_elements[k];
                        }

                        localC[(i - bi) * n + j] += temp_sum;
#else
                        // Use a scalar loop for matrix multiplication if AVX512 is not available.
                        localC[(i - bi) * n + j] = 0.0f;
                        for (uint32_t bk = 0; bk < k; ++bk) {
                            localC[(i - bi) * n + j] += localA[(i - bi) * k + bk] * transposedB[j * k + bk];
                        }
#endif
                    }
                }
            }
        }
    }

    // Gather the results from all processes to reconstruct the complete matrix C.
    MPI_Allgather(localC.data(), local_m * n, MPI_FLOAT, vC.data(), local_m * n, MPI_FLOAT, MPI_COMM_WORLD);

    // Verify the results against the expected values.
    if (printErrors && rank == 0) {
        float maxError = *std::max_element(vC.begin(), vC.end(),
            [&localC, &n](const T& a, size_t idx) {
                float diff = a - localC[static_cast<float>(idx) / n * n + idx % n];
                return diff * diff;
            });

        if (maxError > toleranceSquared) {
            std::cerr << "Error: Maximum squared error is above tolerance." << "\n";
        }
    }

    // Output the matrix dimensions.
    if (rank == 0) {
        std::cerr << "Matrix dimensions: " << m << " x " << k << " x " << n << "\n";
    }
}

/// <summary>
/// Verifies the SGEMM operation using GPU buffers.
/// </summary>
/// <param name="pbA">GPU buffer for matrix A.</param>
/// <param name="pbB">GPU buffer for matrix B.</param>
/// <param name="pbC">GPU buffer for matrix C (result).</param>
/// <param name="m">The number of rows in matrix A.</param>
/// <param name="k">The number of columns in matrix A and the number of rows in matrix B.</param>
/// <param name="n">The number of columns in matrix B.</param>
/// <param name="tolerance">The tolerance for error verification (default: 0.000001f).</param>
/// <param name="numThreads">The number of OpenMP threads to use (default: -1 for automatic).</param>
/// <param name="printErrors">Whether to print error messages if the verification fails (default: true).</param>
void verifySGEMM(GpuBuffer<float>* pbA, GpuBuffer<float>* pbB, GpuBuffer<float>* pbC, uint32_t m, uint32_t k, uint32_t n, float tolerance = 0.000001f, int numThreads = -1, bool printErrors = true) {
    // Allocate vectors to store the matrix data.
    std::vector<float> vA(m * k);
    std::vector<float> vB(k * n);
    std::vector<float> vC(m * n);

    // Download the matrix data from the GPU buffers.
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());

    // Initialize MPI.
    MPI_Init(NULL, NULL);

    // Call the implementation function to verify SGEMM.
    verifySGEMMImpl(vA, vB, vC, m, k, n, tolerance, numThreads, printErrors);

    // Finalize MPI.
    MPI_Finalize();
}

#endif
