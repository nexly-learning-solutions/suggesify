#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <curand.h>
#include <vector_functions.h>

#ifdef _MSC_VER
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <span>
#include <string>
#include <tuple>
#include <vector>
#else
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#endif

constexpr size_t MEMORY_ALIGNMENT = 32;

constexpr long long int ESCALE = (1ll << 30);
constexpr double ERRORSCALE = static_cast<double>(ESCALE);
constexpr float ERRORSCALEF = static_cast<float>(ESCALE);
constexpr double ONEOVERERRORSCALE = 1.0 / ERRORSCALE;
constexpr float ONEOVERERRORSCALEF = static_cast<float>(1.0 / ERRORSCALE);
constexpr long long int FPSCALE = (1ll << 40);
constexpr long long int DFSCALE = (1ll << 44);

constexpr uint32_t SM_6X_MAXSPARSE = 4608;
constexpr uint32_t SM_6X_MAXSPARSEANALOG = 2304;
constexpr uint32_t SM_5X_MAXSPARSE = 4608;
constexpr uint32_t SM_5X_MAXSPARSEANALOG = 2304;
constexpr uint32_t SM_3X_MAXSPARSE = 2304;
constexpr uint32_t SM_3X_MAXSPARSEANALOG = 1152;

constexpr bool bShadowedOutputBuffers = false;

constexpr uint32_t SM_3X_THREADS_PER_BLOCK = 1024;
constexpr uint32_t SM_5X_THREADS_PER_BLOCK = 1024;
constexpr uint32_t SM_6X_THREADS_PER_BLOCK = 1024;


using uint32_t = unsigned int;
using uint64_t = unsigned long long int;
using int32_t = int;
using int64_t = long long int;

template <typename T, size_t Alignment = MEMORY_ALIGNMENT>
struct AlignedType
{
    alignas(Alignment) T value;
};

using AlignedDouble = AlignedType<double>;
using AlignedULI = AlignedType<uint32_t>;
using AlignedLLI = AlignedType<int64_t>;
using UInt64 = unsigned long long int;

enum class PrecisionMode
{
    DoublePrecision,
    SinglePrecision,
    HalfPrecision
};

constexpr PrecisionMode precisionMode = PrecisionMode::DoublePrecision;

template <PrecisionMode Mode>
struct PrecisionTypes;

template <>
struct PrecisionTypes<PrecisionMode::DoublePrecision>
{
    using NNAccumulator = AlignedDouble;
    using NNDouble = AlignedDouble;
    using FLOAT = AlignedDouble;
    using NNDouble2 = AlignedType<double2>;
    using NNDouble4 = AlignedType<double4>;
    using Float2 = AlignedType<double2>;
    using Float4 = AlignedType<double4>;

    static constexpr MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;
    static constexpr MPI_Datatype MPI_Float = MPI_DOUBLE_PRECISION;
    static constexpr MPI_Datatype MPI_NNACCUMULATOR = MPI_FLOAT;
};

template <>
struct PrecisionTypes<PrecisionMode::SinglePrecision>
{
    using NNAccumulator = float;
    using NNDouble = AlignedDouble;
    using FLOAT = float;
    using NNDouble2 = AlignedType<double2>;
    using NNDouble4 = AlignedType<double4>;
    using Float2 = AlignedType<float2>;
    using Float4 = AlignedType<float4>;

    static constexpr MPI_Datatype MPI_NNDOUBLE = MPI_DOUBLE_PRECISION;
    static constexpr MPI_Datatype MPI_Float = MPI_FLOAT;
    static constexpr MPI_Datatype MPI_NNACCUMULATOR = MPI_LONG_LONG_INT;
};

#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 800)
constexpr int SM_THREADS_PER_BLOCK = 1024;
constexpr int MAX_THREADS_PER_SM = 2048;
#elif (__CUDA_ARCH__ >= 750)
constexpr int SM_THREADS_PER_BLOCK = 1024;
constexpr int MAX_THREADS_PER_SM = 2048;
#elif (__CUDA_ARCH__ >= 600)
constexpr int SM_THREADS_PER_BLOCK = 128;
constexpr int MAX_THREADS_PER_SM = 2048;
#elif (__CUDA_ARCH__ >= 500)
constexpr int SM_THREADS_PER_BLOCK = 128;
constexpr int MAX_THREADS_PER_SM = 2048;
#else
constexpr int SM_THREADS_PER_BLOCK = 128;
constexpr int MAX_THREADS_PER_SM = 1536;
#endif
#else
constexpr int SM_THREADS_PER_BLOCK = 128;
constexpr int MAX_THREADS_PER_SM = 1536;
#endif

#define LAUNCH_BOUNDS() __launch_bounds__(SM_THREADS_PER_BLOCK, 1)

struct GpuData
{
    unsigned int _warpSize;
    unsigned int _warpBits;
    unsigned int _warpMask;
    unsigned long long int* _pAccumulator;
    float _LRN_k;
    int _LRN_n;
    float _LRN_alpha;
    float _LRN_beta;
    int _maxout_k;
    float _deltaBoost_one;
    float _deltaBoost_zero;
    float _SMCE_oneTarget;
    float _SMCE_zeroTarget;
    float _SMCE_oneScale;
    float _SMCE_zeroScale;
    bool _bSparsenessPenalty;
    float _sparsenessPenalty_p;
    float _sparsenessPenalty_beta;
    bool _bDenoising;
    float _denoising_p;
    float _denoising_q;
    bool _bShuffleIndices;
    unsigned int* _pShuffleIndex;
    AlignedULI _deviceMemory;
    uint32_t _maxUint32_t;
    int32_t _maxInt32_t;
    uint64_t _maxUint64_t;
    int64_t _maxInt64_t;
    float _maxFloat;
    float _minFloat;
};

template <typename T>
struct MultiGpuBuffer
{
    std::vector<GpuBuffer<T>> _buffers;
};

class Network;

struct GpuContext
{

    enum SM_VERSION
    {
        SM_3X,
        SM_5X,
        SM_6X
    };

    enum
    {
        PADDING = 32,
        PADDINGBITS = 5,
        PADDINGMASK = ~(0xFFFFFFFFu << PADDINGBITS)
    };

    GpuData _data;
    bool _bECCSupport;
    bool _bCanMapHostMemory;
    AlignedULI _totalMemory;
    AlignedULI _totalCPUMemory;
    AlignedULI _totalGPUMemory;
    bool _bUnifiedMemory;
    SM_VERSION _sm_version;
    unsigned int _sm_major;
    unsigned int _threadsPerBlock;
    unsigned int _warpSize;
    unsigned int _warpBits;
    unsigned int _warpMask;
    int _numprocs;
    int _id;
    int _device;
    uint32_t _maxSparse;
    uint32_t _maxSparseAnalog;

    cublasHandle_t _cuBLASHandle;
    std::vector<cublasHandle_t> _cuBLASHandles;

    curandGenerator_t _RNG;
    cudnnHandle_t _cuDNNHandle;
    Network* _pNetwork;

    std::unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;
    bool _bCPUValidate;
    float _acceptableError;
    bool _bSingleNode;
    bool _bP2P;

    GpuContext();
    ~GpuContext();

    std::tuple<int, int> GetMemoryUsage() const;
    void SetRandomSeed(unsigned long seed);
    unsigned long ModifySeed(unsigned long seed) const;
    void SetNeuralNetwork(Network* pNetwork);
    void SetFastMath(bool flag);
    void Startup(int argc, char** argv);
    void CopyConstants();
    void Shutdown();
    void SetCPUValidate(bool bCPUValidate);

    static unsigned int Pad(unsigned int x)
    {
        return (x + PADDING - 1) & PADDINGMASK;
    }

    void SetsortingGpuData(GpuData const& data);
    void SetsparseGpuData(GpuData const& data);
    void SetactivationGpuData(GpuData const& data);
    void SetdeltaGpuData(GpuData const& data);
    void SetstopCriteriaGpuData(GpuData const& data);
    void SetbanBadWordsGpuData(GpuData const& data);
    void SetbanRepeatNgramGpuData(GpuData const& data);
    void SetdecodingGpuData(GpuData const& data);
    void SetbeamSearchPenaltyGpuData(GpuData const& data);
    void SetonlineSoftmaxBeamsearchGpuData(GpuData const& data);
    void SetrmsnormGpuData(GpuData const& data);
    void SetquantizationGpuData(GpuData const& data);
    void SetpreQuantScaleGpuData(GpuData const& data);
    void SetbeamSearchTopkGpuData(GpuData const& data);
    void SetcomputeSeqOffsetsGpuData(GpuData const& data);
    void SetlayernormalisationGpuData(GpuData const& data);
    void SetlookupGpuData(GpuData const& data);

    static constexpr uint64_t getMemoryFactor()
    {
        return 1024ll;
    }
};

extern struct GpuContext& getGpu();

template <typename T>
class GpuBuffer
{
public:
    GpuBuffer(size_t length, bool bSysMem = false, bool bManaged = false);
    ~GpuBuffer();

    void Allocate();
    void Resize(size_t length);
    void Deallocate();
    void Upload(T const* pBuff = nullptr) const;
    void Download(T* pBuff = nullptr);
    void Copy(T* pBuff);
    size_t GetLength() const;
    size_t GetSize() const;

    // Make _pDevData public
    T* _pDevData;
    size_t _length;

private:
    bool _bSysMem;
    bool _bManaged;
    T* _pSysData;
};

template <typename T>
GpuBuffer<T>::GpuBuffer(size_t length, bool bSysMem, bool bManaged)
    : _length(length)
    , _bSysMem(bSysMem)
    , _bManaged(bManaged)
    , _pSysData(nullptr)
    , _pDevData(nullptr)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::~GpuBuffer()
{
    Deallocate();
}

template <typename T>
void GpuBuffer<T>::Allocate()
{
    cudaError_t status = cudaSuccess;
    cudaStream_t stream;

    status = cudaStreamCreate(&stream);
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA stream creation failed: " << cudaGetErrorString(status) << "\n";
        return;
    }

    size_t allocationSize = _length * sizeof(T);

    std::unique_ptr<std::byte[]> devDataPtr;
    std::unique_ptr<T[]> sysDataPtr;

    auto cudaMallocFn = [&](void** ptr, size_t size)
    {
        status = cudaMalloc(ptr, size);
        return status;
    };

    if (_bManaged)
    {
        _bSysMem = true;
        cudaMallocFn(reinterpret_cast<void**>(&devDataPtr), allocationSize);
    }

    if (status != cudaSuccess)
    {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(status) << "\n";
        cudaStreamDestroy(stream);
        return;
    }
    else if (!_bSysMem)
    {
        sysDataPtr.reset(new T[_length]);
        std::span<T> devDataSpan(reinterpret_cast<T*>(devDataPtr.get()), _length);
        cudaMemcpyAsync(devDataSpan.data(), sysDataPtr.get(), allocationSize, cudaMemcpyHostToDevice, stream);
    }

    _pDevData = reinterpret_cast<T*>(devDataPtr.release());
    if (sysDataPtr)
    {
        _pSysData = sysDataPtr.release();
    }

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

#ifdef MEMTRACKING
    std::cout << "Mem++: " << getGpu().totalGPUMemory << " " << getGpu().totalCPUMemory << "\n";
#endif
}

template <typename T>
void GpuBuffer<T>::Resize(size_t length)
{
    if (length > _length)
    {
        Deallocate();
        _length = length;
        Allocate();
    }
}

template <typename T>
void GpuBuffer<T>::Deallocate()
{
    if (auto status = cudaFree(_pDevData); status != cudaSuccess)
    {
        std::cerr << "GpuBuffer::Deallocate failed (cudaFree) (CUDA error: " << cudaGetErrorString(status) << ")\n";
        throw std::runtime_error("CUDA memory deallocation failed.");
    }
}

template <typename T>
void GpuBuffer<T>::Copy(T* pBuff)
{
    cudaError_t status;
    status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "cudaMemcpy GpuBuffer::Copy failed (CUDA error: " << cudaGetErrorString(status) << ")" << '\n';
        std::exit(1);
    }
}

template <typename T>
void GpuBuffer<T>::Upload(T const* pBuff) const
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
        {
            std::cerr << "cudaMemcpy GpuBuffer::Upload failed (CUDA error: " << cudaGetErrorString(status) << ")"
                      << '\n';
            std::exit(1);
        }
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, _pSysData, _length * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
        {
            std::cerr << "cudaMemcpy GpuBuffer::Upload failed (CUDA error: " << cudaGetErrorString(status) << ")"
                      << '\n';
            std::exit(1);
        }
    }
}

template <typename T>
void GpuBuffer<T>::Download(T* pBuff)
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(pBuff, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
        {
            std::cerr << "cudaMemcpy GpuBuffer::Download failed (CUDA error: " << cudaGetErrorString(status) << ")"
                      << '\n';
            std::exit(1);
        }
    }
    else if (_bSysMem && !_bManaged)
    {
        cudaError_t status;
        status = cudaMemcpy(_pSysData, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
        {
            std::cerr << "cudaMemcpy GpuBuffer::Download failed (CUDA error: " << cudaGetErrorString(status) << ")"
                      << '\n';
            std::exit(1);
        }
    }
}

template <typename T>
size_t GpuBuffer<T>::GetLength() const
{
    return _length;
}

template <typename T>
size_t GpuBuffer<T>::GetSize() const
{
    return _length * sizeof(T);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)

#define std ::printf(f, ...)
#endif