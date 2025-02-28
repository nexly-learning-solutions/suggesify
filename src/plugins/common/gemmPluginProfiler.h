#pragma once

#include "pluginUtils.h"
#include "../common/logger.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace suggestify::plugins
{

struct GemmDims
{
    using DimType64 = utils::DimType64;

    DimType64 minM;
    DimType64 maxM;
    DimType64 n;
    DimType64 k;

    GemmDims()
        : minM(-1)
        , maxM(-1)
        , n(-1)
        , k(-1)
    {
    }

    GemmDims(DimType64 minM_, DimType64 maxM_, DimType64 n_, DimType64 k_)
        : minM(minM_)
        , maxM(maxM_)
        , n(n_)
        , k(k_)
    {
    }

    [[nodiscard]] bool isInitialized() const
    {
        return minM >= 0 && maxM >= 0 && n >= 0 && k >= 0;
    }
};

class GemmIdCore
{
public:
    int n;
    int k;
    nvinfer1::DataType dtype;

    GemmIdCore(int n_, int k_, nvinfer1::DataType const& dtype_)
        : n(n_)
        , k(k_)
        , dtype(dtype_)
    {
    }

    GemmIdCore()
        : n(-1)
        , k(-1)
        , dtype(nvinfer1::DataType::kFLOAT)
    {
    }

    bool operator==(GemmIdCore const& id) const
    {
        return isEqual(id);
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIdCore const& id)
    {
        out << "(N;K)=(" << id.n << ";" << id.k << "),";
        out << " type=" << static_cast<int>(id.dtype);
        return out;
    }

protected:
    bool isEqual(GemmIdCore const& id) const
    {
        return n == id.n && k == id.k && dtype == id.dtype;
    }
};

struct GemmIdCoreHash
{
    std::size_t operator()(GemmIdCore const& id) const
    {
        auto h1 = std::hash<int>{}(id.n);
        auto h2 = std::hash<int>{}(id.k);
        auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
        return h1 ^ h2 ^ h3;
    }
};

class GemmIdCublas : public GemmIdCore
{
public:
    bool transA{};
    bool transB{};
    nvinfer1::DataType outputDtype;

    GemmIdCublas(int n_, int k_, nvinfer1::DataType const& dtype_, bool transA_, bool transB_,
        nvinfer1::DataType const& output_dtype_)
        : GemmIdCore(n_, k_, dtype_)
        , transA(transA_)
        , transB(transB_)
        , outputDtype(output_dtype_)
    {
    }

    GemmIdCublas() {}

    bool operator==(GemmIdCublas const& id) const
    {
        return isEqual(id) && transA == id.transA && transB == id.transB && outputDtype == id.outputDtype;
    }

    friend std::ostream& operator<<(std::ostream& out, GemmIdCublas const& id)
    {
        out << "(N;K)=(" << id.n << ";" << id.k << "),";
        out << " type=" << static_cast<int>(id.dtype);
        out << " transA=" << id.transA;
        out << " transB=" << id.transB;
        out << " outputDtype=" << static_cast<int>(id.outputDtype);
        return out;
    }
};

struct GemmIdCublasHash
{
    std::size_t operator()(GemmIdCublas const& id) const
    {
        auto h1 = std::hash<int>{}(id.n);
        auto h2 = std::hash<int>{}(id.k);
        auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
        auto h4 = std::hash<bool>{}(id.transA);
        auto h5 = std::hash<bool>{}(id.transB);
        auto h6 = std::hash<bool>{}(static_cast<int>(id.outputDtype));
        return h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6;
    }
};

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
class GemmPluginProfiler
{
public:
    using MProfileMap = std::unordered_map<int, std::optional<Config>>;
    using MProfileMapPtr = std::shared_ptr<MProfileMap>;

    using reader_lock = std::unique_lock<std::shared_timed_mutex>;
    using writer_lock = std::shared_lock<std::shared_timed_mutex>;

    struct MNKProfileMap
    {
        std::shared_timed_mutex mutex;
        std::unordered_map<GemmIdType, MProfileMapPtr, GemmIdHashType> profileMap;

        bool existsMProfileMap(GemmIdType const& id)
        {
            auto const iter = profileMap.find(id);
            return iter != profileMap.end();
        }

        void createMProfileMap(GemmIdType const& id)
        {
            profileMap[id] = std::make_shared<MProfileMap>();
        }

        MProfileMapPtr getMProfileMap(GemmIdType const& id)
        {
            auto const iter = profileMap.find(id);
            if (iter == profileMap.end())
            {
                std::ostringstream msg;
                msg << "Cannot find ID (" << id << ") in the profile map. Abort.";
                THROW(msg.str());
            }
            return iter->second;
        }
    };

    using MNKProfileMapPtr = std::shared_ptr<MNKProfileMap>;

    GemmPluginProfiler();

    void serialize(char*& buffer, GemmIdType const& gemmId) const;

    void deserialize(char const*& data, GemmDims& dims, GemmIdType const& gemmId);
    size_t getSerializationSize(GemmIdType const& gemmId) const;

    void profileTactics(
        RunnerPtr const& runner, nvinfer1::DataType const& type, GemmDims const& dims, GemmIdType const& gemmId);

    void setSelectionTactics(MNKProfileMapPtr const& map)
    {
        mMNKProfileMap = map;
    }

    void setTmpWorkspaceSizeInBytes(size_t bytes)
    {
        mTmpWorkspaceSizeInBytes = bytes;
    }

    void setSkip(bool skip)
    {
        mSkip = mSkip || skip;
    }

    std::optional<Config> getBestConfig(int m, GemmIdType const& gemmId) const;

    virtual int getMaxProfileM() const;

protected:
    virtual void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) = 0;

    virtual void computeTmpSize(size_t maxM, size_t n, size_t k) = 0;

    virtual bool checkTactic(int m, int n, int k, Config const& tactic) const
    {
        return true;
    }

    virtual std::vector<Config> getTactics(int m, int n, int k) const = 0;

    virtual void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream);

private:
    void allocateTmpData();

    void freeTmpData();

    std::optional<Config> profileTacticsForProblem(int m, int n, int k, std::vector<Config> const& tactics);

    float profileTacticForProblem(int m, int n, int k, Config const& tactic);

    int nextPowerOfTwo(int v) const
    {
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return ++v;
    }

protected:
    RunnerPtr mRunner{nullptr};

    nvinfer1::DataType mType{};

private:
    MNKProfileMapPtr mMNKProfileMap{};

    size_t mTmpWorkspaceSizeInBytes{0};

    char* mWorkspaceTmp{nullptr};

    cudaStream_t mStream;

    GemmDims mDims{};

    bool mSkip{false};
};

template <typename GemmPluginProfilerType>
class GemmPluginProfilerManager
{
public:
    using MNKProfileMap = typename GemmPluginProfilerType::MNKProfileMap;
    using MNKProfileMapPtr = typename GemmPluginProfilerType::MNKProfileMapPtr;
    using GemmPluginProfilerPtr = std::shared_ptr<GemmPluginProfilerType>;

    GemmPluginProfilerManager()
    {
        mMNKProfileMap = std::make_shared<MNKProfileMap>();
    }

    GemmPluginProfilerPtr createGemmPluginProfiler(bool inference, bool skip = false)
    {
        auto profiler = std::make_shared<GemmPluginProfilerType>();
        profiler->setSkip(skip);
        if (!inference)
        {
            profiler->setSelectionTactics(mMNKProfileMap);
        }
        return profiler;
    }

private:
    MNKProfileMapPtr mMNKProfileMap{};
};

}
