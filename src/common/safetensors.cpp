
#include "safetensors.h"
#include "nlohmann/json.hpp"
#include "assert.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace suggestify::common::safetensors
{
using nvinfer1::DataType;

static DataType convertDataTypeStrToEnum(std::string const& str)
{
    if (str == "BOOL")
        return DataType::kBOOL;
    if (str == "I8")
        return DataType::kINT8;
    if (str == "I32")
        return DataType::kINT32;
    if (str == "I64")
        return DataType::kINT64;
    if (str == "U8")
        return DataType::kUINT8;
    if (str == "F16")
        return DataType::kHALF;
    if (str == "F32")
        return DataType::kFLOAT;
    if (str == "BF16")
        return DataType::kBF16;
    if (str == "F8_E4M3")
        return DataType::kFP8;
    THROW("Unsupported data type: " + str);
}

class SafeTensorArray : public INdArray
{
    std::vector<int64_t> mShape;
    mutable std::unique_ptr<std::byte[]> mData;
    DataType mDataType;
    int64_t mOffsetBegin;
    int64_t mOffsetEnd;
    std::shared_ptr<std::ifstream> mFs;

public:
    SafeTensorArray(std::shared_ptr<std::ifstream> const& fs, std::string const& dtypeStr,
        std::vector<int64_t> const& shape, int64_t offsetBegin, int64_t offsetEnd)
        : mFs(fs)
        , mShape(shape)
        , mDataType(convertDataTypeStrToEnum(dtypeStr))
        , mOffsetBegin(offsetBegin)
        , mOffsetEnd(offsetEnd)
    {
    }

    [[nodiscard]] void const* data() const override
    {
        if (!mData)
        {
            mFs->seekg(mOffsetBegin);
            int64_t size = mOffsetEnd - mOffsetBegin;
            mData.reset(new std::byte[size]);
            mFs->read(reinterpret_cast<char*>(mData.get()), size);
        }

        return mData.get();
    }

    [[nodiscard]] int ndim() const override
    {
        return static_cast<int>(mShape.size());
    }

    [[nodiscard]] std::vector<int64_t> const& dims() const override
    {
        return mShape;
    }

    [[nodiscard]] DataType dtype() const override
    {
        return mDataType;
    }
};

class SafeTensor : public ISafeTensor
{
    int64_t mJsonSize;
    std::map<std::string, std::string> mMetadata;
    std::map<std::string, nlohmann::basic_json<>> mTensorInfo;
    std::shared_ptr<std::ifstream> mFs;

public:
    SafeTensor(char const* filename)
        : mFs(new std::ifstream(filename, std::ios::binary))
    {
        if (!mFs->is_open())
        {
            THROW("Failed to open file: " + std::string(filename));
        }
        mFs->read(reinterpret_cast<char*>(&mJsonSize), sizeof(mJsonSize));
        std::vector<char> jsonBuffer(mJsonSize);
        mFs->read(jsonBuffer.data(), mJsonSize);
        nlohmann::json attributes = nlohmann::json::parse(jsonBuffer);
        for (auto const& [key, value] : attributes.items())
        {
            if (key == "__metadata__")
            {
                mMetadata = value;
            }
            else
            {
                mTensorInfo[key] = value;
            }
        }
    }

    std::vector<std::string> keys() override
    {
        std::vector<std::string> result;
        result.reserve(mTensorInfo.size());
        for (auto const& [key, value] : mTensorInfo)
        {
            result.push_back(key);
        }
        return result;
    }

    std::shared_ptr<INdArray> getTensor(char const* name) override
    {
        auto it = mTensorInfo.find(name);
        if (it != mTensorInfo.end())
        {
            auto const& value = it->second;
            int64_t offset = mJsonSize + sizeof(mJsonSize);
            return std::make_shared<SafeTensorArray>(mFs, value["dtype"], value["shape"],
                static_cast<int64_t>(value["data_offsets"][0]) + offset,
                static_cast<int64_t>(value["data_offsets"][1]) + offset);
        }
        THROW("Tensor not found: " + std::string(name));
    }
};

std::shared_ptr<ISafeTensor> ISafeTensor::open(char const* filename)
{
    return std::make_shared<SafeTensor>(filename);
}
}
