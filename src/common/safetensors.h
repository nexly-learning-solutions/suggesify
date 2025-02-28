
#pragma once
#include "assert.h"
#include "logger.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

namespace suggestify::common::safetensors
{
class INdArray
{
public:
    [[nodiscard]] virtual void const* data() const = 0;
    [[nodiscard]] virtual int ndim() const = 0;
    [[nodiscard]] virtual std::vector<int64_t> const& dims() const = 0;
    [[nodiscard]] virtual nvinfer1::DataType dtype() const = 0;

    [[nodiscard]] nvinfer1::Dims trtDims() const
    {
        nvinfer1::Dims dims;
        dims.nbDims = ndim();
        CHECK(dims.nbDims <= nvinfer1::Dims::MAX_DIMS);
        memset(dims.d, 0, sizeof(dims.d));
        for (int i = 0; i < dims.nbDims; ++i)
        {
            dims.d[i] = this->dims()[i];
        }
        return dims;
    }

    virtual ~INdArray() = default;
};

class ISafeTensor
{
public:
    static std::shared_ptr<ISafeTensor> open(char const* filename);
    virtual std::shared_ptr<INdArray> getTensor(char const* name) = 0;
    virtual std::vector<std::string> keys() = 0;
    virtual ~ISafeTensor() = default;
};

}
