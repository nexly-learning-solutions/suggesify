
#pragma once

#include "../runtime/iTensor.h"

#include <string>

namespace sugesstify::batch_manager
{
template <typename TTensor>
class GenericNamedTensor
{
public:
    using TensorPtr = TTensor;

    TensorPtr tensor;
    std::string name;

    GenericNamedTensor() = default;
    ~GenericNamedTensor() = default;

    GenericNamedTensor(TensorPtr _tensor, std::string _name)
        : tensor{std::move(_tensor)}
        , name{std::move(_name)}
    {
    }

    explicit GenericNamedTensor(std::string _name)
        : tensor{}
        , name{std::move(_name)}
    {
    }

    TensorPtr operator()()
    {
        return tensor;
    }

    TensorPtr const& operator()() const
    {
        return tensor;
    }
};

class NamedTensor : public GenericNamedTensor<sugesstify::runtime::ITensor::SharedPtr>
{
public:
    using Base = GenericNamedTensor<sugesstify::runtime::ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;

    NamedTensor(
        nvinfer1::DataType _type, std::vector<int64_t> const& _shape, std::string _name, void const* _data = nullptr);

    NamedTensor(TensorPtr _tensor, std::string _name)
        : Base(std::move(_tensor), std::move(_name)){};

    explicit NamedTensor(std::string _name)
        : Base(std::move(_name)){};

    [[nodiscard]] std::vector<int64_t> serialize() const;

    void serialize(int64_t* out, const size_t totalSize) const;

    [[nodiscard]] size_t serializedSize() const;

    static NamedTensor deserialize(int64_t const* packed);
};
}
