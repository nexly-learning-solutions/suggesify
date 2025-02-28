#pragma once

#include "../runtime/bufferManager.h"

namespace suggestify::runtime::utils
{

template <typename T>
bool tensorHasInvalid(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);

bool tensorHasInvalid(
    size_t M, size_t K, nvinfer1::DataType type, void const* data, cudaStream_t stream, std::string const& infoStr);

int stallStream(
    char const* name, std::optional<cudaStream_t> stream = std::nullopt, std::optional<int> delay = std::nullopt);

}
