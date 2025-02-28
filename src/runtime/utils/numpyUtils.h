
#pragma once

#include "bufferManager.h"
#include "iTensor.h"

#include <string>

namespace suggestify::runtime::utils
{

[[nodiscard]] ITensor::UniquePtr loadNpy(BufferManager const& manager, std::string const& npyFile, MemoryType where);

void saveNpy(BufferManager const& manager, ITensor const& tensor, std::string const& filename);

}
