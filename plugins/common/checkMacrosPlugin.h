#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"

namespace suggestify::plugins
{

void logError(char const* msg, char const* file, char const* fn, int line);

void caughtError(std::exception const& e);

}
