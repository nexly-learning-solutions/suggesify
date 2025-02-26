
#pragma once

#include <NvInferRuntime.h>

namespace suggestify::runtime
{

class TllmLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;

    Severity getLevel();

    void setLevel(Severity level);
};

}
