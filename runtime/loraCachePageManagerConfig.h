
#pragma once

#include "common.h"
#include "iBuffer.h"

#include <NvInferRuntime.h>

#include <ostream>
#include <sstream>
#include <type_traits>

namespace suggestify::runtime
{
class LoraCachePageManagerConfig
{
public:
    explicit constexpr LoraCachePageManagerConfig(runtime::MemoryType memType, nvinfer1::DataType dType,
        SizeType32 totalNumPages, SizeType32 maxPagesPerBlock, SizeType32 slotsPerPage, SizeType32 pageWidth,
        SizeType32 numCopyStreams)
        : mMemoryType(memType)
        , mDataType(dType)
        , mTotalNumPages(totalNumPages)
        , mMaxPagesPerBlock(maxPagesPerBlock)
        , mSlotsPerPage(slotsPerPage)
        , mPageWidth(pageWidth)
        , mInitToZero(false)
    {
    }

    [[nodiscard]] runtime::MemoryType constexpr getMemoryType() const noexcept
    {
        return mMemoryType;
    }

    void constexpr setMemoryType(runtime::MemoryType const& memoryType) noexcept
    {
        mMemoryType = memoryType;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    void constexpr setDataType(nvinfer1::DataType const& dtype) noexcept
    {
        mDataType = dtype;
    }

    [[nodiscard]] SizeType32 constexpr getTotalNumPages() const noexcept
    {
        return mTotalNumPages;
    }

    void constexpr setTotalNumPage(SizeType32 const& totalNumPages) noexcept
    {
        mTotalNumPages = totalNumPages;
    }

    [[nodiscard]] SizeType32 constexpr getMaxPagesPerBlock() const noexcept
    {
        return mMaxPagesPerBlock;
    }

    void constexpr setMaxPagesPerBlock(SizeType32 const& maxPagesPerBlock) noexcept
    {
        mMaxPagesPerBlock = maxPagesPerBlock;
    }

    [[nodiscard]] SizeType32 constexpr getSlotsPerPage() const noexcept
    {
        return mSlotsPerPage;
    }

    void constexpr setSlotsPerPage(SizeType32 const& slotsPerPage) noexcept
    {
        mSlotsPerPage = slotsPerPage;
    }

    [[nodiscard]] SizeType32 constexpr getPageWidth() const noexcept
    {
        return mPageWidth;
    }

    void constexpr setPageWidth(SizeType32 const& pageWidth) noexcept
    {
        mPageWidth = pageWidth;
    }

    [[nodiscard]] bool constexpr getInitToZero() const noexcept
    {
        return mInitToZero;
    }

    void constexpr setInitToZero(bool initToZero) noexcept
    {
        mInitToZero = initToZero;
    }

    [[nodiscard]] SizeType32 constexpr getNumCopyStreams() const noexcept
    {
        return mNumCopyStreams;
    }

    void constexpr setNumCopyStreams(SizeType32 numCopyStreams) noexcept
    {
        mNumCopyStreams = numCopyStreams;
    }

private:
    runtime::MemoryType mMemoryType;
    nvinfer1::DataType mDataType;

    SizeType32 mTotalNumPages;
    SizeType32 mMaxPagesPerBlock;
    SizeType32 mSlotsPerPage;
    SizeType32 mPageWidth;

    SizeType32 mNumCopyStreams = 1;

    bool mInitToZero;
};

inline std::ostream& operator<<(std::ostream& os, LoraCachePageManagerConfig const& c)
{
    os << "{"
       << "memoryType=" << static_cast<typename std::underlying_type<runtime::MemoryType>::type>(c.getMemoryType())
       << " dataType=" << static_cast<typename std::underlying_type<nvinfer1::DataType>::type>(c.getDataType())
       << " totalNumPages=" << c.getTotalNumPages() << " maxPagesPerBlock=" << c.getMaxPagesPerBlock()
       << " slotsPerPage=" << c.getSlotsPerPage() << " pageWidth=" << c.getPageWidth()
       << " initToZero=" << c.getInitToZero() << "}";
    return os;
}

inline std::string to_string(LoraCachePageManagerConfig const& c)
{
    std::stringstream sstream;
    sstream << c;
    return sstream.str();
}
}
