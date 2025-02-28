
#pragma once

#include "../batch_manager/common.h"
#include "../executor/executor.h"
#include "../runtime/bufferManager.h"
#include "../runtime/iTensor.h"

namespace xgrammar
{
class GrammarMatcher;
class GrammarCompiler;
}

namespace sugesstify::batch_manager
{

class GuidedDecoder
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = sugesstify::runtime::SizeType32;
    using BitmaskT = uint32_t;

    GuidedDecoder(executor::GuidedDecodingConfig const& guidedDecodingConfig, SizeType32 maxNumSequences,
        SizeType32 vocabSizePadded, nvinfer1::DataType logitsDtype, runtime::BufferManager const& runtimeBufferManager);
    void build(ScheduledRequests const& scheduledRequests);
    void execute(ScheduledRequests const& scheduledRequests, runtime::BufferManager const& runtimeBufferManager,
        std::vector<TensorPtr> const& decoderBuffersLogits);

private:
    executor::GuidedDecodingConfig::GuidedDecodingBackend mGuidedDecodingBackend;
    std::vector<std::shared_ptr<xgrammar::GrammarMatcher>> mXGrammarMatchers;
    std::shared_ptr<xgrammar::GrammarCompiler> mXGrammarCompiler;

    SizeType32 mMaxNumSequences;
    SizeType32 mVocabSizePadded;
    SizeType32 mBitmaskSize;
    nvinfer1::DataType mLogitsDtype;

    TensorPtr mLogitsBitmask;
    TensorPtr mLogitsBitmaskHost;
    TensorPtr mLogitsBitmaskPtrVec;
    TensorPtr mLogitsBitmaskPtrVecHost;
    TensorPtr mLogitsPtrVec;
    TensorPtr mLogitsPtrVecHost;

    runtime::BufferManager mCopyBufferManager;
};

}
