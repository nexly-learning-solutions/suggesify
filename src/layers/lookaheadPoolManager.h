
#pragma once

#include <list>
#include <unordered_map>

#include "bufferManager.h"
#include "iTensor.h"

namespace suggestify::layers
{

class LookaheadPoolManager
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;
    using Key = runtime::TokenIdType;

    LookaheadPoolManager(runtime::SizeType32 maxG)
        : mGuessSetSizeMax(maxG)
    {
    }

    void setup(runtime::SizeType32 guessSetSize);

    void accept(TensorConstPtr const& prompt, runtime::SizeType32 level);

    std::list<TensorConstPtr> guess(Key lastToken, runtime::SizeType32 guessSize) const;

    void update(TensorConstPtr const& keyTokens, TensorConstPtr const& ngramTokens);

    std::unordered_map<Key, std::list<TensorConstPtr>> const& getMap() const
    {
        return mTokenMap;
    }

private:
    void insertOne(Key key, TensorConstPtr const& ngram);

private:
    std::unordered_map<Key, std::list<TensorConstPtr>> mTokenMap;
    runtime::SizeType32 const mGuessSetSizeMax;
    runtime::SizeType32 mGuessSetSize;
};

}
