
#include "lookaheadPoolManager.h"
#include "../common/logger.h"
#include "lookaheadDecodingUtils.h"

namespace suggestify::layers
{

using namespace suggestify::runtime;

void LookaheadPoolManager::setup(SizeType32 guessSetSize)
{
    CHECK(guessSetSize >= 0 && guessSetSize <= mGuessSetSizeMax);
    mGuessSetSize = guessSetSize;
    mTokenMap.clear();
}

void LookaheadPoolManager::insertOne(Key key, TensorConstPtr const& ngram)
{
    if (UNLIKELY(ITensor::volume(ngram->getShape()) == 0 || mGuessSetSize == 0))
    {
        return;
    }

    auto search = mTokenMap.find(key);
    if (search != mTokenMap.end())
    {
        search->second.remove_if(
            [&ngram](TensorConstPtr const& item)
            {
                BufferRange<TokenIdType const> ngramRange(*ngram);
                BufferRange<TokenIdType const> itemRange(*item);
                return std::equal(ngramRange.begin(), ngramRange.end(), itemRange.begin());
            });
        if (mGuessSetSize > 0 && search->second.size() >= mGuessSetSize)
        {
            search->second.pop_front();
        }
        search->second.push_back(ngram);
    }
    else
    {
        mTokenMap.insert({key, std::list<TensorConstPtr>({ngram})});
    }
}

void LookaheadPoolManager::accept(TensorConstPtr const& prompt, SizeType32 level)
{
    SizeType32 length = prompt->getShape().d[0];
    BufferRange<Key const> promptRange(*prompt);
    for (SizeType32 ti = 0; ti + level - 1 < length; ti++)
    {
        auto key = promptRange[ti];
        TensorPtr ngram = BufferManager::cpu(ITensor::makeShape({level - 1}), nvinfer1::DataType::kINT32);
        BufferRange<TokenIdType const> sourceRange(*ITensor::slice(prompt, ti + 1, level - 1));
        BufferRange<TokenIdType> ngramRange(*ngram);
        std::copy(sourceRange.begin(), sourceRange.end(), ngramRange.begin());

        insertOne(key, ngram);
    }
}

std::list<LookaheadPoolManager::TensorConstPtr> LookaheadPoolManager::guess(Key lastToken, SizeType32 guessSize) const
{
    auto search = mTokenMap.find(lastToken);
    if (search != mTokenMap.end())
    {
        auto ngrams = search->second;
        if (ngrams.size() > guessSize)
        {
            auto it = std::prev(ngrams.end(), guessSize);
            return std::list<TensorConstPtr>(it, ngrams.end());
        }
        else
        {
            return ngrams;
        }
    }
    else
    {
        return std::list<TensorConstPtr>();
    }
}

void LookaheadPoolManager::update(TensorConstPtr const& keyTokens, TensorConstPtr const& ngramTokens)
{
    CHECK(keyTokens->getShape().d[0] == ngramTokens->getShape().d[0]);
    BufferRange<Key const> keyRange(*keyTokens);
    auto window = ngramTokens->getShape().d[0];

    for (SizeType32 wi = 0; wi < window; wi++)
    {
        TensorConstPtr source = ITensor::at(ngramTokens, {wi});
        TensorPtr ngram = BufferManager::cpu(source->getShape(), nvinfer1::DataType::kINT32);
        BufferRange<TokenIdType const> sourceRange(*source);
        BufferRange<TokenIdType> ngramRange(*ngram);
        std::copy(sourceRange.begin(), sourceRange.end(), ngramRange.begin());
        insertOne(keyRange[wi], ngram);
    }
}

}
