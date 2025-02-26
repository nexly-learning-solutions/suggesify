
#include "utils/speculativeChoicesUtils.h"
#include <stack>
#include <vector>

namespace suggestify::runtime::utils
{

static SizeType32 constexpr PREFIX_CHUNK_SIZE_BITS = 4;
static SizeType32 constexpr PREFIX_MAX_VALUE = 16;

using TensorPtr = ITensor::SharedPtr;
using Choices = std::vector<std::vector<SizeType32>>;

void copyPackedMask(
    SpeculativeDecodingModule const& speculativeDecodingModule, TensorPtr mask, SizeType32 srcIdx, SizeType32 dstIdx)
{
    auto srcRow = ITensor::slice(mask, srcIdx, 1);
    auto dstRow = ITensor::slice(mask, dstIdx, 1);
    std::memcpy(bufferCast<SizeType32>(*dstRow), bufferCast<SizeType32>(*srcRow),
        speculativeDecodingModule.getNumPackedMasks() * sizeof(SizeType32));
}

void setOnePackedMask(
    SpeculativeDecodingModule const& speculativeDecodingModule, TensorPtr mask, SizeType32 row, SizeType32 col)
{
    auto const maskIdx = static_cast<SizeType32>(col / 32);
    auto const bitIdx = col % 32;
    auto setMask = 1 << bitIdx;
    bufferCast<SizeType32>(*mask)[row * speculativeDecodingModule.getNumPackedMasks() + maskIdx] |= setMask;
}

void computePathsAndMask(SpeculativeDecodingModule const& speculativeDecodingModule, std::vector<TreeNode> const& tree,
    TensorPtr packedMask, TensorPtr paths)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto pathsPtr = paths ? bufferCast<SizeType32>(*paths) : nullptr;

    if (pathsPtr)
    {
        std::fill(pathsPtr, pathsPtr + paths->getSize(), -1);
    }
    if (packedMask)
    {
        std::fill(bufferCast<SizeType32>(*packedMask), bufferCast<SizeType32>(*packedMask) + packedMask->getSize(), 0);
    }

    SizeType32 numPaths = 0;
    for (auto const& node : tree)
    {
        if (node.childLinearIndices.size() == 0)
        {
            numPaths++;
        }
    }

    std::stack<SizeType32> stack;
    stack.push(0);

    SizeType32 pathIdx = 0;

    while (!stack.empty())
    {
        auto const ci = stack.top();
        stack.pop();

        auto const& node = tree[ci];

        if (packedMask)
        {
            if (node.nodeId != -1)
            {
                copyPackedMask(speculativeDecodingModule, packedMask, node.parentLinearIdx, ci);
            }
            setOnePackedMask(speculativeDecodingModule, packedMask, ci, ci);
        }

        if (node.childLinearIndices.size() == 0)
        {
            SizeType32 nodeIdx = ci;
            while (tree[nodeIdx].nodeId != -1)
            {
                auto const& curNode = tree[nodeIdx];
                if (pathsPtr)
                {
                    pathsPtr[(numPaths - 1 - pathIdx) * speculativeDecodingModule.getMaxPathLen() + curNode.depth]
                        = curNode.linearIdx;
                }
                nodeIdx = curNode.parentLinearIdx;
            }
            if (pathsPtr)
            {
                pathsPtr[(numPaths - 1 - pathIdx) * speculativeDecodingModule.getMaxPathLen() + 0] = 0;
            }
            pathIdx++;
        }
        for (auto const& childLinearIdx : node.childLinearIndices)
        {
            stack.push(childLinearIdx);
        }
    }
    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

uint64_t computePrefix(std::vector<SizeType32> const& vec, SizeType32 len)
{
    SizeType32 constexpr BITS_PER_BYTE = 8;
    CHECK_WITH_INFO(static_cast<SizeType32>(sizeof(uint64_t)) * BITS_PER_BYTE / PREFIX_CHUNK_SIZE_BITS >= len,
        "Provided choices have depth (%d) larger than Prefix can fit (%ld).", len,
        sizeof(uint64_t) * BITS_PER_BYTE / PREFIX_CHUNK_SIZE_BITS);

    uint64_t prefix = 0;
    for (SizeType32 ci = 0; ci < len; ++ci)
    {
        auto val = vec[ci];
        CHECK_WITH_INFO(val <= PREFIX_MAX_VALUE,
            "Provided choices have too large node degree (%d). Larger than Prefix can fit (%d).", val,
            PREFIX_MAX_VALUE);
        prefix |= (vec[ci] << PREFIX_CHUNK_SIZE_BITS * (len - 1 - ci));
    }
    return prefix;
}

void dumpChoices(Choices const& choices, std::vector<SizeType32> const& indices)
{
    std::stringstream ss;
    ss << "Choices = [";
    for (size_t ci = 0; ci < indices.size(); ++ci)
    {
        auto const idx = indices[ci];
        auto const& choice = choices[idx];
        ss << "[";
        for (size_t vi = 0; vi < choice.size(); ++vi)
        {
            ss << choice[vi];
            if (vi < choice.size() - 1)
            {
                ss << ", ";
            }
        }
        ss << "]";
        if (ci < indices.size() - 1)
        {
            ss << ", ";
        }
    }
    ss << "]" << std::endl;
    LOG_DEBUG(ss.str().c_str());
}

void checkNumNonLeafNodesPerLayer(std::vector<TreeNode> const& tree, SizeType32 maxNonLeafNodesPerLayer)
{
    std::unordered_map<SizeType32, SizeType32> nonLeavesPerLayer;
    for (auto const& node : tree)
    {
        if (node.childLinearIndices.size() > 0)
        {
            nonLeavesPerLayer[node.depth]++;
        }
    }
    for (auto const& [depth, numNodes] : nonLeavesPerLayer)
    {
        CHECK_WITH_INFO(numNodes <= maxNonLeafNodesPerLayer,
            "Choices tree at level %d has %d non leaf nodes, while only %d are allowed.", depth, numNodes,
            maxNonLeafNodesPerLayer);
    }
}

SizeType32 initTensorsFromChoices(SpeculativeDecodingModule const& speculativeDecodingModule, Choices const& choices,
    std::vector<SizeType32>& topKs, TensorPtr generationInputLengths, TensorPtr positionOffsets, TensorPtr treeIds,
    TensorPtr paths, TensorPtr packedMask, std::optional<SizeType32> maxNonLeafNodesPerLayer)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const numChoices = static_cast<SizeType32>(choices.size());

    std::vector<SizeType32> sortedIndices(numChoices);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::vector<uint64_t> prefixes(numChoices);
    for (SizeType32 ci = 0; ci < numChoices; ++ci)
    {
        auto const& choice = choices[ci];
        prefixes[ci] = computePrefix(choice, choice.size());
    }

    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&prefixes, &choices](SizeType32 const& a, SizeType32 const& b)
        {
            auto const aSize = choices[a].size();
            auto const bSize = choices[b].size();
            return aSize < bSize || (aSize == bSize && prefixes[a] < prefixes[b]);
        });

    topKs.resize(speculativeDecodingModule.getMaxDraftPathLen(), 0);
    auto generationInputLengthsPtr = generationInputLengths ? bufferCast<SizeType32>(*generationInputLengths) : nullptr;
    auto positionOffsetsPtr = positionOffsets ? bufferCast<SizeType32>(*positionOffsets) : nullptr;
    auto treeIdsPtr = treeIds ? bufferCast<SizeType32>(*treeIds) : nullptr;

    if (generationInputLengthsPtr)
    {
        std::fill(generationInputLengthsPtr, generationInputLengthsPtr + generationInputLengths->getSize(),
            speculativeDecodingModule.getMaxDecodingTokens());
    }
    if (positionOffsetsPtr)
    {
        std::fill(positionOffsetsPtr, positionOffsetsPtr + positionOffsets->getSize(), -1);
    }
    if (treeIdsPtr)
    {
        std::fill(treeIdsPtr, treeIdsPtr + treeIds->getSize(), -1);
    }

    dumpChoices(choices, sortedIndices);

    std::vector<TreeNode> tree(choices.size() + 1);
    auto& rootNode = tree[0];
    rootNode.depth = 0;
    rootNode.nodeId = -1;

    SizeType32 depth = 1;
    SizeType32 maxTopK = 0;
    SizeType32 globalNodeInTreeIdx = 0;
    std::unordered_map<uint64_t, SizeType32> prevPrefixToLinearIdxMap;
    std::unordered_map<uint64_t, SizeType32> curPrefixToLinearIdxMap;

    prevPrefixToLinearIdxMap[0] = 0;
    if (positionOffsetsPtr)
    {
        positionOffsetsPtr[0] = 0;
    }

    CHECK(numChoices <= speculativeDecodingModule.getMaxDecodingDraftTokens());

    for (SizeType32 ci = 0; ci < numChoices; ++ci)
    {
        auto const index = sortedIndices[ci];
        auto const& choice = choices[index];
        auto const curDepth = static_cast<SizeType32>(choice.size());

        if (curDepth != depth)
        {
            CHECK(depth + 1 == curDepth);
            CHECK_WITH_INFO(curDepth <= speculativeDecodingModule.getMaxDraftPathLen(),
                "Choices require larger maxPathLen than the engine was built with.");
            topKs[depth - 1] = maxTopK;

            globalNodeInTreeIdx += maxTopK;

            prevPrefixToLinearIdxMap = curPrefixToLinearIdxMap;

            maxTopK = 0;
            curPrefixToLinearIdxMap.clear();

            depth++;
        }

        TreeNode node;
        node.depth = depth;
        node.linearIdx = ci + 1;
        node.nodeId = choice.back();

        curPrefixToLinearIdxMap[prefixes[index]] = node.linearIdx;

        auto const parentPrefix = computePrefix(choice, choice.size() - 1);

        node.parentLinearIdx = prevPrefixToLinearIdxMap[parentPrefix];

        maxTopK = std::max(maxTopK, node.nodeId + 1);

        if (positionOffsetsPtr)
        {
            positionOffsetsPtr[node.linearIdx] = depth;
        }
        if (treeIdsPtr)
        {
            treeIdsPtr[node.linearIdx - 1] = globalNodeInTreeIdx + node.nodeId;
        }

        tree[node.linearIdx] = node;
    }

    topKs[depth - 1] = maxTopK;

    for (SizeType32 ci = 0; ci < numChoices + 1; ++ci)
    {
        auto& node = tree[ci];
        if (node.nodeId != -1)
        {
            tree[node.parentLinearIdx].childLinearIndices.push_back(ci);
        }
    }

    if (maxNonLeafNodesPerLayer)
    {
        checkNumNonLeafNodesPerLayer(tree, maxNonLeafNodesPerLayer.value());
    }

    computePathsAndMask(speculativeDecodingModule, tree, packedMask, paths);

    LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return depth;
}
}
