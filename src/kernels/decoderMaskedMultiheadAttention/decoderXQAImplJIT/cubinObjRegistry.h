
#pragma once
#include "cubinObj.h"

#include "compileEngine.h"
#include "serializationUtils.h"
#include "../src/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include <functional>
#include <unordered_map>

namespace suggestify
{
namespace kernels
{
namespace jit
{

// A thread-safe collection of CubinObjs, with caching functionality.
template <typename Key, class Hash = std::hash<Key>>
class CubinObjRegistryTemplate
{
public:
    CubinObjRegistryTemplate() = default;

    CubinObjRegistryTemplate(void const* buffer_, size_t buffer_size)
    {
        size_t remaining_buffer_size = buffer_size;
        uint8_t const* buffer = static_cast<uint8_t const*>(buffer_);
        // First 4 bytes: num of elements.
        uint32_t n = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);

        for (uint32_t i = 0; i < n; ++i)
        {
            uint32_t key_size = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);
            CHECK(key_size <= remaining_buffer_size);
            Key key(buffer, key_size);
            buffer += key_size;
            remaining_buffer_size -= key_size;

            uint32_t obj_size = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);
            CHECK(obj_size <= remaining_buffer_size);
            CubinObj obj(buffer, obj_size);
            buffer += obj_size;
            remaining_buffer_size -= obj_size;

            mMap.insert({key, std::move(obj)});
        }
        CHECK(remaining_buffer_size == 0);
    }

    std::unique_ptr<CubinObjRegistryTemplate<Key, Hash>> clone() const noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto result = std::make_unique<CubinObjRegistryTemplate<Key, Hash>>();
        for (auto const& p : mMap)
        {
            result->mMap.insert(p);
        }
        return result;
    }

    size_t getSerializationSize() const noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        size_t result = sizeof(uint32_t);
        for (auto&& p : mMap)
        {
            result += 2 * sizeof(uint32_t);
            result += p.first.getSerializationSize() + p.second.getSerializationSize();
        }
        return result;
    }

    void serialize(void* buffer_, size_t buffer_size) const noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        size_t remaining_buffer_size = buffer_size;
        uint8_t* buffer = static_cast<uint8_t*>(buffer_);
        uint32_t n = mMap.size();
        writeToBuffer<uint32_t>(n, buffer, remaining_buffer_size);
        for (auto&& p : mMap)
        {
            uint32_t key_size = p.first.getSerializationSize();
            CHECK(key_size <= remaining_buffer_size);
            writeToBuffer<uint32_t>(key_size, buffer, remaining_buffer_size);
            p.first.serialize(buffer, key_size);
            buffer += key_size;
            remaining_buffer_size -= key_size;

            uint32_t obj_size = p.second.getSerializationSize();
            CHECK(obj_size <= remaining_buffer_size);
            writeToBuffer<uint32_t>(obj_size, buffer, remaining_buffer_size);
            p.second.serialize(buffer, obj_size);
            buffer += obj_size;
            remaining_buffer_size -= obj_size;
        }
        CHECK(remaining_buffer_size == 0);
    }

    // Compiles and inserts the cubin if not found in mMap. Does nothing otherwise.
    // When initialize is true, also initialize cubins.
    void insertCubinIfNotExists(Key const& key, CompileEngine* compileEngine, bool initialize)
    {
        CHECK(compileEngine != nullptr);

        std::lock_guard<std::mutex> lock(mMutex);

        auto iter = mMap.find(key);
        if (iter != mMap.end())
        {
            return;
        }

        CubinObj obj = compileEngine->compile();
        if (initialize)
        {
            obj.initialize();
        }
        mMap.insert({key, std::move(obj)});
        return;
    }

    CubinObj* getCubin(Key const& key)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto iter = mMap.find(key);
        if (iter != mMap.end())
        {
            return &iter->second;
        }
        else
        {
            return nullptr;
        }
    }

    // When initialize is true, initialize cubins.
    void merge(CubinObjRegistryTemplate<Key, Hash> const& other, bool initialize)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        for (auto&& p : other.mMap)
        {
            auto [iter, success] = mMap.insert(p);
            if (success && initialize)
            {
                // If insertion takes place, initialize the cubin.
                iter->second.initialize();
            }
        }
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mMap.clear();
    }

private:
    std::unordered_map<Key, CubinObj, Hash> mMap;
    mutable std::mutex mMutex;
};

using CubinObjKey = XQAKernelFullHashKey;
using CubinObjHasher = XQAKernelFullHasher;
using CubinObjRegistry = CubinObjRegistryTemplate<CubinObjKey, CubinObjHasher>;

} // namespace jit
} // namespace kernels
} // namespace suggestify
