#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "utils.h"

namespace fs = std::filesystem;

char* getCmdOption(char** begin, char** end, std::string const& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return nullptr;
}

bool cmdOptionExists(char** begin, char** end, std::string const& option)
{
    return std::find(begin, end, option) != end;
}

std::string getRequiredArgValue(int argc, char** argv, std::string flag, std::string message, void (*usage)())
{
    if (!cmdOptionExists(argv, argv + argc, flag))
    {
        std::cout << "Error: Missing required argument: " << flag << ": " << message << std::endl;
        usage();
        std::exit(1);
    }
    else
    {
        return std::string(getCmdOption(argv, argv + argc, flag));
    }
}

std::string getOptionalArgValue(int argc, char** argv, std::string flag, std::string defaultValue)
{
    if (!cmdOptionExists(argv, argv + argc, flag))
    {
        return defaultValue;
    }
    else
    {
        return std::string(getCmdOption(argv, argv + argc, flag));
    }
}

bool isArgSet(int argc, char** argv, std::string flag)
{
    return cmdOptionExists(argv, argv + argc, flag);
}

bool fileExists(std::string const& fileName)
{
    std::ifstream stream(fileName);
    return stream.good();
}

bool isNetCDFfile(std::string const& filename)
{
    if (filename.find_last_of(".") == std::string::npos)
    {
        return false;
    }

    return filename.substr(filename.find_last_of(".") + 1) == NETCDF_FILE_EXTENTION;
}

std::vector<std::string>& split(std::string const& s, char delim, std::vector<std::string>& elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(std::string const& s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

bool isDirectory(std::string const& dirname)
{
    return fs::is_directory(dirname);
}

bool isFile(std::string const& filename)
{
    return fs::is_regular_file(filename);
}

int listFiles(std::string const& dirname, bool const recursive, std::vector<std::string>& files)
{
    try
    {
        if (isFile(dirname))
        {
            files.push_back(dirname);
        }
        else if (isDirectory(dirname))
        {
            for (auto const& entry : fs::recursive_directory_iterator(dirname))
            {
                if (recursive && isDirectory(entry.path().string()))
                {
                    listFiles(entry.path().string(), recursive, files);
                }
                else
                {
                    files.push_back(entry.path().string());
                }
            }
            std::sort(files.begin(), files.end());
        }
        else
        {
            return 1;
        }
        return 0;
    }
    catch (fs::filesystem_error const& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}

template <typename Tkey, typename Tval>
bool cmpFirst(std::pair<Tkey, Tval> const& left, std::pair<Tkey, Tval> const& right)
{
    return left.first > right.first;
}

template <typename Tkey, typename Tval>
bool cmpSecond(std::pair<Tkey, Tval> const& left, std::pair<Tkey, Tval> const& right)
{
    return left.second > right.second;
}

template <typename Tkey, typename Tval>
void topKsort(
    Tkey* keys, Tval* vals, int const size, Tkey* topKkeys, Tval* topKvals, int const topK, bool const sortByKey)
{
    if (!keys || !topKkeys || !topKvals)
    {
        std::cout << "null input array" << std::endl;
        std::exit(0);
    }
    std::vector<std::pair<Tkey, Tval>> data(size);
    if (vals)
    {
        for (int i = 0; i < size; i++)
        {
            data[i].first = keys[i];
            data[i].second = vals[i];
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            data[i].first = keys[i];
            data[i].second = i;
        }
    }

    if (sortByKey)
    {
        std::nth_element(data.begin(), data.begin() + topK, data.end(), cmpFirst<Tkey, Tval>);
        std::ranges::sort(data.begin(), data.begin() + topK, cmpFirst<Tkey, Tval>);
    }
    else
    {
        std::nth_element(data.begin(), data.begin() + topK, data.end(), cmpSecond<Tkey, Tval>);
        std::ranges::sort(data.begin(), data.begin() + topK, cmpSecond<Tkey, Tval>);
    }
    for (int i = 0; i < topK; i++)
    {
        topKkeys[i] = data[i].first;
        topKvals[i] = data[i].second;
    }
}

template void topKsort<float, unsigned int>(
    float*, unsigned int*, int const, float*, unsigned int*, int const, bool const);

template void topKsort<float, float>(float*, float*, int const, float*, float*, int const, bool const);