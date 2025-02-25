#include <algorithm>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DataReader.h"

namespace
{
[[nodiscard]] auto splitKeyVector(std::string const& line, char keyValueDelimiter)
{
    if (auto const keyValDelimIndex = line.find_first_of(keyValueDelimiter); keyValDelimIndex != std::string::npos)
    {
        return std::pair<std::string, std::string>{line.substr(0, keyValDelimIndex), line.substr(keyValDelimIndex + 1)};
    }
    else
    {
        return std::pair<std::string, std::string>{"", ""};
    }
}

[[nodiscard]] std::vector<float> parseVector(std::string const& vectorStr, char vectorDelimiter)
{
    std::vector<float> vector;
    std::stringstream vectorStrStream(vectorStr);
    std::string elementStr;

    while (std::getline(vectorStrStream, elementStr, vectorDelimiter))
    {
        try
        {
            vector.push_back(std::stof(elementStr));
        }
        catch (std::exception const& e)
        {
            std::stringstream msg;
            msg << "Error parsing vector element: " << elementStr;
            throw std::invalid_argument(msg.str());
        }
    }

    return vector;
}
}

uint32_t DataReader::getRows() const
{
    return rows;
}

int DataReader::getColumns() const
{
    return columns;
}

TextFileDataReader::TextFileDataReader(std::string const& fileName, char keyValueDelimiter, char vectorDelimiter)
    : fileName(fileName)
    , fileStream(fileName, std::ios_base::in)
    , keyValueDelimiter(keyValueDelimiter)
    , vectorDelimiter(vectorDelimiter)
{
    findDataDimensions();
}

void TextFileDataReader::findDataDimensions()
{
    std::fstream fs(fileName, std::ios_base::in);

    rows = 0;
    columns = 0;

    std::string line;
    while (std::getline(fs, line))
    {
        if (line.empty())
        {
            continue;
        }

        ++rows;

        auto const [key, vectorStr] = splitKeyVector(line, keyValueDelimiter);

        if (key.empty() || vectorStr.empty())
        {
            std::stringstream msg;
            msg << "In file: " << fileName << "#" << rows << ". Malformed line. Key-value delimiter ["
                << keyValueDelimiter << "] not found in: " << line;
            throw std::invalid_argument(msg.str());
        }

        if (columns == 0)
        {
            columns = static_cast<int>(parseVector(vectorStr, vectorDelimiter).size());
        }
        else if (columns != static_cast<int>(parseVector(vectorStr, vectorDelimiter).size()))
        {
            std::stringstream msg;
            msg << "In file: " << fileName << "#" << rows
                << ". Inconsistent number of columns detected. Expected: " << columns
                << " Actual: " << parseVector(vectorStr, vectorDelimiter).size();
            throw std::invalid_argument(msg.str());
        }
    }

    fs.close();
}

bool TextFileDataReader::readRow(std::string* key, float* vector)
{
    std::string line;
    if (std::getline(fileStream, line))
    {
        auto const [keyStr, vectorStr] = splitKeyVector(line, keyValueDelimiter);
        *key = keyStr;

        auto const parsedVector = parseVector(vectorStr, vectorDelimiter);

        if (parsedVector.size() != static_cast<size_t>(columns))
        {
            std::stringstream msg;
            msg << "Inconsistent number of columns in row: " << parsedVector.size() << ", expected: " << columns;
            throw std::invalid_argument(msg.str());
        }

        std::copy_n(parsedVector.begin(), columns, vector);

        return true;
    }
    else
    {
        return false;
    }
}

TextFileDataReader::~TextFileDataReader()
{
    fileStream.close();
}