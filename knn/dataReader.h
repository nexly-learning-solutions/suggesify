#ifndef DATAREADER_H_
#define DATAREADER_H_

#include <fstream>

class DataReader
{

  public:

    virtual bool readRow(std::string *key, float *vector) = 0;

    uint32_t getRows() const;

    int getColumns() const;

    virtual ~DataReader()
    {
    }

  protected:
    uint32_t rows;
    int columns;
};

class TextFileDataReader: public DataReader
{

  public:
    TextFileDataReader(std::string const& fileName, char keyValueDelimiter = '\t', char vectorDelimiter = ' ');

    void findDataDimensions();

    bool readRow(std::string *key, float *vector);



    ~TextFileDataReader();

  private:
    std::string fileName;
    std::fstream fileStream;
    char keyValueDelimiter;
    char vectorDelimiter;
};

#endif
