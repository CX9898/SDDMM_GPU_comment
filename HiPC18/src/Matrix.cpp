#include <iostream>
#include <fstream>
#include <string>

#include "Matrix.hpp"

/**
 *
 **/
std::string iterateOneWord(const std::string &line, int &lineIter) {
    const int begin = lineIter;
    while (lineIter < line.size() && line[lineIter] != ' ') {
        ++lineIter;
    }
    const int end = lineIter;
    ++lineIter;

    return line.substr(begin, end - begin);
}

template<>
bool SparseMatrix<float>::initializeFromMatrixMarketFile(const std::string &filePath) {
    std::ifstream inFile;
    inFile.open(filePath, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cout << "Error,Matrix Market file cannot be opened." << std::endl;
        return false;
    }

    std::string line; // Store the data for each line
    getline(inFile, line); // First line does not operate

    getline(inFile, line);
    int lineIter = 0;
    _row = std::stoi(iterateOneWord(line, lineIter));
    _col = std::stoi(iterateOneWord(line, lineIter));
    _nnz = std::stoi(iterateOneWord(line, lineIter));

    if(lineIter < line.size()){
        std::cerr << "Error, Matrix Market file " << line <<  " line format is incorrect!" << std::endl;
    }

    _rowIndex.resize(_nnz);
    _colIndex.resize(_nnz);
    _values.resize(_nnz);

    int idx = 0;
    while (getline(inFile, line)) {
        lineIter = 0;
        const int row = std::stoi(iterateOneWord(line, lineIter));
        const int col = std::stoi(iterateOneWord(line, lineIter));
        const float val = (float) std::stof(iterateOneWord(line, lineIter));

        if(lineIter < line.size()){
            std::cerr << "Error, Matrix Market file " << line <<  " line format is incorrect!" << std::endl;
        }

        _rowIndex[idx] = row;
        _colIndex[idx] = col;
        _values[idx] = val;

        ++idx;
    }

    inFile.close();

    return true;
}
