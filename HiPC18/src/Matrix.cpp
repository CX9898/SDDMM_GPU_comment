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

    if (lineIter < line.size()) {
        std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
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

        if (lineIter < line.size()) {
            std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
        }

        _rowIndex[idx] = row;
        _colIndex[idx] = col;
        _values[idx] = val;

        ++idx;
    }

    inFile.close();

    return true;
}

template<typename T>
bool Matrix<T>::initializeFromSparseMatrix(const SparseMatrix<T> &matrixS) {
    _row = matrixS.row();
    _col = matrixS.col();
    const int size = matrixS.row() * matrixS.col();
    _size = size;
    _matrixOrder = MatrixOrder::row_major;
    const int  ld = matrixS.col();
    _leadingDimension = ld;

    const auto &rowIndexS = matrixS.rowIndex();
    const auto &colIndexS = matrixS.colIndex();
    const auto &valuesS = matrixS.values();

    _values.clear();
    _values.resize(size);
    for(int idx = 0; idx < matrixS.nnz(); ++idx){
        const auto row = rowIndexS[idx];
        const auto col = colIndexS[idx];
        const auto val = valuesS[idx];

        _values[row * ld + col] = val;
    }
}