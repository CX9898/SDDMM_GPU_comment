#include <iostream>
#include <string>

#include "Matrix.hpp"

const std::string folderPath("../../dataset/");
const std::string fileName = ("test1.mtx");
const std::string filePath = folderPath + fileName;

int main() {
    SparseMatrix<float> matrixS;
    matrixS.initializeFromMatrixMarketFile(filePath);

    Matrix<float> matrixS2D;
    matrixS2D.initializeFromSparseMatrix(matrixS);

    return 0;
}