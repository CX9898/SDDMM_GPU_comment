#include <iostream>
#include <string>

#include "Matrix.hpp"

const std::string folderPath("../../dataset/");
const std::string fileName = ("nips.mtx");
const std::string filePath = folderPath + fileName;

int main() {
    SparseMatrix<float> matrixS;
    matrixS.initializeFromMatrixMarketFile(filePath);

    std::cout << "row : " << matrixS.row() << std::endl;
    std::cout << "col : " << matrixS.col() << std::endl;
    std::cout << "nnz : " << matrixS.nnz() << std::endl;



    return 0;
}