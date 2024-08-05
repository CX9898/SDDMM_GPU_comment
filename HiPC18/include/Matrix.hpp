#pragma  once

#include <string>
#include <vector>

enum MatrixOrder {
  row_major,
  col_major
};

template<typename T>
class SparseMatrix {
 public:
  SparseMatrix() = default;
  ~SparseMatrix() = default;
  bool initializeFromMatrixMarketFile(const std::string &filePath);

  size_t nnz() const {
      return _nnz;
  }

  size_t row() const {
      return _row;
  }
  size_t col() const {
      return _col;
  }

  const std::vector<int> &rowIndex() const {
      return _rowIndex;
  }
  const std::vector<int> &colIndex() const {
      return _colIndex;
  }
  const std::vector<int> &values() const {
      return _values;
  }

 private:
  size_t _nnz;
  size_t _row;
  size_t _col;

  std::vector<int> _rowIndex;
  std::vector<int> _colIndex;
  std::vector<T> _values;
};

template<typename T>
class Matrix {
 public:
  Matrix() = default;
  ~Matrix() = default;

  bool initializeFromSparseMatrix(const SparseMatrix<T> &matrixS);

  size_t size() const {
      return _size;
  }
  MatrixOrder matrixOrder() const {
      return _matrixOrder;
  }
  size_t ld() const {
      return _leadingDimension;
  }
  size_t row() const {
      return _row;
  }
  size_t col() const {
      return  _col;
  }

//  void setSize(size_t size) {
//      _size = size;
//      _values.resize(size);
//  }
//  void setld(size_t ld) {
//      _leadingDimension = ld;
//  }
//  void setRow(size_t row) {
//      _row = row;
//  }
//  void setCol(size_t col) {
//      _row = col;
//  }

  const std::vector<T> &values() const {
      return _values;
  }

 private:
  size_t _row;
  size_t _col;
  size_t _size;
  MatrixOrder _matrixOrder;
  size_t _leadingDimension;

  std::vector<T> _values;
};