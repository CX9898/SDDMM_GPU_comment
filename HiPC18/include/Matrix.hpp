#pragma  once

#include <string>
#include <vector>

template<typename T>
class Matrix {
 public:
  Matrix() = default;
  ~Matrix() = default;

  size_t size() const {
      return _size;
  }
  size_t ld() const {
      return _leadingDimension;
  }

  void setSize(size_t size) {
      _size = size;
      _values.resize(size);
  }
  void setld(size_t ld){
      _leadingDimension = ld;
  }
  void setRow(size_t row) {
      _row = row;
  }
  void setCol(size_t col) {
      _row = col;
  }

  std::vector<T> &values() {
      return _values;
  }

 private:
  size_t _size;
  size_t _leadingDimension;
  size_t _row;
  size_t _col;

  std::vector<T> _values;
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

  // test
  std::vector<int> &rowIndex() {
      return _rowIndex;
  }
  std::vector<int> &colIndex() {
      return _colIndex;
  }
  std::vector<int> &values() {
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






